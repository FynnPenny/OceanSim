# Omniverse Import
import carb
import numpy as np
import omni.replicator.core as rep
import omni.ui as ui
import warp as wp
import yaml
from omni.replicator.core.scripts.functional import write_image

# Custom import
from isaacsim.oceansim.utils.camera_base import OceanSimCameraBase
from isaacsim.oceansim.utils.UWrenderer_utils import (
    UW_depth_turbidity_attenuator,
    UW_render,
)


class UW_Camera(OceanSimCameraBase):

    def __init__(
        self,
        prim_path,
        name="UW_Camera",
        frequency=None,
        dt=None,
        resolution=None,
        position=None,
        orientation=None,
        translation=None,
        render_product_path=None,
    ):
        """Initialize an underwater camera sensor.

        Args:
            prim_path (str): prim path of the Camera Prim to encapsulate or create.
            name (str, optional): shortname to be used as a key by Scene class.
                                    Note: needs to be unique if the object is added to the Scene.
                                    Defaults to "UW_Camera".
            frequency (Optional[int], optional): Frequency of the sensor (i.e: how often is the data frame updated).
                                                Defaults to None.
            dt (Optional[str], optional): dt of the sensor (i.e: period at which a the data frame updated). Defaults to None.
            resolution (Optional[Tuple[int, int]], optional): resolution of the camera (width, height). Defaults to None.
            position (Optional[Sequence[float]], optional): position in the world frame of the prim. shape is (3, ).
                                                        Defaults to None, which means left unchanged.
            translation (Optional[Sequence[float]], optional): translation in the local frame of the prim
                                                            (with respect to its parent prim). shape is (3, ).
                                                            Defaults to None, which means left unchanged.
            orientation (Optional[Sequence[float]], optional): quaternion orientation in the world/ local frame of the prim
                                                            (depends if translation or position is specified).
                                                            quaternion is scalar-first (w, x, y, z). shape is (4, ).
                                                            Defaults to None, which means left unchanged.
            render_product_path (str): path to an existing render product, will be used instead of creating a new render product
                                    the resolution and camera attached to this render product will be set based on the input arguments.
                                    Note: Using same render product path on two Camera objects with different camera prims, resolutions is not supported
                                    Defaults to None
        """
        self._name = name
        self._prim_path = prim_path
        self._res = resolution
        self._writing = False
        # GPU-resident output buffers, reused across frames
        self._uw_image = None
        self._degraded_depth = None
        # Host copies, only filled by subclasses that publish frames (e.g. UW_Camera_ROS)
        self._uw_frame = None
        self._degraded_depth_frame = None
        self._camera_k = None

        super().__init__(
            prim_path,
            name,
            frequency,
            dt,
            resolution,
            position,
            orientation,
            translation,
            render_product_path,
        )

    def initialize(
        self,
        UW_param: np.ndarray = np.array(
            [0.0, 0.31, 0.24, 0.05, 0.05, 0.2, 0.05, 0.05, 0.05]
        ),
        depth_noise_sigma: float = 0.01,
        max_range: float = 20.0,
        viewport: bool = True,
        writing_dir: str = None,
        UW_yaml_path: str = None,
        physics_sim_view=None,
    ):
        """Configure underwater rendering properties and initialize pipelines.

        Args:
            UW_param (np.ndarray, optional): Underwater parameters array:
                [0:3] - Backscatter value (RGB)
                [3:6] - Attenuation coefficients (RGB)
                [6:9] - Backscatter coefficients (RGB)
                Defaults to typical coastal water values.
            depth_noise_sigma (float, optional): Standard deviation of Gaussian
                depth noise applied after turbidity attenuation. Defaults to 0.01.
            max_range (float, optional): Maximum depth range for turbidity
                attenuation. Defaults to 20.0.
            viewport (bool, optional): Enable viewport visualization. Defaults to True.
            writing_dir (str, optional): Directory to save rendered images. Defaults to None.
            UW_yaml_path (str, optional): Path to YAML file with water properties. Defaults to None.
            physics_sim_view (_type_, optional): _description_. Defaults to None.
        """
        self._id = 0
        self._viewport = viewport
        self._device = wp.get_preferred_device()
        self._depth_noise_sigma = wp.float32(depth_noise_sigma)
        self._max_range = max_range
        super().initialize(physics_sim_view)

        if UW_yaml_path is not None:
            with open(UW_yaml_path, "r") as file:
                try:
                    yaml_content = yaml.safe_load(file)
                    self._backscatter_value = wp.vec3f(
                        *yaml_content["backscatter_value"]
                    )
                    self._atten_coeff = wp.vec3f(*yaml_content["atten_coeff"])
                    self._backscatter_coeff = wp.vec3f(
                        *yaml_content["backscatter_coeff"]
                    )
                    carb.log_info(
                        f"[{self._name}] On {str(self._device)}. Using loaded render parameters: {yaml_content}"
                    )
                except yaml.YAMLError as exc:
                    carb.log_error(f"[{self._name}] Error reading YAML file: {exc}")
        else:
            self._backscatter_value = wp.vec3f(*UW_param[0:3])
            self._atten_coeff = wp.vec3f(*UW_param[6:9])
            self._backscatter_coeff = wp.vec3f(*UW_param[3:6])
            carb.log_info(
                f"[{self._name}] On {str(self._device)}. Using default render parameters."
            )

        self._rgba_annot = rep.AnnotatorRegistry.get_annotator(
            "LdrColor", device=str(self._device)
        )
        self._depth_annot = rep.AnnotatorRegistry.get_annotator(
            "distance_to_camera", device=str(self._device)
        )

        self._rgba_annot.attach(self._render_product_path)
        self._depth_annot.attach(self._render_product_path)

        if self._viewport:
            self.make_viewport()

        if writing_dir is not None:
            self._writing = True
            self._writing_backend = rep.BackendDispatch(
                {"paths": {"out_dir": writing_dir}}
            )

        carb.log_info(
            f"[{self._name}] Initialized successfully. Data writing: {self._writing}"
        )

    def render(self):
        """Process and display a single frame with underwater effects.

        Returns:
            bool: True if a frame was rendered, False if the annotators had no data yet.

        Note:
            - Updates viewport display if enabled
            - Saves image to disk if writing_dir was specified
            - Output stays on the GPU; see get_uw_image() / get_degraded_depth()
        """
        raw_rgba = self._rgba_annot.get_data()
        depth = self._depth_annot.get_data()
        if raw_rgba.size == 0:
            return False

        # Both kernels write every pixel, so the output buffers are allocated once and reused
        # instead of being zero-filled every frame.
        if self._uw_image is None or self._uw_image.shape != raw_rgba.shape:
            self._uw_image = wp.empty_like(raw_rgba)
            self._degraded_depth = wp.empty_like(depth)
        uw_image = self._uw_image
        degraded_depth = self._degraded_depth

        wp.launch(
            dim=np.flip(self.get_resolution()),
            kernel=UW_render,
            inputs=[
                raw_rgba,
                depth,
                self._backscatter_value,
                self._atten_coeff,
                self._backscatter_coeff,
            ],
            outputs=[uw_image],
        )

        wp.launch(
            dim=np.flip(self.get_resolution()),
            kernel=UW_depth_turbidity_attenuator,
            inputs=[
                raw_rgba,
                depth,
                self._max_range,
                self._backscatter_value,
                self._atten_coeff,
                self._backscatter_coeff,
                self._depth_noise_sigma,
                int(self._id),
            ],
            outputs=[degraded_depth],
        )

        if self._viewport:
            self._provider.set_bytes_data_from_gpu(
                uw_image.ptr, self.get_resolution()
            )
        if self._writing:
            # The writer runs asynchronously, so hand it a copy: uw_image is overwritten next frame.
            self._writing_backend.schedule(
                write_image, path=f"UW_image_{self._id}.png", data=wp.clone(uw_image)
            )
            carb.log_info(
                f"[{self._name}] [{self._id}] Rendered image saved to {self._writing_backend.output_dir}"
            )

        self._id += 1
        return True

    def get_uw_image(self) -> wp.array:
        """Latest underwater-rendered RGBA frame, GPU-resident (None before the first frame)."""
        return self._uw_image

    def get_degraded_depth(self) -> wp.array:
        """Latest turbidity-degraded depth frame, GPU-resident (None before the first frame)."""
        return self._degraded_depth

    def _build_pointcloud(self, stride: int = 4) -> np.ndarray:
        """Build a downsampled XYZ point cloud from the degraded depth frame.

        Returns:
            np.ndarray: Contiguous float32 array of shape (N, 3), or empty (0, 3)
            if intrinsics or depth are unavailable.
        """
        if self._camera_k is None or self._degraded_depth_frame is None:
            return np.empty((0, 3), dtype=np.float32)

        fx = self._camera_k[0, 0]
        fy = self._camera_k[1, 1]
        cx = self._camera_k[0, 2]
        cy = self._camera_k[1, 2]
        height, width = self._degraded_depth_frame.shape
        sampled_depth = self._degraded_depth_frame[::stride, ::stride]
        u, v = np.meshgrid(
            np.arange(0, width, stride, dtype=np.float32),
            np.arange(0, height, stride, dtype=np.float32),
        )
        x = (u - cx) / fx
        y = (v - cy) / fy
        norm = np.sqrt(x * x + y * y + 1.0)
        valid = np.isfinite(sampled_depth) & (sampled_depth > 0.0)
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float32)

        scale = sampled_depth[valid] / norm[valid]
        return np.ascontiguousarray(
            np.column_stack(
                (
                    x[valid] * scale,
                    y[valid] * scale,
                    scale,
                )
            ).astype(np.float32)
        )

    def make_viewport(self):
        """Create a viewport window for real-time visualization.

        Note:
            - Window size fixed at 1280x760 pixels
        """
        self.wrapped_ui_elements = []
        self.window = ui.Window(self._name, width=1280, height=720 + 40, visible=True)
        self._provider = ui.ByteImageProvider()
        with self.window.frame:
            with ui.ZStack(height=720):
                ui.Rectangle(style={"background_color": 0xFF000000})
                ui.Label(
                    "Run the scenario for image to be received",
                    style={"font_size": 55, "alignment": ui.Alignment.CENTER},
                    word_wrap=True,
                )
                image_provider = ui.ImageWithProvider(
                    self._provider,
                    width=1280,
                    height=720,
                    style={
                        "fill_policy": ui.FillPolicy.PRESERVE_ASPECT_FIT,
                        "alignment": ui.Alignment.CENTER,
                    },
                )

        self.wrapped_ui_elements.append(image_provider)
        self.wrapped_ui_elements.append(self._provider)
        self.wrapped_ui_elements.append(self.window)

    def close(self):
        """Clean up resources by detaching annotators and clearing caches.

        Note:
            - Required for proper shutdown when done using the sensor
            - Also closes viewport window if one was created
        """
        self._rgba_annot.detach(self._render_product_path)
        self._depth_annot.detach(self._render_product_path)

        rep.AnnotatorCache.clear(self._rgba_annot)
        rep.AnnotatorCache.clear(self._depth_annot)

        self.close_render_product()

        if self._viewport:
            self.ui_destroy()

        carb.log_info(f"[{self._name}] Annotator detached. AnnotatorCache cleaned.")

    def ui_destroy(self):
        """Explicitly destroy viewport UI elements.

        Note:
            - Called automatically by close()
            - Only needed if manually managing UI lifecycle
        """
        for elem in self.wrapped_ui_elements:
            elem.destroy()
