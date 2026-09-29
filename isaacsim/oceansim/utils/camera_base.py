from typing import Optional, Sequence, Tuple

import numpy as np
import omni.replicator.core as rep
from isaacsim.core.experimental.objects import Camera as _ExperimentalCamera


class OceanSimCameraBase(_ExperimentalCamera):
    """Compatibility base class exposing the single-camera surface OceanSim's sensors were built
    against (the deprecated Isaac Sim 5.x single-camera sensor wrapper), implemented on top of
    the experimental ``Camera`` object for prim/optical-property authoring and a manually
    created Replicator render product for data annotators.

    .. note::

        The 6.1 RTX camera-sensor successor class was considered instead of a manual render
        product, but its annotator allowlist doesn't
        include the raw annotators OceanSim's sonar/underwater-camera pipeline needs (``CameraParams``,
        ``bounding_box_3d_fast``, ``pointcloud`` with ``includeUnlabelled``), so annotators are still
        attached directly via ``omni.replicator.core`` here, unchanged from the pre-migration code.

    Args:
        prim_path: Prim path of the Camera prim to encapsulate or create.
        name: Short name for the sensor.
        resolution: Resolution of the camera (width, height).
        position: Position in the world frame of the prim. Shape is (3, ).
        translation: Translation in the local frame of the prim.
        orientation: Quaternion orientation (w, x, y, z). Shape is (4, ).
        render_product_path: Path to an existing render product to reuse instead of creating a new one.
    """

    def __init__(
        self,
        prim_path: str,
        name: str = "camera",
        frequency: Optional[int] = None,
        dt: Optional[float] = None,
        resolution: Optional[Tuple[int, int]] = None,
        position: Optional[Sequence[float]] = None,
        orientation: Optional[Sequence[float]] = None,
        translation: Optional[Sequence[float]] = None,
        render_product_path: Optional[str] = None,
    ) -> None:
        # `frequency`/`dt` are accepted for call-site compatibility only: OceanSim's camera-derived
        # sensors never used them for anything besides the (deprecated) Camera's internal render-rate
        # setting, which is orthogonal to how these sensors are actually stepped/rendered.
        self._name = name
        self._resolution = resolution if resolution is not None else (128, 128)
        self._render_product = None
        self._render_product_path = render_product_path

        super().__init__(
            prim_path,
            positions=[position] if position is not None else None,
            translations=[translation] if translation is not None else None,
            orientations=[orientation] if orientation is not None else None,
        )
        # Match the old Camera's guarantee that apertures use square pixels even before initialize().
        self.enforce_square_pixels(resolutions=[np.flip(self._resolution)], modes="horizontal")

    @property
    def prim_path(self) -> str:
        return self.paths[0]

    @property
    def prim(self):
        return self.prims[0]

    @property
    def name(self) -> str:
        return self._name

    def initialize(self, physics_sim_view=None) -> None:
        if self._render_product_path is None:
            self._render_product = rep.create.render_product(self.prim_path, resolution=self._resolution)
            self._render_product_path = self._render_product.path

    def get_render_product_path(self) -> str:
        return self._render_product_path

    def close_render_product(self) -> None:
        """Destroy this camera's own render product (its GPU-resident Hydra texture), if it created one.

        A render product passed in via ``render_product_path`` and owned by another camera is left
        alone. Detaching annotators (as sensor ``close()`` methods already do) does not release the
        render product itself -- its Hydra texture stays allocated on the GPU until Python's garbage
        collector eventually reclaims it, which is not guaranteed to happen before the next stage
        reload.

        The stored path is cleared too, so a later ``initialize()`` (e.g. from the Reset button, which
        closes and re-initializes the same sensor objects) creates a fresh render product instead of
        attaching annotators to the destroyed one.
        """
        if self._render_product is not None:
            self._render_product.destroy()
            self._render_product = None
            self._render_product_path = None

    def get_resolution(self) -> Tuple[int, int]:
        """Camera resolution in pixels: (width, height)."""
        return self._resolution

    def get_world_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        positions, orientations = self.get_world_poses()
        return positions.numpy()[0], orientations.numpy()[0]

    def get_focal_length(self) -> float:
        return float(self.get_focal_lengths().numpy()[0, 0])

    def set_focal_length(self, value: float) -> None:
        self.set_focal_lengths([value])

    def get_horizontal_aperture(self) -> float:
        horizontal, _ = self.get_apertures()
        return float(horizontal.numpy()[0, 0])

    def set_horizontal_aperture(self, value: float, maintain_square_pixels: bool = True) -> None:
        self.set_apertures(horizontal_apertures=[value])
        if maintain_square_pixels:
            self.enforce_square_pixels(resolutions=[np.flip(self._resolution)], modes="horizontal")

    def get_clipping_range(self) -> Tuple[float, float]:
        near, far = self.get_clipping_ranges()
        return float(near.numpy()[0, 0]), float(far.numpy()[0, 0])

    def set_clipping_range(self, near_distance: Optional[float] = None, far_distance: Optional[float] = None) -> None:
        self.set_clipping_ranges(
            near_distances=[near_distance] if near_distance is not None else None,
            far_distances=[far_distance] if far_distance is not None else None,
        )
