# Omniverse import
import numpy as np
import omni.graph.core as og
import omni.physx
import carb

# Isaac sim import
import isaacsim.core.experimental.utils.transform as transform_utils
from isaacsim.core.experimental.prims import RigidPrim
from isaacsim.sensors.experimental.physics import Raycast, RaycastSensor
from isaacsim.util.debug_draw import _debug_draw

# Custom import
from isaacsim.oceansim.utils.MultivariateNormal import MultivariateNormal


class DVLsensor:
    def __init__(self,
                 name: str = "DVL",
                 elevation:float = 22.5, # deg
                 rotation: float = 45, # deg
                 vel_cov = 0,
                 depth_cov = 0,
                 min_range: float = 0.1,
                 max_range: float = 100,
                 num_beams_out_range_threshold: int = 2,
                 freq: int = None, # Hz
                 freq_bound: tuple[int] = [5, 100], # Hz
                 freq_dependenet_range_bound: tuple[float] = [7.5, 50.0], # m
                 sound_speed: float = 1500, # m/s
                 ):
        """Initialize a DVL sensor with configurable beam geometry and operating parameters.

        Args:
            name (str): Identifier for the sensor. Defaults to "DVL".
            elevation (float): Beam elevation angle from horizontal in degrees. Defaults to 22.5°.
            rotation (float): Beam rotation about Z-axis in degrees. Defaults to 45° (Janus configuration).
            vel_cov (float): Velocity measurement noise covariance. Defaults to 0 (no noise).
            depth_cov (float): Depth measurement noise covariance. Defaults to 0 (no noise).
            min_range (float): Minimum valid range in meters. Defaults to 0.1m.
            max_range (float): Maximum valid range in meters. Defaults to 100m.
            num_beams_out_range_threshold (int): Number of lost beams before declaring dropout. Defaults to 2.
            freq (int, optional): Fixed operating frequency in Hz. If None, uses adaptive frequency. Defaults to None.
            freq_bound (tuple[int]): (min_freq, max_freq) for adaptive operation. Defaults to (5, 100)Hz.
            freq_dependenet_range_bound (tuple[float]): (min_range, max_range) for frequency adaptation. Defaults to (7.5, 50.0)m.
            sound_speed (float): Speed of sound in water in m/s. Defaults to 1500m/s.
        """


        self._name = name

        # DVL configuration params
        self._elevation = elevation
        self._rotation = rotation
        self._min_range = min_range
        self._max_range = max_range

        # DVL noise params
        self._mvn_vel = MultivariateNormal(4)
        self._mvn_vel.init_cov(vel_cov)
        self._mvn_dep = MultivariateNormal(4)
        self._mvn_dep.init_cov(depth_cov)

        sinElev = np.sin(np.deg2rad(self._elevation))
        cosElev = np.cos(np.deg2rad(self._elevation))
        self._transform = np.array([[1/(2*sinElev), 0, -1/(2*sinElev), 0],
                                    [0, 1/(2*sinElev), 0, -1/(2*sinElev)],
                                    [1/(4*cosElev), 1/(4*cosElev), 1/(4*cosElev), 1/(4*cosElev)]
                                    ])

        # sensor dropout related params
        self._num_beams_out_range_threshold = num_beams_out_range_threshold

        # Realistic DVL frequency dependent params
        self._user_static_freq_flag = False
        if freq is not None:
            self._user_static_freq_flag = True
            self._dt = 1/freq
        else:
            self._freq_bound = freq_bound
            self._freq_dependent_range_bound = freq_dependenet_range_bound
            self._sound_speed = sound_speed

        # Initialization
        self._rigid_body_path = None
        self._elapsed_time_vel = 0.0
        self._elapsed_time_depth = 0.0
        self._debug_lines_sub = None



    def attachDVL(self,
                  rigid_body_path:str,
                  position = None,
                  translation = None,
                  orientation = None
                  ):

        """Attach the DVL sensor to a rigid body in the simulation.
        ..note::
            This function will create a single ``IsaacRaycastSensor`` prim under the parent rigid
            body prim, casting one ray per beam in the Janus configuration.

        Args:
            rigid_body_path (str): USD path to the parent rigid body prim.
            position (Optional[Sequence[float]], optional): position in the world frame of the prim. shape is (3, ).
                                                    Defaults to None, which means left unchanged.
            translation (Optional[Sequence[float]], optional): translation in the local frame of the prim
                                                            (with respect to its parent prim). shape is (3, ).
                                                            Defaults to None, which means left unchanged.
            orientation (Optional[Sequence[float]], optional): quaternion orientation in the world/ local frame of the prim
                                                            (depends if translation or position is specified).
                                                            quaternion is scalar-first (w, x, y, z). shape is (4, ).
                                                            Defaults to None, which means left unchanged.
        Raises:
            Exception: if translation and position defined at the same time

        """
        self._rigid_body_path = rigid_body_path
        self._rigid_body_prim = RigidPrim(rigid_body_path)
        sensor_prim_path = rigid_body_path + "/" + self._name

        elevation = self._elevation
        rotation = self._rotation
        orients_euler = np.array([[elevation, 0.0, rotation],
                                  [0.0, elevation, rotation],
                                  [-elevation, 0.0, rotation],
                                  [0.0, -elevation, rotation]])
        beam_directions = []
        for i in range(orients_euler.shape[0]):
            quat = transform_utils.euler_angles_to_quaternion(orients_euler[i, :], degrees=True).numpy()
            rot_m = transform_utils.quaternion_to_rotation_matrix(quat).numpy()
            beam_directions.append((rot_m @ np.array([0.0, 0.0, -1.0])).tolist())

        self._raycast = Raycast(
            sensor_prim_path,
            positions=[position] if position is not None else None,
            translations=[translation] if translation is not None else None,
            orientations=[orientation] if orientation is not None else None,
            min_range=self._min_range,
            max_range=self._max_range,
            ray_origins=[[0.0, 0.0, 0.0]] * 4,
            ray_directions=beam_directions,
            output_frame="WORLD",
        )
        self._DVL = self._raycast
        self._DVL_sensor = RaycastSensor(self._raycast)

    def add_single_beam(self):
        """Add a single vertical beam to the DVL for simplified depth measurements.

        Creates an additional single-ray raycast sensor oriented straight downward (along -Z axis).
        The beam is created at: <rigid_body_path>/<DVL_name>/SingleBeam

        Note:
            Primarily used for debugging or when single-beam depth measurement is sufficient.
            Uses the same min/max range settings as the main DVL beams.
        """
        self._single_beam_path = self._rigid_body_path + "/" + self._name + "/SingleBeam"
        self._single_beam = Raycast(
            self._single_beam_path,
            min_range=self._min_range,
            max_range=self._max_range,
            ray_origins=[[0.0, 0.0, 0.0]],
            ray_directions=[[0.0, 0.0, -1.0]],
        )
        self._single_beam_sensor = RaycastSensor(self._single_beam)

    def get_single_beam_range(self):
        """Get depth measurement from the vertical single beam. Only call this function after you added a singlebeam.

        Returns:
            float: Depth measurement in meters along the central beam.
                Returns the configured max range if no valid return.

        Note:
            This is a simpler alternative to get_depth() when only vertical range is needed.

        """
        return float(self._single_beam_sensor.get_sensor_reading().depths[0])

    def get_DVL_interface(self):
        """Get direct access to the underlying DVL raycast sensor runtime.

        Returns:
            RaycastSensor: The runtime object providing per-beam depth/hit data.

        Note:
            Advanced use only - provides low-level access to the raycast physics data.
        """
        return self._DVL_sensor

    def get_baseSensor(self):
        """Get the core sensor prim wrapper of the DVL.

        Returns:
            Raycast: The authoring object wrapping the sensor prim (transform, visibility, ...).

        Note:
            Useful for modifying transform or visibility properties.
        """
        return self._DVL

    def get_beam_paths(self):
        """Get the USD path to the DVL's raycast sensor prim, which carries all four beams.

        Returns:
            str: Prim path of the shape ``<rigid_body_path>/<DVL_name>``.

        Note:
            Unlike the pre-migration API, all four beams live on a single prim (one ray per beam)
            rather than four separate beam prims.
        """
        return self._raycast.paths[0]

    def get_depth(self):
        """Get depth measurements from all four beams.

        Returns:
            list[float]: Four depth measurements in meters. Returns NaN for beams with no return.

        Note:
            - Applies Gaussian noise if depth_cov > 0
            - Logs warning if >= num_beams_out_range_threshold beams are lost
        """
        reading = self._DVL_sensor.get_sensor_reading()
        depths = np.asarray(reading.depths, dtype=np.float64) if reading.is_valid else np.full(4, self._max_range)
        if_hit = (depths < self._max_range).tolist()
        depth = depths.tolist()
        if (self._mvn_dep.is_uncertain()):
            sample = self._mvn_dep.sample_array()
            for i in range(4):
                depth[i] += sample[i]
        # check if the sensor is in dropout state
        if if_hit.count(False) >= self._num_beams_out_range_threshold:
            carb.log_warn(f'[{self._name}] Measurement is dropped out')

        # set the no hit depth to nan
        depth = [value if hit else float('nan') for value, hit in zip(depth, if_hit)]
        return depth

    def get_dt(self):
        """Get current sensor update period based on operating mode.

        Returns:
            float: Update period in seconds.

        Note:
            For adaptive frequency mode, calculates period based on:
            - Fixed maximum frequency at close range
            - Sound-speed limited frequency at long range
            - Linear transition between bounds
        """
        if self._user_static_freq_flag:
            return self._dt
        else:
            min_range = min(self.get_depth())
            if min_range <= self._freq_dependent_range_bound[0]:
                self._dt = 1 / self._freq_bound[1]
            elif self._freq_dependent_range_bound[0] < min_range < self._freq_dependent_range_bound[1]:
                # To avoid abrupt jumps at h_min and h_max, smooth the transitions with linear ramp
                freq = self._freq_bound[1] - (self._freq_bound[1] - self._sound_speed/(2 * min_range))/(self._freq_dependent_range_bound[1] - self._freq_dependent_range_bound[0]) * (min_range - self._freq_dependent_range_bound[0])
                self._dt = 1 / freq
            else:
                self._dt = 1 / self._freq_bound[0]
            return self._dt

    def get_beam_hit(self):
        """Get hit detection status for all four DVL beams.

        Returns:
            list[bool]: Boolean hit status for each beam in order [beam_0, beam_1, beam_2, beam_3]
                        True indicates beam has valid return, False indicates no return detected.

        Note:
            - Useful for monitoring individual beam performance
            - Mirrors the hit detection used internally in get_depth() and get_linear_vel()
            - A beam reports a miss when its depth reading equals the configured max range
        """
        reading = self._DVL_sensor.get_sensor_reading()
        if not reading.is_valid:
            return [False] * 4
        return (np.asarray(reading.depths, dtype=np.float64) < self._max_range).tolist()

    def get_linear_vel(self):
        """Get 3D velocity vector in body frame.

        Returns:
            np.ndarray: [vx, vy, vz] velocity in m/s. Returns zeros during dropout.

        Note:
            - Applies Gaussian noise if vel_cov > 0
        """
        if_hit = self.get_beam_hit()
        if if_hit.count(False) >= self._num_beams_out_range_threshold:
            carb.log_warn(f'[{self._name}] Measurement is dropped out')
            return np.zeros(3)

        world_vel = self._rigid_body_prim.get_velocities()[0].numpy()[0]
        _, world_orient = self._rigid_body_prim.get_world_poses()
        rot_m = transform_utils.quaternion_to_rotation_matrix(world_orient.numpy()[0]).numpy()
        vel = rot_m.T @ world_vel
        if (self._mvn_vel.is_uncertain()):
            sample = self._mvn_vel.sample_array()
            for i in range(4):
                for j in range(3):
                    vel[j] += self._transform[j][i] * sample[i]

        return vel


    def get_linear_vel_fd(self, physics_dt: float):
        """Frequency-dependent version of get_linear_vel() that respects sensor update rate.

        Args:
            physics_dt (float): Current physics timestep duration.

        Returns:
            Union[np.ndarray, float]: Velocity vector if update is due, otherwise NaN.
        """
        if self.get_dt() < physics_dt:
            carb.log_warn(f'[{self._name}] Simulation physics_dt is larger than sensor_dt. Reduced to get_linear_vel().')
        self._elapsed_time_vel += physics_dt
        if self._elapsed_time_vel >= self.get_dt():
            self._elapsed_time_vel = 0.0
            return self.get_linear_vel()
        else:
            return float('nan')

    def get_depth_fd(self, physics_dt: float):
        """Frequency-dependent version of get_depth() that respects sensor update rate.

        Args:
            physics_dt (float): Current physics timestep duration.

        Returns:
            Union[list[float], float]: Depth measurements if update is due, otherwise NaN.
        """
        if self.get_dt() < physics_dt:
            carb.log_warn(f'[{self._name}] Simulation physics_dt is larger than sensor_dt. Reduced to get_depth().')
        self._elapsed_time_depth += physics_dt
        if self._elapsed_time_depth >= self.get_dt():
            self._elapsed_time_depth = 0.0
            return self.get_depth()
        else:
            return float('nan')

    def set_freq(self, freq: float):
        """Set a fixed operating frequency for the DVL sensor.

        Args:
            freq (float): Desired operating frequency in Hz (must be > 0)

        Note:
            - Overrides any adaptive frequency behavior
            - Automatically calculates the corresponding period (dt = 1/freq)
            - Sets internal flag to maintain fixed frequency mode
            - To revert to adaptive frequency, create a new DVL instance

        Example:
            >>> dvl.set_freq(10)  # Sets DVL to update at 10Hz
        """
        self._user_static_freq_flag = True
        self._dt = 1 / freq

    def add_debug_lines(self, color=(0.0, 1.0, 0.0, 1.0), width=2):
        """Visualize DVL beams in the viewport using debug drawing.

        Subscribes to the physics step event stream and redraws the four beam rays
        (from the sensor origin to the hit point, or to max range if no hit) every step.
        """
        draw_interface = _debug_draw.acquire_debug_draw_interface()
        physx_interface = omni.physx.get_physx_interface()

        def _on_physics_step(dt):
            reading = self._DVL_sensor.get_sensor_reading()
            draw_interface.clear_lines()
            if not reading.is_valid:
                return
            starts = [tuple(p) for p in reading.ray_origins_world]
            ends = [tuple(p) for p in reading.ray_end_points_world]
            draw_interface.draw_lines(starts, ends, [color] * len(starts), [width] * len(starts))

        self._debug_lines_sub = physx_interface.subscribe_physics_step_events(_on_physics_step)
