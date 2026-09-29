# Omniverse import
import traceback

import carb
import numpy as np
from pxr import Gf

# Isaac sim import
from isaacsim.core.experimental.prims import RigidPrim
from isaacsim.core.experimental.utils.prim import get_prim_path


class MHL_Sensor_Example_Scenario():
    def __init__(self):
        self._rob = None
        self._rob_view = None
        self._imu = None
        self._sonar = None
        self._cam = None
        self._DVL = None
        self._baro = None

        self._ctrl_mode = None
        self._cmd_vel_controller = None
        self._use_ros = True
        self.omni_ros = None
        # Replicator writers attached by the ros2_helpers publish_* functions
        self._ros_writers = []

        self._running_scenario = False
        self._time = 0.0

    def setup_scenario(self, imu, rob, sonar, cam, DVL, baro, ctrl_mode, use_ros=True):
        self._use_ros = use_ros
        self._imu = imu
        self._rob = rob
        self._sonar = sonar
        self._cam = cam
        self._DVL = DVL
        self._baro = baro
        self._ctrl_mode = ctrl_mode

        if self._use_ros:
            from isaacsim.oceansim.sensors import ros2_helpers

            self.omni_ros = ros2_helpers.OmniHandler(
                name="SensorExample",
                use_camera=self._cam is not None,
                use_sonar=self._sonar is not None,
                use_imu=self._imu is not None,
                use_dvl=self._DVL is not None,
                use_baro=self._baro is not None,
            )

            if self._imu is not None:
                self._imu.initialize(og_node=self.omni_ros._imu_node)

            if self._sonar is not None:
                approx_freq = 30
                self._sonar.sonar_initialize(
                    include_unlabelled=True, og_node=self.omni_ros._sonar_node
                )
                # TODO: The camera_info/depth/pointcloud topics below come from the sonar's virtual
                # pinhole camera (2000x307, 130x20 deg FOV), so they are ground-truth geometry with
                # none of the sonar physics that make_sonar_data() applies: no reflectivity, no
                # range/azimuth binning, no speckle noise, no attenuation, and full elevation
                # resolution a real FLS lacks. Add sonar-physics versions of these outputs (or relabel
                # them as ground truth). The physical sonar output is the <name>/sonar_image topic.
                self._ros_writers += [
                    ros2_helpers.publish_camera_info(self._sonar, approx_freq),
                    ros2_helpers.publish_depth(self._sonar, approx_freq),
                    ros2_helpers.publish_pointcloud_from_depth(self._sonar, approx_freq),
                ]
                ros2_helpers.publish_camera_tf(self._sonar)

            if self._cam is not None:
                self._cam.initialize(
                    og_node=self.omni_ros._rgb_node,
                    depth_og_node=self.omni_ros._depth_node,
                    pointcloud_og_node=self.omni_ros._pointcloud_node,
                )
                approx_freq = 30
                self._ros_writers.append(ros2_helpers.publish_camera_info(self._cam, approx_freq))
                ros2_helpers.publish_camera_tf(self._cam)

            if self._DVL is not None:
                self._DVL.initialize(og_node=self.omni_ros._dvl_node)
                self._DVL_reading = [0.0, 0.0, 0.0]

            if self._baro is not None:
                self._baro.initialize(og_node=self.omni_ros._baro_node)
                self._baro_reading = 101325.0  # atmospheric pressure (Pa)
        else:
            self.omni_ros = None

            if self._imu is not None and hasattr(self._imu, "initialize"):
                self._imu.initialize()

            if self._sonar is not None:
                self._sonar.sonar_initialize(include_unlabelled=True)

            if self._cam is not None:
                self._cam.initialize()

            if self._DVL is not None:
                self._DVL_reading = [0.0, 0.0, 0.0]

            if self._baro is not None:
                self._baro_reading = 101325.0  # atmospheric pressure (Pa)

        # Setup cmd_vel ROS2 subscriber (works with any control mode except Manual)
        if ctrl_mode != "Manual control":
            from ...utils.cmd_vel_subscriber import CmdVelController
            robot_path = get_prim_path(self._rob)
            self._cmd_vel_controller = CmdVelController(robot_prim_path=robot_path)

        # Create the RigidPrim tensor view once; building one per physics step is expensive
        if ctrl_mode in ("Manual control", "Straight line"):
            self._rob_view = RigidPrim(get_prim_path(self._rob))

        # Apply forces via RigidPrim tensor view if manual control
        if ctrl_mode == "Manual control":
            from ...utils.keyboard_cmd import keyboard_cmd

            self._force_cmd = keyboard_cmd(base_command=np.array([0.0, 0.0, 0.0]),
                                      input_keyboard_mapping={
                                        # forward command
                                        "W": [10.0, 0.0, 0.0],
                                        # backward command
                                        "S": [-10.0, 0.0, 0.0],
                                        # leftward command
                                        "A": [0.0, 10.0, 0.0],
                                        # rightward command
                                        "D": [0.0, -10.0, 0.0],
                                         # rise command
                                        "UP": [0.0, 0.0, 10.0],
                                        # sink command
                                        "DOWN": [0.0, 0.0, -10.0],
                                      })
            self._torque_cmd = keyboard_cmd(base_command=np.array([0.0, 0.0, 0.0]),
                                      input_keyboard_mapping={
                                        # yaw command (left)
                                        "J": [0.0, 0.0, 10.0],
                                        # yaw command (right)
                                        "L": [0.0, 0.0, -10.0],
                                        # pitch command (up)
                                        "I": [0.0, -10.0, 0.0],
                                        # pitch command (down)
                                        "K": [0.0, 10.0, 0.0],
                                        # row command (left)
                                        "LEFT": [-10.0, 0.0, 0.0],
                                        # row command (negative)
                                        "RIGHT": [10.0, 0.0, 0.0],
                                      })
            
        self._running_scenario = True
    # This function will only be called if ctrl_mode==waypoints and waypoints files are changed
    def setup_waypoints(self, waypoint_path, default_waypoint_path):
        def read_data_from_file(file_path):
            # Initialize an empty list to store the floats
            data = []
            
            # Open the file in read mode
            with open(file_path, 'r') as file:
                # Read each line in the file
                for line in file:
                    # Strip any leading/trailing whitespace and split the line by spaces
                    float_strings = line.strip().split()
                    
                    # Convert the list of strings to a list of floats
                    floats = [float(x) for x in float_strings]
                    
                    # Append the list of floats to the data list
                    data.append(floats)
            
            return data
        try:
            self.waypoints = read_data_from_file(waypoint_path)
            print('Waypoints loaded successfully.')
            print(f'Waypoint[0]: {self.waypoints[0]}')
        except:
            self.waypoints = read_data_from_file(default_waypoint_path)
            print('Fail to load this waypoints. Back to default waypoints.')

        
    @staticmethod
    def _run_teardown_step(description, fn):
        """Run one cleanup step, logging (not raising) on failure so the remaining steps still run."""
        try:
            fn()
        except Exception:
            carb.log_error(f"[SensorExample] Teardown step '{description}' failed:\n{traceback.format_exc()}")

    def teardown_scenario(self):
        # Stop update_scenario() from touching sensors while they are being closed.
        self._running_scenario = False
        try:
            # Detach the ROS writers first: close() destroys the render products they are attached to.
            for writer in self._ros_writers:
                self._run_teardown_step(f"detach ROS writer {type(writer).__name__}", writer.detach)

            # Because these two sensors create annotator cache in GPU,
            # close() will detach annotator from render product and clear the cache.
            # TODO: the IMU is only dereferenced, not closed.
            if self._sonar is not None:
                self._run_teardown_step("close sonar", self._sonar.close)
            if self._cam is not None:
                self._run_teardown_step("close camera", self._cam.close)

            # Clear cmd_vel controller
            if self._cmd_vel_controller is not None:
                self._run_teardown_step("clean up cmd_vel controller", self._cmd_vel_controller.cleanup)

            # clear the keyboard subscription
            if self._ctrl_mode=="Manual control":
                self._run_teardown_step("clean up keyboard force command", lambda: self._force_cmd.cleanup())
                self._run_teardown_step("clean up keyboard torque command", lambda: self._torque_cmd.cleanup())
        finally:
            self._ros_writers = []
            self._cmd_vel_controller = None
            self._imu = None
            self._rob = None
            self._rob_view = None
            self._sonar = None
            self._cam = None
            self._DVL = None
            self._baro = None
            self.omni_ros = None
            self._time = 0.0


    def update_scenario(self, step: float):


        if not self._running_scenario:
            return

        self._time += step
        if self._imu is not None:
            if self._use_ros:
                self._imu.read()
            else:
                self._imu.get_data()
        if self._sonar is not None:
            self._sonar.make_sonar_data()
        if self._cam is not None:
            self._cam.render()
        if self._DVL is not None:
            if self._use_ros:
                self._DVL_reading = self._DVL.read()
            else:
                self._DVL_reading = self._DVL.get_linear_vel()
        if self._baro is not None:
            if self._use_ros:
                self._baro_reading = float(self._baro.read())
            else:
                self._baro_reading = float(self._baro.get_pressure())

        # Update cmd_vel controller if active
        if self._cmd_vel_controller is not None:
            self._cmd_vel_controller.update(self._rob)

        if self._ctrl_mode=="Manual control":
            # TODO: Investigate unexplained drift. In Manual control the ROV sometimes drifts slowly
            # along its x axis as if in a current, seen against a fixed point in the camera view;
            # observed after rolling the ROV. Seen both backwards and (another run) forwards. In the
            # forwards case, holding S stopped it completely and holding W moved faster, i.e. a
            # constant offset of exactly one key's force (+10 in x). That strongly suggests a stuck
            # keyboard command: commands are press/release counted, so one unmatched press or
            # release leaves a constant force. Less likely: the bounding-cube collider overlapping
            # the tank/rock, or the local-frame force applied away from the centre of mass.
            force = np.asarray(self._force_cmd._base_command, dtype=np.float32).reshape(1, 3)
            torque = np.asarray(self._torque_cmd._base_command, dtype=np.float32).reshape(1, 3)
            self._rob_view.apply_forces_and_torques_at_pos(
                forces=force, torques=torque, local_frame=True
            )
        elif self._ctrl_mode=="Waypoints":
            if len(self.waypoints) > 0:
                waypoints = self.waypoints[0]
                self._rob.GetAttribute('xformOp:translate').Set(Gf.Vec3f(waypoints[0], waypoints[1], waypoints[2]))
                self._rob.GetAttribute('xformOp:orient').Set(Gf.Quatd(waypoints[3], waypoints[4], waypoints[5], waypoints[6]))
                self.waypoints.pop(0)
            else:
                print('Waypoints finished')                
        elif self._ctrl_mode=="Straight line":
            self._rob_view.set_velocities(linear_velocities=[[0.5, 0.0, 0.0]])
