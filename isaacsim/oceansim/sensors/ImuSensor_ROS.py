import omni.graph.core as og
import omni.timeline
from isaacsim.sensors.experimental.physics import IMU, IMUSensor


class ImuSensor_ROS(IMUSensor):
    """OceanSim IMU wrapper that publishes IMU data via OmniGraph ROS2PublishImu."""

    def __init__(
        self,
        prim_path,
        name="Imu",
        translation=None,
        og_node=None,
    ):
        self._name = name
        self._og_node = og_node
        imu = IMU(
            prim_path,
            translations=[translation] if translation is not None else None,
        )
        super().__init__(imu)

    def initialize(self, physics_sim_view=None, og_node=None):
        if og_node is not None:
            self._og_node = og_node

    def read(self):
        # imu api: https://docs.isaacsim.omniverse.nvidia.com/6.1.0/py/source/extensions/isaacsim.sensors.experimental.physics/docs/index.html#isaacsim.sensors.experimental.physics.IMUSensor
        # graph node attributes: https://docs.isaacsim.omniverse.nvidia.com/6.1.0/py/source/extensions/isaacsim.ros2.bridge/docs/ogn/OgnROS2PublishImu.html
        imu_data = self.get_data()
        if self._og_node is None:
            return imu_data

        sim_time = float(omni.timeline.get_timeline_interface().get_current_time())
        if self._og_node.get_attribute_exists("inputs:timeStamp"):
            og.Controller.attribute(
                self._og_node.get_attribute("inputs:timeStamp")
            ).set(sim_time)
        og.Controller.attribute(
            self._og_node.get_attribute("inputs:angularVelocity")
        ).set(imu_data["angular_velocity"])
        og.Controller.attribute(
            self._og_node.get_attribute("inputs:linearAcceleration")
        ).set(imu_data["linear_acceleration"])
        og.Controller.attribute(
            self._og_node.get_attribute("inputs:orientation")
        ).set(imu_data["orientation"])
        return imu_data
