import math

import carla
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from transforms3d.euler import euler2quat
from transforms3d.euler import quat2euler


def carla_location_to_ros_point(carla_location):
    """Convert a carla location to a ROS point."""
    ros_point = Point()
    ros_point.x = carla_location.x
    ros_point.y = -carla_location.y
    ros_point.z = carla_location.z

    return ros_point


def carla_rotation_to_ros_quaternion(carla_rotation):
    """Convert a carla rotation to a ROS quaternion."""
    roll = math.radians(carla_rotation.roll)
    pitch = -math.radians(carla_rotation.pitch)
    yaw = -math.radians(carla_rotation.yaw)
    quat = euler2quat(roll, pitch, yaw)
    ros_quaternion = Quaternion(w=quat[0], x=quat[1], y=quat[2], z=quat[3])

    return ros_quaternion


def ros_quaternion_to_carla_rotation(ros_quaternion):
    """Convert ROS quaternion to carla rotation."""
    roll, pitch, yaw = quat2euler(
        [ros_quaternion.w, ros_quaternion.x, ros_quaternion.y, ros_quaternion.z]
    )

    return carla.Rotation(
        roll=math.degrees(roll), pitch=-math.degrees(pitch), yaw=-math.degrees(yaw)
    )


def ros_pose_to_carla_transform(ros_pose):
    """Convert ROS pose to carla transform."""
    return carla.Transform(
        carla.Location(ros_pose.position.x, -ros_pose.position.y, ros_pose.position.z),
        ros_quaternion_to_carla_rotation(ros_pose.orientation),
    )