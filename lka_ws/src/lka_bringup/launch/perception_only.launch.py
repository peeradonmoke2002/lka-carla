"""
Launch a single perception node for closed-loop trials.

Usage:
  ros2 launch lka_bringup perception_only.launch.py method:=yolo
  ros2 launch lka_bringup perception_only.launch.py method:=pure_vision
  ros2 launch lka_bringup perception_only.launch.py method:=scnn
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

METHOD_MAP = {
    'yolo':        ('yolo_node.py',        'yolo_node',        'yolo_params.yaml',       '/lka/yolo/lane_center'),
    'pure_vision': ('pure_vision_node.py', 'pure_vision_node', 'pure_vision_params.yaml', '/lka/pure_vision/lane_center'),
    'scnn':        ('scnn_node.py',        'scnn_node',        'scnn_params.yaml',        '/lka/scnn/lane_center'),
}


def launch_setup(context, *args, **kwargs):
    method = LaunchConfiguration('method').perform(context)
    if method not in METHOD_MAP:
        raise ValueError(f"method must be one of {list(METHOD_MAP)}; got '{method}'")

    executable, node_name, config_yaml, output_topic = METHOD_MAP[method]
    pkg_dir = get_package_share_directory('lka_perception')
    config_path = os.path.join(pkg_dir, 'config', config_yaml)

    return [Node(
        package='lka_perception',
        executable=executable,
        name=node_name,
        output='screen',
        parameters=[config_path, {'use_sim_time': True, 'enable_hysteresis': True}],
        remappings=[(output_topic, '/lka/lane_center')],
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'method',
            description='Perception method: yolo | pure_vision | scnn',
        ),
        OpaqueFunction(function=launch_setup),
    ])
