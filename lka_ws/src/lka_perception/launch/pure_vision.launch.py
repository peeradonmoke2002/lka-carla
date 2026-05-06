from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    roi_arg = DeclareLaunchArgument(
        'roi_yaml',
        default_value=os.path.join(
            get_package_share_directory('lka_perception'),
            'config', 'roi.yaml'
        ),
        description='Path to roi.yaml'
    )

    pure_vision_node = Node(
        package='lka_perception',
        executable='pure_vision_node.py',
        name='pure_vision_node',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'roi_yaml': LaunchConfiguration('roi_yaml')},
        ],
    )

    return LaunchDescription([
        roi_arg,
        pure_vision_node,
    ])
