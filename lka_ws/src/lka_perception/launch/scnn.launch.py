from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_dir = get_package_share_directory('lka_perception')

    params_arg = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(pkg_dir, 'config', 'scnn_params.yaml'),
        description='Path to scnn_params.yaml',
    )

    scnn_node = Node(
        package='lka_perception',
        executable='scnn_node.py',
        name='scnn_node',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {'use_sim_time': True},
        ],
    )

    return LaunchDescription([
        params_arg,
        scnn_node,
    ])
