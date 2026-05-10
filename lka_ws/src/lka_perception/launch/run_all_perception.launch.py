import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import RegisterEventHandler
from launch_ros.actions import Node
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    package_name = "lka_perception"
    pkg_dir = get_package_share_directory(package_name)
    config_path = "config"

    yolo_file_path = os.path.join(pkg_dir, config_path, 'lane_detection_params.yaml')
    scnn_file_path = os.path.join(pkg_dir, config_path, 'scnn_params.yaml')
    pure_vision_file_path = os.path.join(pkg_dir, config_path, 'pure_vision_params.yaml')

    yolo_params_arg = DeclareLaunchArgument(
        'yolo_file_path',
        default_value=yolo_file_path,
        description='Path to lane_detection_params.yaml',
    )
    scnn_params_arg = DeclareLaunchArgument(
        'scnn_file_path',
        default_value=scnn_file_path,
        description='Path to scnn_params.yaml',
    )
    pure_vision_params_arg = DeclareLaunchArgument(
        'pure_vision_file_path',
        
        default_value=pure_vision_file_path,
        description='Path to pure_vision_params.yaml',
    )


    gt = Node(
        package=package_name,
        executable='gt_node.py',
        name='gt_node',
        output='screen',
        parameters=[
            {'use_sim_time': True},
        ],
    )

    yolo = Node(
        package=package_name,
        executable='yolo_node.py',
        name='yolo_node',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            LaunchConfiguration('yolo_file_path'),
        ],
    )

    scnn = Node(
        package=package_name,
        executable='scnn_node.py',
        name='scnn_node',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            LaunchConfiguration('scnn_file_path'),
        ],
    )

    pure_vision = Node(
        package=package_name,
        executable='pure_vision_node.py',
        name='pure_vision_node',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            LaunchConfiguration('pure_vision_file_path'),
        ],
    )

    ld = LaunchDescription()


    ld.add_action(yolo_params_arg)
    ld.add_action(scnn_params_arg)
    ld.add_action(pure_vision_params_arg)
    ld.add_action(gt)
    ld.add_action(yolo)
    ld.add_action(scnn)
    ld.add_action(pure_vision)

    return ld


