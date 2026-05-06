"""
Full LKA pipeline launcher.

Starts:
  1. CARLA bridge + vehicle/sensor spawning  (bring_up_carla.launch.py)
  2. Perception node — YOLO or Pure Vision   (lka_perception)
  3. Pure Pursuit controller                 (lka_control)

Usage:
  ros2 launch lka_bringup lka_pipeline.launch.py detection_method:=yolo
  ros2 launch lka_bringup lka_pipeline.launch.py detection_method:=pure_vision
  ros2 launch lka_bringup lka_pipeline.launch.py detection_method:=yolo town:=Town03
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    pkg_bringup     = get_package_share_directory('lka_bringup')
    pkg_perception  = get_package_share_directory('lka_perception')
    pkg_control     = get_package_share_directory('lka_control')

    # ── Args ──────────────────────────────────────────────────────────────────
    detection_method_arg = DeclareLaunchArgument(
        'detection_method', default_value='yolo',
        description='Detection method: yolo | pure_vision'
    )
    weights_arg = DeclareLaunchArgument(
        'weights',
        default_value='/home/peeradon/lka-carla-yolo/models/best_vision.pt',
        description='Path to YOLO .pt weights (used only when detection_method:=yolo)'
    )
    conf_arg = DeclareLaunchArgument(
        'conf_threshold', default_value='0.25',
        description='YOLO confidence threshold (yolo only)'
    )
    roi_arg = DeclareLaunchArgument(
        'roi_yaml',
        default_value=os.path.join(pkg_perception, 'config', 'roi.yaml'),
        description='Path to roi.yaml (pure_vision only)'
    )
    town_arg  = DeclareLaunchArgument('town', default_value='Town01')
    host_arg  = DeclareLaunchArgument('host', default_value='localhost')
    port_arg  = DeclareLaunchArgument('port', default_value='2000')

    # ── CARLA bridge + vehicle ─────────────────────────────────────────────────
    carla = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_bringup, 'launch', 'bring_up_carla.launch.py')
        ),
        launch_arguments={
            'host': LaunchConfiguration('host'),
            'port': LaunchConfiguration('port'),
            'town': LaunchConfiguration('town'),
        }.items(),
    )

    # ── YOLO perception ────────────────────────────────────────────────────────
    yolo_perception = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_perception, 'launch', 'lane_detection.launch.py')
        ),
        launch_arguments={
            'weights':        LaunchConfiguration('weights'),
            'conf_threshold': LaunchConfiguration('conf_threshold'),
        }.items(),
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration('detection_method'), "' == 'yolo'"])
        ),
    )

    # ── Pure Vision perception ─────────────────────────────────────────────────
    pure_vision_perception = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_perception, 'launch', 'pure_vision.launch.py')
        ),
        launch_arguments={
            'roi_yaml': LaunchConfiguration('roi_yaml'),
        }.items(),
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration('detection_method'), "' == 'pure_vision'"])
        ),
    )

    # ── Pure Pursuit controller ────────────────────────────────────────────────
    control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_control, 'launch', 'lka_controller.launch.py')
        ),
    )

    return LaunchDescription([
        detection_method_arg,
        weights_arg,
        conf_arg,
        roi_arg,
        town_arg,
        host_arg,
        port_arg,
        carla,
        yolo_perception,
        pure_vision_perception,
        control,
    ])
