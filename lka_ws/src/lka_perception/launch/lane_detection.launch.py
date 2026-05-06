from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    weights_arg = DeclareLaunchArgument(
        'weights',
        default_value='/home/peeradon/lka-carla-yolo/models/best_vision.pt',
        description='Path to YOLOv26-seg weights (.pt)'
    )
    conf_arg = DeclareLaunchArgument(
        'conf_threshold', default_value='0.25', description='YOLO confidence threshold'
    )

    lane_detection_node = Node(
        package='lka_perception',
        executable='lane_detection_node.py',
        name='lane_detection_node',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'weights': LaunchConfiguration('weights')},
            {'conf_threshold': LaunchConfiguration('conf_threshold')},
        ],
    )

    return LaunchDescription([
        weights_arg,
        conf_arg,
        lane_detection_node,
    ])
