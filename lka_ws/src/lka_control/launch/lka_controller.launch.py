from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('wheel_base',        default_value='2.875', description='Wheelbase [m] (Lincoln MKZ default)'),
        DeclareLaunchArgument('lane_width',        default_value='3.5',   description='Lane width [m]'),
        DeclareLaunchArgument('min_lookahead',     default_value='3.0',   description='Minimum lookahead distance [m]'),
        DeclareLaunchArgument('max_lookahead',     default_value='10.0',  description='Maximum lookahead distance [m]'),
        DeclareLaunchArgument('ld_velocity_ratio', default_value='2.4',   description='Speed multiplier for lookahead (Autoware default)'),
        DeclareLaunchArgument('max_steer_rad',     default_value='1.22',  description='Max steering angle [rad] (~70°)'),
        DeclareLaunchArgument('throttle',          default_value='0.3',   description='Constant throttle [0,1]'),

        Node(
            package='lka_control',
            executable='lka_controller_node.py',
            name='pure_pursuit_node',
            output='screen',
            parameters=[
                {'use_sim_time': True},
                {'wheel_base':        LaunchConfiguration('wheel_base')},
                {'lane_width':        LaunchConfiguration('lane_width')},
                {'min_lookahead':     LaunchConfiguration('min_lookahead')},
                {'max_lookahead':     LaunchConfiguration('max_lookahead')},
                {'ld_velocity_ratio': LaunchConfiguration('ld_velocity_ratio')},
                {'max_steer_rad':     LaunchConfiguration('max_steer_rad')},
                {'throttle':          LaunchConfiguration('throttle')},
            ],
        ),
    ])
