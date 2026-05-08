"""
Record a ROS2 bag for perception evaluation.
Records BOTH YOLO and Pure Vision topics simultaneously.
Auto-cycles through 4 weather conditions at a fixed interval.

Weather order: Rain (t=0) → Clear (t=D) → Fog (t=2D) → Night (t=3D)
where D = weather_duration seconds (default 60)

Usage:
  ros2 launch lka_bringup record_bag.launch.py
  ros2 launch lka_bringup record_bag.launch.py bag_path:=/tmp/my_bag
"""

import os
from datetime import datetime
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction, LogInfo
from launch.substitutions import LaunchConfiguration


RECORD_TOPICS = [
    '/carla/weather_control',
    '/lka/yolo/lane_center',
    '/lka/pure_vision/lane_center',
]

WEATHER_PRESETS = {
    'rain': (
        '{cloudiness: 60.0, precipitation: 40.0, precipitation_deposits: 40.0,'
        ' wind_intensity: 30.0, sun_azimuth_angle: 275.0, sun_altitude_angle: 20.0,'
        ' fog_density: 5.0, fog_distance: 0.75, wetness: 80.0}'
    ),
    'clear': (
        '{cloudiness: 0.0, precipitation: 0.0, precipitation_deposits: 0.0,'
        ' wind_intensity: 0.0, sun_azimuth_angle: 0.0, sun_altitude_angle: 75.0,'
        ' fog_density: 0.0, fog_distance: 0.0, wetness: 0.0}'
    ),
    'fog': (
        '{cloudiness: 80.0, precipitation: 0.0, precipitation_deposits: 0.0,'
        ' wind_intensity: 0.0, sun_azimuth_angle: 0.0, sun_altitude_angle: 45.0,'
        ' fog_density: 80.0, fog_distance: 10.0, wetness: 0.0}'
    ),
    'night': (
        '{cloudiness: 0.0, precipitation: 0.0, precipitation_deposits: 0.0,'
        ' wind_intensity: 0.0, sun_azimuth_angle: 0.0, sun_altitude_angle: -90.0,'
        ' fog_density: 0.0, fog_distance: 0.0, wetness: 0.0}'
    ),
}

WEATHER_ORDER = ['rain', 'clear', 'fog', 'night']
D = 60  # seconds per weather condition


def pub_weather(name):
    return ExecuteProcess(
        cmd=[
            'ros2', 'topic', 'pub', '--once',
            '/carla/weather_control',
            'carla_msgs/msg/CarlaWeatherParameters',
            WEATHER_PRESETS[name],
        ],
        output='screen',
    )


def generate_launch_description():
    default_bag = os.path.join(
        '/home/peeradon/lka-carla-yolo/bags',
        datetime.now().strftime('%Y%m%d_%H%M%S'),
    )

    bag_path_arg = DeclareLaunchArgument(
        'bag_path', default_value=default_bag,
        description='Output bag directory'
    )

    record = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'record',
            '--output', LaunchConfiguration('bag_path'),
            '--storage', 'sqlite3',
        ] + RECORD_TOPICS,
        output='screen',
    )

    weather_actions = []
    for i, name in enumerate(WEATHER_ORDER):
        delay = float(i * D)
        weather_actions.append(
            TimerAction(
                period=delay,
                actions=[
                    LogInfo(msg=f'[weather] t={int(delay)}s → switching to {name.upper()}'),
                    pub_weather(name),
                ],
            )
        )

    return LaunchDescription([
        bag_path_arg,
        record,
    ] + weather_actions)
