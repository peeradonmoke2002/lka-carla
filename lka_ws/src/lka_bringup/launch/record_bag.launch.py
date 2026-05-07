"""
Record a ROS2 bag for perception evaluation.
Auto-cycles through 4 weather conditions at a fixed interval.

Weather order: Rain (t=0) → Clear (t=D) → Fog (t=2D) → Night (t=3D)
where D = weather_duration seconds (default 30)

Usage:
  ros2 launch lka_bringup record_bag.launch.py detection_method:=yolo
  ros2 launch lka_bringup record_bag.launch.py detection_method:=pure_vision
  ros2 launch lka_bringup record_bag.launch.py detection_method:=yolo weather_duration:=60
  ros2 launch lka_bringup record_bag.launch.py detection_method:=yolo bag_path:=/tmp/my_bag
"""

import os
from datetime import datetime
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression


# Raw camera excluded — too large (~120 MB/s). Debug image is sufficient.
COMMON_TOPICS = [
    '/carla/weather_control',
    '/lka/lane_center',
]

YOLO_EXTRA = [
    '/lka/enhanced_image',
    '/lka/detection_confidence',
]

PURE_VISION_EXTRA = [
    '/lka/pure_vision_image',
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
D = 30  # seconds per weather condition


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
        os.path.expanduser('~'),
        'lka_bags',
        datetime.now().strftime('%Y%m%d_%H%M%S'),
    )

    detection_arg = DeclareLaunchArgument(
        'detection_method', default_value='yolo',
        description='yolo | pure_vision'
    )
    bag_path_arg = DeclareLaunchArgument(
        'bag_path', default_value=default_bag,
        description='Output bag directory'
    )
    duration_arg = DeclareLaunchArgument(
        'weather_duration', default_value=str(D),
        description='Seconds per weather condition (change D in file to apply)'
    )

    record_yolo = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'record',
            '--output', LaunchConfiguration('bag_path'),
            '--storage', 'sqlite3',
        ] + COMMON_TOPICS + YOLO_EXTRA,
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration('detection_method'), "' == 'yolo'"])
        ),
        output='screen',
    )

    record_pure_vision = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'record',
            '--output', LaunchConfiguration('bag_path'),
            '--storage', 'sqlite3',
        ] + COMMON_TOPICS + PURE_VISION_EXTRA,
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration('detection_method'), "' == 'pure_vision'"])
        ),
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
        detection_arg,
        bag_path_arg,
        duration_arg,
        record_yolo,
        record_pure_vision,
    ] + weather_actions)
