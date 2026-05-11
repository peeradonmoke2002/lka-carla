"""
Closed-loop controller test: one perception method × one weather condition.

Usage:
  ros2 launch lka_bringup closed_loop.launch.py method:=yolo        weather:=rain
  ros2 launch lka_bringup closed_loop.launch.py method:=pure_vision weather:=clear
  ros2 launch lka_bringup closed_loop.launch.py method:=scnn        weather:=fog

Assumes CARLA is already running with the ego vehicle spawned.

What this launcher does:
  1. Starts only the selected perception node with enable_hysteresis:=true
  2. Remaps /lka/<method>/lane_center → /lka/lane_center for the controller
  3. Starts the Pure Pursuit controller
  4. Sets weather after 3 s (time for CARLA to stabilise)
  5. Records bag to bags/closed_loop/<method>_<weather>/
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


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

METHOD_MAP = {
    # method_key: (executable, node_name, config_yaml, output_topic)
    'yolo': (
        'yolo_node.py', 'yolo_node',
        'yolo_params.yaml', '/lka/yolo/lane_center',
    ),
    'pure_vision': (
        'pure_vision_node.py', 'pure_vision_node',
        'pure_vision_params.yaml', '/lka/pure_vision/lane_center',
    ),
    'scnn': (
        'scnn_node.py', 'scnn_node',
        'scnn_params.yaml', '/lka/scnn/lane_center',
    ),
}

RECORD_TOPICS = [
    '/lka/lane_center',
    '/lka/gt/cross_track_m',
    '/carla/ego_vehicle/odometry',
    '/carla/ego_vehicle/vehicle_control_cmd',
    '/carla/weather_control',
]

BAGS_ROOT = '/home/peeradon/lka-carla-yolo/bags/closed_loop'


def launch_setup(context, *args, **kwargs):
    method  = LaunchConfiguration('method').perform(context)
    weather = LaunchConfiguration('weather').perform(context)

    if method not in METHOD_MAP:
        raise ValueError(f"method must be one of {list(METHOD_MAP)}; got '{method}'")
    if weather not in WEATHER_PRESETS:
        raise ValueError(f"weather must be one of {list(WEATHER_PRESETS)}; got '{weather}'")

    executable, node_name, config_yaml, output_topic = METHOD_MAP[method]
    pkg_dir = get_package_share_directory('lka_perception')
    config_path = os.path.join(pkg_dir, 'config', config_yaml)

    # ── Perception node ───────────────────────────────────────────────
    perception_node = Node(
        package='lka_perception',
        executable=executable,
        name=node_name,
        output='screen',
        parameters=[
            config_path,
            {'use_sim_time': True, 'enable_hysteresis': True},
        ],
        remappings=[(output_topic, '/lka/lane_center')],
    )

    # ── Pure Pursuit controller ───────────────────────────────────────
    controller_node = Node(
        package='lka_control',
        executable='lka_controller_node.py',
        name='pure_pursuit_node',
        output='screen',
        parameters=[{
            'use_sim_time':     True,
            'wheel_base':       3.0046,
            'lane_width':       3.5,
            'min_lookahead':    3.0,
            'max_lookahead':    10.0,
            'ld_velocity_ratio': 2.4,
            'max_steer_rad':    1.2217,
            'throttle':         0.3,
        }],
    )

    # ── Weather setter (delayed 3 s to let CARLA settle) ─────────────
    set_weather = TimerAction(
        period=3.0,
        actions=[
            LogInfo(msg=f'[closed_loop] Setting weather → {weather.upper()}'),
            ExecuteProcess(
                cmd=[
                    'ros2', 'topic', 'pub', '--once',
                    '/carla/weather_control',
                    'carla_msgs/msg/CarlaWeatherParameters',
                    WEATHER_PRESETS[weather],
                ],
                output='screen',
            ),
        ],
    )

    # ── Bag recorder ─────────────────────────────────────────────────
    bag_path = os.path.join(BAGS_ROOT, f'{method}_{weather}')
    recorder = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'record',
            '--output', bag_path,
            '--storage', 'sqlite3',
        ] + RECORD_TOPICS,
        output='screen',
    )

    return [
        LogInfo(msg=f'[closed_loop] method={method}  weather={weather}  bag → {bag_path}'),
        perception_node,
        controller_node,
        set_weather,
        recorder,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'method',
            description='Perception method: yolo | pure_vision | scnn',
        ),
        DeclareLaunchArgument(
            'weather',
            default_value='clear',
            description='Weather condition: clear | rain | fog | night',
        ),
        OpaqueFunction(function=launch_setup),
    ])
