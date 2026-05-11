import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                             Shutdown, RegisterEventHandler, GroupAction)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg = "lka_bringup"

    host = DeclareLaunchArgument(
        'host',
        default_value='localhost'
    )

    port = DeclareLaunchArgument(
        'port',
        default_value='2000',
        description='CARLA port'
    )
    town = DeclareLaunchArgument(
        'town',
        default_value='Town01'
    )
    timeout = DeclareLaunchArgument(
        'timeout',
        default_value='15.0'
    )
    passive = DeclareLaunchArgument(
        'passive',
        default_value='False'
    )
    synchronous_mode = DeclareLaunchArgument(
        'synchronous_mode',
        default_value='True'
    )
    synchronous_mode_wait_for_vehicle_control_command = DeclareLaunchArgument(
        'synchronous_mode_wait_for_vehicle_control_command',
        default_value='False'
    )
    fixed_delta_seconds = DeclareLaunchArgument(
        'fixed_delta_seconds',
        default_value='0.05' # 20 FPS
    )
    register_all_sensors = DeclareLaunchArgument(
        'register_all_sensors',
        default_value='True'
    )
    ego_vehicle_role_name = DeclareLaunchArgument(
        'ego_vehicle_role_name',
        default_value='ego_vehicle'
    )
    
    carla_bridge = Node(
        package='carla_ros_bridge',
        executable='bridge',
        name='carla_ros_bridge',
        output='screen',
        emulate_tty='True',
        on_exit= Shutdown(),
        parameters=[
            {
                'use_sim_time': True
            },
            {
                'host': LaunchConfiguration('host')
            },
            {
                'port': LaunchConfiguration('port')
            },
            {
                'timeout': LaunchConfiguration('timeout')
            },
            {
                'passive': LaunchConfiguration('passive')
            },
            {
                'synchronous_mode': LaunchConfiguration('synchronous_mode')
            },
            {
                'synchronous_mode_wait_for_vehicle_control_command': LaunchConfiguration('synchronous_mode_wait_for_vehicle_control_command')
            },
            {
                'fixed_delta_seconds': LaunchConfiguration('fixed_delta_seconds')
            },
            {
                'town': LaunchConfiguration('town')
            },
            {
                'register_all_sensors': LaunchConfiguration('register_all_sensors')
            },
            {
                'ego_vehicle_role_name': LaunchConfiguration('ego_vehicle_role_name')
            }
        ],

    )

    spawn_entity = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory(pkg),
                'launch', 'carla_spawn_vehicle.launch.py'
            )
        )
    )


    gt_node = Node(
        package='lka_perception',
        executable='gt_node.py',
        name='gt_node',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    manual_control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(
                'carla_manual_control'), 'carla_manual_control.launch.py')
        ),
        launch_arguments={'role_name': 'ego_vehicle'}.items(),
    )

    ld = LaunchDescription()
    ld.add_action(host)
    ld.add_action(port)
    ld.add_action(timeout)
    ld.add_action(passive)
    ld.add_action(synchronous_mode)
    ld.add_action(synchronous_mode_wait_for_vehicle_control_command)
    ld.add_action(fixed_delta_seconds)
    ld.add_action(town)
    ld.add_action(register_all_sensors)
    ld.add_action(ego_vehicle_role_name)
    ld.add_action(carla_bridge)
    ld.add_action(spawn_entity)
    ld.add_action(gt_node)
    ld.add_action(manual_control)

    return ld