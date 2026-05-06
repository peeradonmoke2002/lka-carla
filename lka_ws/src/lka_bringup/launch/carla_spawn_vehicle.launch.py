import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource


def generate_launch_description():
    pkg = "lka_bringup"
    
    object_file = os.path.join(
        get_package_share_directory(pkg),
        'config','objects.json'
    )

    spawn_entity = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(
                'carla_spawn_objects'), 'carla_spawn_objects.launch.py')
        ),
        launch_arguments={
            'objects_definition_file': object_file,
            'spawn_sensors_only': 'False',
            'spawn_point_ego_vehicle': 'None',
        }.items()
    )

    init_pose = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('carla_spawn_objects'),
                'set_initial_pose.launch.py'
            )
        ),
        launch_arguments={
            'role_name': 'ego_vehicle',
            'control_id': 'control'
        }.items()
    )
    ld = LaunchDescription()
    ld.add_action(spawn_entity)
    ld.add_action(init_pose)

    return ld