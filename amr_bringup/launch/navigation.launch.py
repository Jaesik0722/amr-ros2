"""Nav2 (localization + navigation) with this robot's parameters."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    nav2_launch = os.path.join(get_package_share_directory('nav2_bringup'), 'launch')
    default_params = os.path.join(
        get_package_share_directory('amr_bringup'), 'config', 'nav2_params.yaml')

    args = {
        'map': LaunchConfiguration('map'),
        'params_file': LaunchConfiguration('params_file'),
        'use_sim_time': LaunchConfiguration('use_sim_time'),
    }

    return LaunchDescription([
        DeclareLaunchArgument('map', description='Full path to the map yaml'),
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(nav2_launch, 'localization_launch.py')),
            launch_arguments=args.items()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(nav2_launch, 'navigation_launch.py')),
            launch_arguments=args.items()),
    ])
