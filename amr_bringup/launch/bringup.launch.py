"""Robot hardware bringup: description, wheels, hitch, lidar, joystick."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bringup = get_package_share_directory('amr_bringup')
    description = get_package_share_directory('amr_description')

    def include(path, condition=None):
        return IncludeLaunchDescription(PythonLaunchDescriptionSource(path),
                                        condition=condition)

    return LaunchDescription([
        DeclareLaunchArgument('lidar', default_value='true'),
        DeclareLaunchArgument('hitch', default_value='true'),
        DeclareLaunchArgument('joystick', default_value='true'),
        include(os.path.join(description, 'launch', 'description.launch.py')),
        Node(
            package='amr_base',
            executable='base_node',
            output='screen',
            parameters=[os.path.join(
                get_package_share_directory('amr_base'), 'config', 'base.yaml')],
        ),
        Node(
            package='amr_hitch',
            executable='hitch_node',
            output='screen',
            parameters=[os.path.join(
                get_package_share_directory('amr_hitch'), 'config', 'hitch.yaml')],
            condition=IfCondition(LaunchConfiguration('hitch')),
        ),
        include(os.path.join(bringup, 'launch', 'lidar.launch.py'),
                IfCondition(LaunchConfiguration('lidar'))),
        include(os.path.join(bringup, 'launch', 'joystick.launch.py'),
                IfCondition(LaunchConfiguration('joystick'))),
    ])
