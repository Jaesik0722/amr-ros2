"""ROS 2 side of the Isaac Sim simulation (start amr_isaac/run_sim.sh first).

The simulator provides what amr_base, amr_hitch and the Livox driver provide on the
robot (/cmd_vel, /odom, /tf, /joint_states, /livox/lidar, /amr_hitch/...), plus
/clock. This starts the rest with use_sim_time:=true.

  ros2 launch amr_bringup sim.launch.py                    # RViz only
  ros2 launch amr_bringup sim.launch.py slam:=true nav:=true   # map while navigating
  ros2 launch amr_bringup sim.launch.py nav:=true map:=/path/map.yaml   # AMCL on a map
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    bringup = get_package_share_directory('amr_bringup')
    description = get_package_share_directory('amr_description')
    nav2 = os.path.join(get_package_share_directory('nav2_bringup'), 'launch')
    slam = LaunchConfiguration('slam')
    nav = LaunchConfiguration('nav')
    params = os.path.join(bringup, 'config', 'nav2_params.yaml')
    slam_params = os.path.join(bringup, 'config', 'slam_toolbox.yaml')
    sim_time = {'use_sim_time': 'true'}
    nav_with_slam = PythonExpression(["'", nav, "' == 'true' and '", slam, "' == 'true'"])
    nav_on_map = PythonExpression(["'", nav, "' == 'true' and '", slam, "' != 'true'"])

    def include(path, args=None, condition=None):
        return IncludeLaunchDescription(PythonLaunchDescriptionSource(path),
                                        launch_arguments=(args or {}).items(),
                                        condition=condition)

    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('slam', default_value='false', description='slam_toolbox mapping'),
        DeclareLaunchArgument('nav', default_value='false', description='Nav2'),
        DeclareLaunchArgument('map', default_value='', description='map yaml (nav without slam)'),
        DeclareLaunchArgument('joystick', default_value='false'),

        include(os.path.join(description, 'launch', 'description.launch.py'), sim_time),
        Node(
            package='pointcloud_to_laserscan',
            executable='pointcloud_to_laserscan_node',
            name='pointcloud_to_laserscan',
            parameters=[os.path.join(bringup, 'config', 'pointcloud_to_laserscan.yaml'),
                        {'use_sim_time': True}],
            remappings=[('cloud_in', '/livox/lidar_filtered'), ('scan', '/scan')],
        ),
        # Removes a towed cart from the cloud and switches Nav2 to towing mode.
        Node(
            package='amr_docking', executable='cart_towing', output='screen',
            parameters=[{'use_sim_time': True}],
        ),
        Node(
            package='rviz2', executable='rviz2', output='log',
            arguments=['-d', os.path.join(bringup, 'rviz', 'sim.rviz')],
            parameters=[{'use_sim_time': True}],
            condition=IfCondition(LaunchConfiguration('rviz')),
        ),
        include(os.path.join(get_package_share_directory('slam_toolbox'), 'launch',
                             'online_async_launch.py'),
                {**sim_time, 'slam_params_file': slam_params},
                IfCondition(slam)),
        # Nav2 with SLAM: navigation only (slam_toolbox owns map->odom).
        include(os.path.join(nav2, 'navigation_launch.py'),
                {**sim_time, 'params_file': params},
                IfCondition(nav_with_slam)),
        # Nav2 on a saved map: AMCL + navigation.
        include(os.path.join(bringup, 'launch', 'navigation.launch.py'),
                {**sim_time, 'map': LaunchConfiguration('map'), 'params_file': params},
                IfCondition(nav_on_map)),
        # Charger / cart docking (ArUco detectors + cart docking server) with Nav2.
        include(os.path.join(bringup, 'launch', 'docking.launch.py'), sim_time,
                IfCondition(nav)),
        include(os.path.join(bringup, 'launch', 'joystick.launch.py'),
                condition=IfCondition(LaunchConfiguration('joystick'))),
    ])
