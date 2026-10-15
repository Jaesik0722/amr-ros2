"""Docking: ArUco detectors (charger: front camera, cart: rear camera).

The charger uses Nav2's docking_server (navigation launch); carts are docked by
amr_docking dock_cart (reverse docking on the rear-camera detections).

  ros2 run amr_docking dock_charger / undock_charger / dock_cart / release_cart
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    def detector(target, camera, output):
        return Node(
            package='amr_docking', executable='aruco_detector',
            name=f'{target}_aruco_detector', output='screen',
            parameters=[{'target': target, 'use_sim_time': use_sim_time}],
            remappings=[('image', f'/{camera}/color/image_raw'),
                        ('camera_info', f'/{camera}/color/camera_info'),
                        ('detected_dock_pose', output)],
        )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        detector('charger', 'front_camera', '/detected_dock_pose'),
        detector('cart', 'rear_camera', '/cart/detected_dock_pose'),
    ])
