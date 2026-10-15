"""Livox Mid-360 driver and a 2D scan slice for AMCL / costmaps."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(get_package_share_directory('amr_bringup'), 'config')

    livox = Node(
        package='livox_ros_driver2',
        executable='livox_ros_driver2_node',
        name='livox_lidar_publisher',
        output='screen',
        parameters=[{
            'xfer_format': 0,  # 0: sensor_msgs/PointCloud2, 1: livox CustomMsg (FAST-LIO).
            'multi_topic': 0,
            'data_src': 0,
            'publish_freq': 10.0,
            'output_data_type': 0,
            'frame_id': 'livox_frame',
            'user_config_path': os.path.join(config, 'MID360_config.json'),
        }],
    )

    to_scan = Node(
        package='pointcloud_to_laserscan',
        executable='pointcloud_to_laserscan_node',
        name='pointcloud_to_laserscan',
        parameters=[os.path.join(config, 'pointcloud_to_laserscan.yaml')],
        remappings=[('cloud_in', '/livox/lidar_filtered'), ('scan', '/scan')],
    )

    # Removes a towed cart from the cloud (livox/lidar_filtered) and handles towing mode.
    towing = Node(package='amr_docking', executable='cart_towing', output='screen')

    return LaunchDescription([livox, towing, to_scan])
