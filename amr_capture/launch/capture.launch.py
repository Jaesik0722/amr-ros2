"""Record the Gaussian Splatting capture rig: ZED 2 + Mid-360 (+ optional T265).

    ros2 launch amr_capture capture.launch.py [bag:=~/gs_capture/site_a] [t265:=true]
        [zed:=true] [livox:=true] [record:=true] [svo:=true]

Records one rosbag (mcap) with the topics in config/capture.yaml and, with svo:=true,
an SVO next to it (full-rate, H.265, as a backup source of images). Keep recording
through all mast heights in one bag so FAST-LIO gives one trajectory for everything.
Drivers that are not installed are skipped with a warning.
"""

import os
import time

from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
                            LogInfo, OpaqueFunction, TimerAction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import yaml


def share(pkg):
    try:
        return get_package_share_directory(pkg)
    except PackageNotFoundError:
        return None


def setup(context):
    arg = lambda name: LaunchConfiguration(name).perform(context)   # noqa: E731
    on = lambda name: arg(name).lower() == 'true'                    # noqa: E731
    cfg_dir = os.path.join(share('amr_capture'), 'config')
    with open(os.path.join(cfg_dir, 'capture.yaml')) as f:
        cfg = yaml.safe_load(f)
    bag = os.path.expanduser(arg('bag')) or os.path.expanduser(
        time.strftime('~/gs_capture/%Y%m%d_%H%M%S'))
    actions = []

    if on('livox'):
        if share('livox_ros_driver2') is None:
            actions.append(LogInfo(msg='[capture] livox_ros_driver2 not found, skipped'))
        else:
            actions.append(Node(
                package='livox_ros_driver2', executable='livox_ros_driver2_node',
                name='livox_lidar_publisher', output='screen',
                parameters=[{
                    'xfer_format': 1,          # CustomMsg: per-point time for FAST-LIO
                    'multi_topic': 0, 'data_src': 0, 'publish_freq': 10.0,
                    'output_data_type': 0, 'frame_id': 'livox_frame',
                    'user_config_path': os.path.join(share('amr_bringup'), 'config',
                                                     'MID360_config.json'),
                }]))
    if on('zed'):
        zed = share('zed_wrapper')
        if zed is None:
            actions.append(LogInfo(msg='[capture] zed_wrapper not found, skipped'))
        else:
            actions.append(IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(zed, 'launch',
                                                           'zed_camera.launch.py')),
                launch_arguments={
                    'camera_model': 'zed2',
                    'ros_params_override_path': os.path.join(cfg_dir, 'zed2_capture.yaml'),
                }.items()))
            if on('svo') and on('record'):
                os.makedirs(bag, exist_ok=True)
                svo = os.path.join(bag, 'zed2.svo2')
                actions.append(TimerAction(period=8.0, actions=[ExecuteProcess(
                    cmd=['ros2', 'service', 'call', '/zed/zed_node/start_svo_rec',
                         'zed_msgs/srv/StartSvoRec',
                         f"{{svo_filename: '{svo}', compression_mode: 2}}"],
                    output='screen')]))
    if on('t265'):
        rs = share('realsense2_camera')
        if rs is None:
            actions.append(LogInfo(msg='[capture] realsense2_camera not found, skipped'))
        else:
            actions.append(Node(
                package='realsense2_camera', executable='realsense2_camera_node',
                namespace='t265', name='t265', output='screen',
                parameters=[{'device_type': 't265', 'enable_pose': True,
                             'enable_fisheye1': False, 'enable_fisheye2': False}]))
    if on('record'):
        os.makedirs(os.path.dirname(bag.rstrip('/')) or '.', exist_ok=True)
        actions.append(TimerAction(period=5.0, actions=[ExecuteProcess(
            cmd=['ros2', 'bag', 'record', '-s', 'mcap', '-o', os.path.join(bag, 'bag')]
            + cfg['record']['topics'], output='screen')]))
        actions.append(LogInfo(msg=f'[capture] recording to {bag}'))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('bag', default_value='',
                              description='output folder (default ~/gs_capture/<time>)'),
        DeclareLaunchArgument('zed', default_value='true'),
        DeclareLaunchArgument('livox', default_value='true'),
        DeclareLaunchArgument('t265', default_value='false'),
        DeclareLaunchArgument('record', default_value='true'),
        DeclareLaunchArgument('svo', default_value='true'),
        OpaqueFunction(function=setup),
    ])
