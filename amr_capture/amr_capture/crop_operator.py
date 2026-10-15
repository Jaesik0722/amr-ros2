"""Remove the person pushing the capture cart from the Mid-360 cloud.

The operator walks behind the rig and moves with it, so FAST-LIO would map them as a
smear and they would end up in the Gaussian initialisation and the collision mesh.
Points inside an axis-aligned box in the lidar frame are dropped.

    ros2 run amr_capture crop_operator     # livox CustomMsg /livox/lidar -> /livox/lidar_crop
    ros2 run amr_capture crop_operator --ros-args -p msg_type:=pointcloud2

Usually run while replaying the bag for FAST-LIO (docs/gaussian_splatting.md).
"""

import os

from ament_index_python.packages import get_package_share_directory
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
import yaml


def inside_box(xyz, box_min, box_max):
    """Boolean mask of the rows of an (N, 3) array inside [box_min, box_max]."""
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    return np.all((xyz >= np.asarray(box_min)) & (xyz <= np.asarray(box_max)), axis=1)


def crop_pointcloud2(msg, box_min, box_max):
    fields = {f.name: f.offset for f in msg.fields}
    raw = np.frombuffer(msg.data, np.uint8).reshape(-1, msg.point_step)
    xyz = np.stack([raw[:, fields[k]:fields[k] + 4].copy().view(np.float32)[:, 0]
                    for k in ('x', 'y', 'z')], axis=1)
    keep = raw[~inside_box(xyz, box_min, box_max)]
    return PointCloud2(header=msg.header, height=1, width=len(keep), fields=msg.fields,
                       is_bigendian=msg.is_bigendian, point_step=msg.point_step,
                       row_step=msg.point_step * len(keep), is_dense=msg.is_dense,
                       data=keep.tobytes())


class CropOperator(Node):

    def __init__(self):
        super().__init__('crop_operator')
        cfg_file = os.path.join(get_package_share_directory('amr_capture'), 'config',
                                'capture.yaml')
        with open(cfg_file) as f:
            box = yaml.safe_load(f)['crop_operator']
        self.declare_parameter('msg_type', 'livox')
        self.declare_parameter('box_min', [float(v) for v in box['box_min']])
        self.declare_parameter('box_max', [float(v) for v in box['box_max']])
        self.box_min = np.array(self.get_parameter('box_min').value)
        self.box_max = np.array(self.get_parameter('box_max').value)
        if self.get_parameter('msg_type').value == 'livox':
            from livox_ros_driver2.msg import CustomMsg
            msg_type, callback = CustomMsg, self._on_custom
        else:
            msg_type, callback = PointCloud2, self._on_cloud2
        self.pub = self.create_publisher(msg_type, 'livox/lidar_crop', qos_profile_sensor_data)
        self.create_subscription(msg_type, 'livox/lidar', callback, qos_profile_sensor_data)

    def _on_cloud2(self, msg):
        self.pub.publish(crop_pointcloud2(msg, self.box_min, self.box_max))

    def _on_custom(self, msg):
        xyz = np.array([(p.x, p.y, p.z) for p in msg.points], dtype=float)
        if len(xyz):
            mask = ~inside_box(xyz, self.box_min, self.box_max)
            msg.points = [p for p, k in zip(msg.points, mask) if k]
            msg.point_num = len(msg.points)
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = CropOperator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
