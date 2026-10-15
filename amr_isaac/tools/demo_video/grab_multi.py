"""Save the sim cameras at 10 Hz as JPEGs: grab_multi.py OUT_DIR SECONDS (system ROS)."""
import os
import sys
import time

import numpy as np
from PIL import Image as PImage
import rclpy
from sensor_msgs.msg import Image

out, dur = sys.argv[1], float(sys.argv[2])
topics = {'overview': '/sim/overview_camera/image_raw', 'chase': '/sim/chase_camera/image_raw',
          'front': '/front_camera/color/image_raw', 'rear': '/rear_camera/color/image_raw'}
rclpy.init()
node = rclpy.create_node('grab_multi')
last = {}
for k, t in topics.items():
    os.makedirs(f'{out}/{k}', exist_ok=True)
    node.create_subscription(Image, t, lambda m, k=k: last.__setitem__(k, m), 2)
while len(last) < len(topics):
    rclpy.spin_once(node, timeout_sec=0.5)
open(f'{out}/started', 'w').close()
t0, i = time.time(), 0
try:
    while time.time() - t0 < dur:
        rclpy.spin_once(node, timeout_sec=0.01)
        if time.time() - t0 >= i / 10.0:
            for k, m in last.items():
                img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)[:, :, :3]
                PImage.fromarray(img).save(f'{out}/{k}/f{i:05d}.jpg', quality=88)
            i += 1
except Exception:   # shut down by record.sh
    pass
print('frames', i)
