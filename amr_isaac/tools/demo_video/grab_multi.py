"""Save the sim cameras at 10 Hz as JPEGs (system ROS).

  grab_multi.py OUT_DIR SECONDS [overview,chase,front,rear]

A world started with --no-cameras publishes only the two sim views, so the grabber
waits a bounded time and then records whichever cameras actually arrived instead of
blocking forever on one that will never publish.
"""
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
if len(sys.argv) > 3:
    topics = {k: v for k, v in topics.items() if k in sys.argv[3].split(',')}
rclpy.init()
node = rclpy.create_node('grab_multi')
last = {}
for k, t in topics.items():
    os.makedirs(f'{out}/{k}', exist_ok=True)
    node.create_subscription(Image, t, lambda m, k=k: last.__setitem__(k, m), 2)
deadline = time.time() + 60.0
while len(last) < len(topics) and time.time() < deadline:
    rclpy.spin_once(node, timeout_sec=0.5)
if not last:
    sys.exit('no camera published within 60 s')
print('recording', ' '.join(sorted(last)), flush=True)
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
