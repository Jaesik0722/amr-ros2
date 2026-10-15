# amr_isaac: Isaac Sim 5.1 simulation

Simulates the towing AMR in Isaac Sim with the **same ROS 2 interface as the real
robot**, so Nav2 and every higher-level node run unchanged on either.

| Topic / service | Real robot | Simulator |
|---|---|---|
| `/cmd_vel` (Twist) | amr_base | same limits (0.5 m/s, 1.5 rad/s) and 0.5 s timeout |
| `/odom`, `/tf` odom→base_footprint | amr_base (wheel odometry) | ground truth, planar, 50 Hz |
| `/joint_states` | amr_base + amr_hitch | wheels + hitch; hitch effort = joint torque [Nm] |
| `/amr_hitch/command` (JointState) | amr_hitch | position targets, ±90° |
| `/amr_hitch/set_torque` (SetBool) | amr_hitch | off = passive (free) hitch |
| `/livox/lidar` (PointCloud2, `livox_frame`) | livox_ros_driver2 | RTX lidar, 10 Hz full scans |
| `/{front,rear}_camera/color/image_raw`, `.../camera_info`, `.../depth/image_rect_raw` | realsense2_camera | 640x480 @ 10 Hz, depth 32FC1 [m] |
| `/clock` | – | sim time, use `use_sim_time:=true` |
| `/amr_hitch/set_yaw_passive` (SetBool) | amr_hitch | yaw joint free while towing |
| `/battery_state` (BatteryState) | battery / OC-251 | +5 A while the RC-100 is on the TC-200 (±4 cm, ±10°) |
| `/indoor_gps/cart` (PoseStamped, map) | indoor GPS | cart pose, 5 Hz, 2 cm / 1° noise |
| `/cart/latched` (Bool) | – | ground truth of the hook coupling |

This directory is not a colcon package (`COLCON_IGNORE`): it runs inside Isaac Sim's
Python 3.11 with Isaac's bundled ROS 2 Jazzy libraries, not the system Python 3.12.

## Requirements

- Isaac Sim 5.1 installed with pip in a conda env (default: `~/anaconda3/envs/isaaclab`;
  override with `ISAAC_PYTHON=/path/to/python`).
- The workspace built (`colcon build`), so `amr_description` is found.
- Same `ROS_DOMAIN_ID` in the terminal running the simulator and the ROS side.

## Run

```bash
# terminal 1: simulator (first start converts the URDF, ~1 min; cached in ~/.cache/amr_isaac)
source ~/ros2_ws/install/setup.bash
./amr_isaac/run_sim.sh                      # GUI, simple warehouse
./amr_isaac/run_sim.sh --headless --env empty --no-cameras

# terminal 2: ROS side (robot_state_publisher, scan slice, RViz, SLAM/Nav2)
ros2 launch amr_bringup sim.launch.py slam:=true nav:=true
# drive: RViz "2D Goal Pose", or
ros2 run teleop_twist_keyboard teleop_twist_keyboard
# hitch:
ros2 topic pub --once /amr_hitch/command sensor_msgs/msg/JointState \
  "{name: [hitch_yaw_joint, hitch_pitch_joint], position: [0.5, 0.0]}"
```

`run_sim.sh` options: `--headless`, `--env warehouse|empty`, `--no-cameras`,
`--no-lidar`, `--chase-camera` (third-person view on `/sim/chase_camera/image_raw`),
`--overview-camera X Y Z TX TY TZ` (fixed view on `/sim/overview_camera/image_raw`),
`--no-realtime`, `--reimport` (force URDF reconversion),
`--selftest` (scripted drive/hitch checks, exits 0 on pass).

Real-site twin (see [docs/gaussian_splatting.md](../docs/gaussian_splatting.md)):
`--env-usd A.usdz [B.usd ...]` references visual environments (e.g. a 3DGRUT NuRec
USDZ) instead of `--env`; `--env-collision mesh.usda` adds static triangle-mesh colliders
that the RTX lidar sees but cameras do not (`primvars:hideForCamera`; an invisible prim
would be ignored by the lidar too). `--env-collision-visible` shows the mesh. A hidden
ground plane at z = 0 carries the robot.

Demo video: `tools/demo_video/` (`record.sh` runs the docking/towing scenario and grabs
all views, `compose.py` builds the captioned 2x2 video from its event log).

## Model notes

- Robot: imported from `amr_description` (CAD meshes, CAD-based inertias). Massless
  frame links get 1 g so they do not distort the dynamics.
- Wheels: velocity-controlled torque drives limited to the MDH300 rated torque
  (11 Nm per wheel; the peak torque is not published), tyre friction 1.0. The collider is a true cylinder narrowed to a 12 mm contact
  band, because the CAD tyre has a round (torus) profile; a full 46 mm wide contact adds
  scrub torque when turning in place. Casters: frictionless spheres.
- Hitch: position drives, kp 50 Nm/rad, kd 2 Nm s/rad, max 10.6 Nm (XM540-W270 stall).
- Physics 200 Hz, rendering 10 Hz (= lidar scan rate, so each cloud is one revolution).
- Mid-360: Isaac Sim has no Livox profile. `mid360.py` approximates it with a rotary
  pattern over the same FOV (360° x -7..52°, 40 channels, ~200k pts/s, 0.1-40 m). The
  real sensor's non-repetitive pattern is not modelled.
- `--selftest` checks: 0.3 m/s straight, 0.5 rad/s in place, cmd_vel timeout stop,
  hitch position tracking. All match the command within 1 % (empty floor and warehouse).

## Cart and charging station

`world.py` adds them when `amr_docking` is built (`--no-cart`, `--no-charger`,
`--cart-pose X Y YAW` default 3.0 -7.5 1.571, `--charger-pose X Y YAW` default 8.3 -1.0 3.142).
Marker ids/positions and the tow bar come from `amr_docking/config/docking_targets.yaml`.

- Cart: folding platform cart (0.84 x 0.56 m green deck, top 0.26 m, white handle)
  loaded with three cardboard boxes, 22 kg in total; rear wheels on axles, front casters as
  frictionless spheres; ArUco 0/1 on a plate under the front deck edge; tow bar (visual
  only). Linear/angular damping 1.0 stands in for rolling resistance (~7 N at 0.3 m/s).
- Coupling: raising the hitch pitch above 0.1 rad with the hook within 3 cm (horizontal)
  of the bar enables a revolute joint (bar axis) between hitch_pitch_link and the cart;
  lowering it below 0 releases. The hook then carries the tongue load, which is what
  `dock_cart` checks through the hitch effort. The simulator reports the hitch effort as
  the servo drive torque kp·error − kd·velocity (like the Dynamixel current): ~0.01 Nm
  without a cart, 1.2-3 Nm hooked.
- Charger: WiBotic TC-200 antenna (215 x 215 x 22 mm, centre 0.117 m) and TR-302 box on a
  stand, datasheet sizes (no public CAD); ArUco 10/11 left and right of the antenna.

## Known limitations / next steps

- Depth is published as 32FC1 metres; the realsense driver publishes 16UC1 millimetres.
- Isaac Sim may print a segfault on exit; it happens after shutdown and is harmless.
