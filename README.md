# AMR ROS 2 (Jazzy)

ROS 2 Jazzy packages for a differential-drive AMR with a 2-DOF (yaw/pitch)
actuated towing hitch, a Livox Mid-360 lidar and an MD Robot motor driver.
All robot-specific code is Python (`ament_python`).

| Package | Contents |
|---|---|
| `amr_base` | MD Robot RS485 driver: `cmd_vel` → wheels, encoders → `odom`, `/tf`, `joint_states`, motor currents |
| `amr_hitch` | Dynamixel X-series driver for the hitch: `joint_states` (position, velocity, current), `~/command`, `~/set_torque` |
| `amr_description` | URDF/xacro generated from the CAD (`Product1_V0.2.stp`), see below |
| `amr_bringup` | Launch files, Nav2 parameters (MPPI), Livox and scan-slice configuration |
| `amr_docking` | ArUco docking: charger (Nav2 docking_server, forward) and carts (reverse docking + hitch-current check), towing mode (cart footprint, cart cloud filter, towing speed limits) |
| `amr_capture` | Gaussian Splatting capture rig (ZED 2 + Mid-360 + T265 on a cart): recording launch, operator crop filter, FAST-LIO config, bag → COLMAP/nerfstudio export, lidar map → collision mesh. See [docs/gaussian_splatting.md](docs/gaussian_splatting.md) |
| `amr_isaac` | Isaac Sim 5.1 simulation with the same ROS 2 interface (not a colcon package), see [amr_isaac/README.md](amr_isaac/README.md) |

## Documentation

- [docs/setup_new_pc.md](docs/setup_new_pc.md): setting up a new PC (ROS 2, Isaac Sim, real robot, capture rig), in Korean
- [docs/gaussian_splatting.md](docs/gaussian_splatting.md): real site → Gaussian Splatting → Isaac Sim twin
- [docs/research_plan.md](docs/research_plan.md): planned papers
- [amr_isaac/README.md](amr_isaac/README.md): simulation

## Setup (Ubuntu 24.04, ROS 2 Jazzy)

```bash
mkdir -p ~/amr_ws/src && cd ~/amr_ws/src
git clone <this repo>
# Livox Mid-360: build Livox-SDK2 and livox_ros_driver2 from source,
# see https://github.com/Livox-SDK/livox_ros_driver2
cd ~/amr_ws
rosdep install --from-paths src --ignore-src -y
colcon build --symlink-install
source install/setup.bash
```

## Usage

```bash
ros2 launch amr_bringup bringup.launch.py              # hardware
ros2 launch amr_bringup bringup.launch.py lidar:=false hitch:=false
ros2 launch amr_bringup navigation.launch.py map:=/path/to/map.yaml
ros2 launch amr_description display.launch.py          # model only, RViz
```

Hitch command example:

```bash
ros2 topic pub --once /amr_hitch/command sensor_msgs/msg/JointState \
  "{name: [hitch_yaw_joint], position: [0.3]}"
```

## Interfaces

- `amr_base`: subscribes `cmd_vel` (`Twist`, or `TwistStamped` with
  `use_stamped_cmd_vel: true`); publishes `odom`, `odom → base_footprint` TF,
  `joint_states`, `~/motor_state` (effort field = motor current in A).
  Stops the wheels if no command arrives within `cmd_vel_timeout`.
- `amr_hitch`: publishes `joint_states` (effort field = motor current in A).
  `operating_mode: current_based_position` makes the hitch compliant.

The Isaac Sim simulation (`amr_isaac`) provides the same interface plus `/clock`,
so Nav2 and higher-level nodes run unchanged:

```bash
./amr_isaac/run_sim.sh                                    # terminal 1
ros2 launch amr_bringup sim.launch.py slam:=true nav:=true  # terminal 2
```

## Charging and carts (amr_docking)

```bash
ros2 run amr_docking dock_charger     # Nav2 to the station, ArUco docking, waits for charge
ros2 run amr_docking undock_charger
ros2 run amr_docking dock_cart        # cart pose from /indoor_gps/cart, reverse onto the
                                      # rear-camera ArUco markers, raise the hook, check
                                      # the hitch current, switch Nav2 to towing mode
ros2 run amr_docking release_cart
```

- Marker ids, sizes and positions, and the docked robot poses: `amr_docking/config/docking_targets.yaml`
  (printable markers in `amr_docking/markers/`, 4x4_50, 80 mm black square).
- Charger: WiBotic TR-302 + TC-200 antenna, markers 10/11 left and right of the antenna.
  Robot side: OC-251 onboard charger + RC-100 receiver (front). Set the station pose in
  `nav2_params.yaml` (`docking_server.home_charger`); docking needs `/battery_state`
  (current > 0.5 A while charging).
- Cart: markers 0/1 on the front lower beam, tow bar 0.58 m ahead of the cart centre at
  0.115 m. The hook is "engaged" when the hitch pitch effort exceeds `engage_effort`
  (amr_docking `dock_cart` parameter: amps on the robot, Nm in simulation; tune it).
- While towing, `cart_towing` publishes the robot + cart footprint, removes the cart from
  the lidar cloud (`/livox/lidar_filtered`, used by the scan slice and costmaps), limits
  MPPI to 0.3 m/s without reversing, and frees the hitch yaw (`amr_hitch/set_yaw_passive`).

## Robot model from CAD

Meshes and inertials are generated from the STEP assembly; joint origins are
measured from cylindrical faces and kept in `amr_description/tools/links.yaml`.

```bash
python3 -m venv cadenv && cadenv/bin/pip install --only-binary :all: \
  cadquery-ocp==7.8.1.1.post1 trimesh pycollada pyyaml
cd amr_description/tools
../../cadenv/bin/python step_dump.py /path/Product1_V0.2.stp /tmp/dump
../../cadenv/bin/python step_to_urdf.py /tmp/dump links.yaml ../meshes ../urdf/amr_inertials.xacro
```

Masses are partly **assumed** (structure as aluminium, battery block by density, ...);
the hub motors use the MDH300 datasheet (3.8 kg each).
Weigh the robot and set `measured_total_mass` in `links.yaml`, or edit the
per-part rules, then rerun `step_to_urdf.py`.

| Frame | Position from `base_link` (wheel axle centre) |
|---|---|
| `hitch_yaw_joint` | 176.8 mm behind, vertical axis |
| `hitch_pitch_joint` | 42.2 mm behind the yaw axis, lateral axis |
| `hitch_coupling_link` | coupling pin hole at the hook tip |
| `livox_frame` | Mid-360 mounting face centre |
| `front_camera_link`, `rear_camera_link` | D435, realsense2_description convention |
| `charging_rx_link` | wireless charging coil face |

## What has not been measured on the robot

Everything here runs in the Isaac Sim simulation. The hardware drivers are written and
unit-tested, but the full stack has not been run on the robot, so the following are
assumptions carried in the configuration rather than measurements, and each will move
once the robot is on the bench:

- Masses are from the CAD with some assumed values; wheel radius and track want a
  calibration drive.
- Hitch joint limits are the previous URDF's; the servo ids and zero offsets in
  `amr_hitch/config/hitch.yaml` have never been read off the bus.
- The Mid-360 addresses in `amr_bringup/config/MID360_config.json` are placeholders, and
  the office LAN shares the sensor's default subnet; put the lidar on its own.
- `livox_joint` is the mounting face, not the point cloud origin.
- `engage_effort` is in N.m in simulation and in amps on the robot, so the hook-engaged
  threshold has to be set from a measurement.
- The indoor GPS frame is not yet registered against the map.

One known geometric limitation is in the build rather than the configuration: the two
rear Bluetooth antennas sit inside the Mid-360's field of view, about +-160 degrees
behind the robot.
- [ ] WiBotic STEP models (not public; ask WiBotic) to replace the datasheet-size boxes
