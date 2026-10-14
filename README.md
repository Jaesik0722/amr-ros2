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
| `amr_isaac` | Isaac Sim 5.1 simulation with the same ROS 2 interface (not a colcon package), see [amr_isaac/README.md](amr_isaac/README.md) |

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

## TODO

- [ ] Measure masses (see above); verify wheel radius / track by calibration
- [ ] Hitch joint limits
- [ ] Livox mounting pose, IP addresses in `config/MID360_config.json`
- [ ] Hitch servo IDs and zero offsets
- [ ] Mid-360 point cloud origin offset above the mounting face (`livox_joint`)
- [ ] The rear Bluetooth antennas are inside the Mid-360 field of view (rear, about +-160 deg)
- [ ] Filter the towed cart out of the lidar cloud; cart-aware footprint
- [ ] Simulation: towed cart model (phase 2)
