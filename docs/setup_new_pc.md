# 새 PC 세팅 가이드

시뮬레이션 실험용 PC(데스크톱), 실제 로봇 PC, Gaussian Splatting(GS) 촬영·학습 PC에서 필요한 것을
순서대로 정리했다. 위에서부터 차례로 하면 된다. 1~5는 모든 PC 공통이다.

## 0. 권장 사양
| 용도 | 최소 | 권장 |
|---|---|---|
| Isaac Sim 시뮬레이션 | RTX 3070 8 GB, 32 GB RAM | RTX 4080/4090 16~24 GB, 64 GB RAM, 8코어 이상 |
| GS 학습(3DGRUT) | RTX 3090 24 GB | RTX 4090 / 5090 24~32 GB |
| 촬영 카트 PC | ZED SDK가 도는 NVIDIA GPU 노트북 또는 Jetson Orin | USB3 포트 2개 이상, 유선 LAN(Mid-360), 2 TB SSD |

- 원래 PC(i7-9750H, RTX 2070 8 GB)에서는 Isaac Sim + Nav2 + RViz를 함께 돌리면 CPU가 부족해
  slam_toolbox의 map→odom이 늦어졌다. CPU 코어 수도 중요하다.
- OS는 **Ubuntu 24.04**(ROS 2 Jazzy 대상)이다.
- NVIDIA 드라이버는 Isaac Sim 5.1 문서의 권장 버전을 설치한다(`nvidia-smi`로 확인).

## 1. 기본 패키지와 ROS 2 Jazzy
```bash
sudo apt update && sudo apt install -y git curl build-essential python3-pip ffmpeg xvfb bc \
  software-properties-common
# ROS 2 Jazzy (https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)
sudo add-apt-repository universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
  sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update && sudo apt install -y ros-jazzy-desktop python3-colcon-common-extensions \
  python3-rosdep python3-vcstool
sudo rosdep init; rosdep update
```

`~/.bashrc`에 다음을 추가한다. 시뮬레이터와 ROS 쪽의 `ROS_DOMAIN_ID`가 같아야 한다(원래 PC는 100).
```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=100
[ -f ~/amr_ws/install/setup.bash ] && source ~/amr_ws/install/setup.bash
```

## 2. 레포 받기
GitHub에 SSH 키를 등록한다(`ssh-keygen -t ed25519`, 공개키를 GitHub Settings → SSH keys에 추가).
```bash
mkdir -p ~/amr_ws/src && cd ~/amr_ws/src
git clone git@github.com:Jaesik0722/amr-ros2.git
git config --global user.name Jaesik0722
git config --global user.email jslvjh@gmail.com
```

## 3. 의존성과 빌드
```bash
cd ~/amr_ws
rosdep install --from-paths src --ignore-src -y     # nav2, slam_toolbox, opennav_docking,
                                                    # pointcloud_to_laserscan, dynamixel_sdk, mcap ...
colcon build --symlink-install
source install/setup.bash
colcon test && colcon test-result --all             # 36 tests, 0 failures
```
rosdep이 못 찾는 패키지가 있으면 apt로 설치한다.
```bash
sudo apt install -y ros-jazzy-navigation2 ros-jazzy-nav2-bringup ros-jazzy-slam-toolbox \
  ros-jazzy-opennav-docking ros-jazzy-pointcloud-to-laserscan ros-jazzy-dynamixel-sdk \
  ros-jazzy-xacro ros-jazzy-rosbag2-storage-mcap ros-jazzy-teleop-twist-joy python3-opencv
```

## 4. Isaac Sim 5.1 (시뮬레이션 PC)
```bash
# Miniconda (https://docs.anaconda.com/miniconda/)
conda create -n isaaclab python=3.11 -y
conda activate isaaclab
pip install --upgrade pip
pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
# (선택) Isaac Lab 2.3.2: 강화학습(논문 2)용. https://isaac-sim.github.io/IsaacLab
conda deactivate
```
- `run_sim.sh`는 기본으로 `~/anaconda3/envs/isaaclab/bin/python`을 쓴다. 경로가 다르면
  `export ISAAC_PYTHON=~/miniconda3/envs/isaaclab/bin/python`을 `.bashrc`에 넣는다.
- 처음 실행할 때 셰이더 캐시를 만드느라 5~10분 걸린다. 로봇 USD는 `~/.cache/amr_isaac`에 자동으로 만들어진다.
- 창고 환경은 NVIDIA 에셋 서버에서 받는다(인터넷 필요). 안 되면 빈 바닥으로 뜬다(`--env empty`).

## 5. 동작 확인 (순서대로)
```bash
cd ~/amr_ws/src/amr-ros2
./amr_isaac/run_sim.sh --headless --selftest           # 5/5 PASS, exit 0
```
터미널 두 개로 전체 시나리오를 돌린다.
```bash
./amr_isaac/run_sim.sh --chase-camera                  # 1: 시뮬레이터 (GUI)
ros2 launch amr_bringup sim.launch.py slam:=true nav:=true   # 2: SLAM + Nav2 + 도킹 + RViz
```
```bash
R="--ros-args -p use_sim_time:=true"
ros2 run amr_docking dock_charger $R      # 충전기 자동 도킹 (+5 A 충전)
ros2 run amr_docking undock_charger $R
ros2 run amr_docking dock_cart $R         # indoor GPS → 후진 도킹 → 히치 결합 확인
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: map}, pose: {position: {x: 6.5, y: -4.0}, orientation: {w: 1.0}}}}"
ros2 run amr_docking release_cart $R
```
데모 영상은 `amr_isaac/tools/demo_video/record.sh` 주석 순서대로 만든다.

## 6. 실제 로봇 PC
1. **Livox Mid-360**: Livox-SDK2와 livox_ros_driver2를 소스 빌드한다.
   ```bash
   git clone https://github.com/Livox-SDK/Livox-SDK2.git && cd Livox-SDK2 && mkdir build && cd build
   cmake .. && make -j && sudo make install
   cd ~/amr_ws/src && git clone https://github.com/Livox-SDK/livox_ros_driver2.git
   cd livox_ros_driver2 && ./build.sh jazzy      # 또는 colcon build (README 참고)
   ```
   - 라이다 IP와 PC IP를 `amr_bringup/config/MID360_config.json`에 맞춘다(지금 값은 placeholder).
   - 사무실 LAN이 192.168.1.x라서 Mid-360 기본 IP와 겹친다. 라이다를 다른 서브넷(예: 192.168.10.x)으로 바꾼다.
2. **시리얼 포트**: 사용자를 dialout 그룹에 넣는다(`sudo usermod -aG dialout $USER`, 재로그인).
   - 모터 드라이버(RS485)는 `/dev/ttyWheel`(19200 baud), Dynamixel은 U2D2 포트다.
   - USB 순서가 바뀌지 않게 udev 규칙으로 이름을 고정한다. 예: `/etc/udev/rules.d/99-amr.rules`
     ```
     SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{serial}=="<RS485 시리얼>", SYMLINK+="ttyWheel"
     SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{serial}=="<U2D2 시리얼>", SYMLINK+="ttyDXL"
     ```
     시리얼 번호는 `udevadm info -a /dev/ttyUSB0 | grep serial`로 확인한다.
3. **RealSense D435 ×2**: `sudo apt install ros-jazzy-realsense2-camera`. T265는 최신 librealsense가
   지원하지 않는다(필요하면 librealsense 2.53.1 이하를 별도로 빌드).
4. 실행: `ros2 launch amr_bringup bringup.launch.py`.
5. **실물에서 보정할 값**(CLAUDE.md "남은 일"):
   - `engage_effort`(A): 카트가 있을 때와 없을 때의 히치 전류를 재서 정한다.
   - indoor GPS 좌표계와 SLAM 지도 좌표계를 정합한다.
   - 충전기 위치(`nav2_params.yaml`의 `home_charger`).
   - 실측 질량, 히치 관절 한계, 서보 ID와 영점.

## 7. Gaussian Splatting 촬영·처리 PC
자세한 방법은 [gaussian_splatting.md](gaussian_splatting.md)에 있다. 설치만 정리하면 다음과 같다.

| 구성 | 설치 |
|---|---|
| ZED SDK 5.x + CUDA | https://www.stereolabs.com/developers/release (Ubuntu 24.04, CUDA 12.x) |
| zed-ros2-wrapper | `cd ~/amr_ws/src && git clone --recursive https://github.com/stereolabs/zed-ros2-wrapper.git`, rosdep, colcon build |
| livox_ros_driver2 | 위 6-1과 같다 |
| FAST-LIO (ROS 2) | `git clone --recursive -b ros2 https://github.com/Ericsii/FAST_LIO.git` (livox_ros_driver2를 먼저 빌드) |
| 3DGRUT | https://github.com/nv-tlabs/3dgrut, 별도 conda 환경. USDZ 내보내기로 Isaac Sim에 넣는다 |
| Open3D | `pip install open3d` (라이다 지도 → 충돌용 메시) |
| COLMAP (선택) | `sudo apt install colmap` (포즈 정제) |

## 8. 원래 PC에만 있는 파일
| 파일 | 위치 | 필요한 경우 |
|---|---|---|
| CAD 원본 `Product1_V0.2.stp`(123 MB) | `~/amr_cad/` | 메쉬·URDF를 다시 생성할 때만(CLAUDE.md 참고) |
| 데모 영상 3개 | `~/amr_cad/*.mp4` | 발표 자료 |

원래 PC(Tailscale 100.113.96.13)가 켜져 있으면 다음처럼 받는다.
```bash
scp -r minioni@100.113.96.13:~/amr_cad ~/
```
시뮬레이션과 실험에는 레포만 있으면 된다. 메쉬, URDF, 마커 이미지가 모두 레포에 들어 있다.

## 9. Claude Code
- 새 PC에서 `claude`를 레포 폴더에서 실행하면 `CLAUDE.md`를 읽고 맥락을 이어받는다.
- 이전 PC의 Claude 메모리는 옮겨지지 않는다. 필요한 규칙(답변은 한국어, 커밋 메시지에 Claude 문구 금지)은
  CLAUDE.md에 적혀 있다.
