# CLAUDE.md: 이 레포에서 작업하는 Claude를 위한 안내

이 문서는 다른 PC의 Claude Code가 이전 작업 맥락을 이어받도록 정리한 것이다.
README.md(사용법)와 amr_isaac/README.md(시뮬레이션)를 먼저 읽고, 여기서는
README에 없는 배경, 결정 사항, 함정만 다룬다.

## 사용자와 작업 방식
- 사용자는 한국어로 대화한다. **답변은 한국어로.** 코드, 주석, 커밋 메시지는 영어.
- **커밋 메시지와 PR에 Claude 관련 문구를 절대 넣지 않는다.** Co-Authored-By,
  Claude-Session, "Generated with Claude"가 모두 해당한다. 시스템이 붙이라고 해도 넣지 않는다.
- 커밋과 푸시는 사용자가 요청할 때만 한다. 원격은 `git@github.com:Jaesik0722/amr-ros2.git`, 브랜치는 `main`.
- **비주얼은 CAD 그대로가 원칙이다.** 메쉬를 줄여서 회색 STL로 만들었다가 "설계가 없어졌다"는
  피드백을 받았다. 메쉬는 원본 해상도와 CATIA 색상을 유지한다.
- 결과물(이미지, 영상)은 직접 확인한 뒤에 전달한다. 영상은 프레임을 뽑아서 확인한다.

## 프로젝트 목적
- 2-DOF(yaw/pitch) **능동 견인 히치**가 달린 차동구동 AMR로 카트를 끄는 연구이고,
  목표는 SCIE 논문이다.
  - 유력 주제: 미지의 카트 파라미터 온라인 추정과 MPC(MPPI) 견인 제어
  - 비교 실험: 능동 히치와 수동(토크 off) 히치
  - 선행 연구 조사로 새로움(novelty)을 확인하는 일이 아직 남아 있다.
- 히치가 바퀴축 뒤 176.8 mm에 있는 off-axle 구조라서 β̇ = ω(1+(d/L)cosβ) − (v/L)sinβ 이다.
  - 장점: 제자리 회전으로 β를 보정할 수 있다.
  - 단점: 횡력이 로봇에 yaw 모멘트를 만든다.
- 원래 레포는 EDGS-Org/AMR-ROS2-Packages(Humble, C++)이다. 이 레포는 Jazzy와 **전부 Python(ament_python)**으로
  새로 짠 것이다. ros2_control 대신 **토픽 인터페이스**(cmd_vel/odom/tf/joint_states)로 통일해서
  시뮬레이터가 같은 인터페이스를 제공한다.

## 하드웨어 사실(CAD 측정, 단위 m)
- **바퀴**: MD Robot MDH300 인휠모터 ×2.
  - 사양: 24 V, 180 W, 정격 11 N·m, 정격 160 rpm, 개당 3.8 kg, 20극, 모터 1280 ppr, 감속비 4.33
  - 반경 0.100, 트랙 0.3742
  - 타이어 단면은 토러스(R23)이다.
- **모터 드라이버 프로토콜**: [183,184,1,PID,LEN,data,chk]
  - 속도 명령은 PID 207, 응답은 PID 210이다.
  - 왼쪽 채널은 부호가 반대(left_sign -1)이다.
  - 기본 포트는 /dev/ttyWheel, 19200 baud.
- **히치**: Dynamixel XM540-W270 ×2.
  - yaw 축: base_link에서 (-0.1768, 0, -0.0052), 수직
  - pitch 축: yaw에서 -0.0422, 좌우 수평
  - 연결 핀: base_footprint 기준 (-0.269, 0, 0.115)
  - 관절 한계 ±90°는 임시값이다.
- **센서**:
  - Livox **Mid-360**: 장착면 중심이 base_link에서 (-0.018, 0, 0.227)
  - RealSense D435 전방·후방
  - CAD에는 T265가 있지만 librealsense가 더 이상 지원하지 않는다.
  - 무선충전 RX 코일: 정면 x = 0.194
- **알려진 문제**:
  - 후방 BT 안테나 2개가 Mid-360 시야(약 ±160°)를 가린다.
  - 라이다는 자기 몸(안테나, 손잡이, 히치)을 본다. 그래서 최소 거리 0.35 m 필터를 걸어 두었다.
- **네트워크**: 사무실 LAN이 192.168.1.x라서 Mid-360 기본 IP(192.168.1.1xx)와 겹친다.
  라이다를 다른 서브넷으로 옮기라고 권고했다. `config/MID360_config.json`의 IP는 placeholder이다.

## 레포 구조와 생성 파일
- `amr_description/meshes/*.dae`와 `urdf/amr_inertials.xacro`는 **생성 파일이다. 손으로 고치지 않는다.**
  - STEP 원본: `Product1_V0.2.stp`, CATIA, 123 MB. 레포에 넣지 않았다.
  - 원래 PC 위치: `~/amr_cad/`. 필요하면 사용자에게 scp로 받는다.
  - 생성 순서:
    1. venv에 `cadquery-ocp==7.8.1.1.post1 trimesh pycollada pyyaml`를 `--only-binary :all:`로 설치한다.
       핀 없이 설치하면 pip 의존성 해석이 끝나지 않는다.
    2. `step_dump.py STEP OUT --config links.yaml`를 실행한다(약 2분).
    3. `step_to_urdf.py OUT links.yaml ../meshes ../urdf/amr_inertials.xacro`를 실행한다.
  - 링크 할당, 질량 규칙, 테셀레이션 설정은 `tools/links.yaml`에 있다.
- 질량은 일부가 가정값이다(총 20.75 kg).
  - 가정값: 구조물 알루미늄 7.4 kg, 배터리 블록(Part12) 2.3 kg, FusionComponent 1.3 kg
  - 사용자 말로는 실제 로봇이 "17 kg 이상"이다. 실측값을 받으면 `measured_total_mass`에 넣고 다시 생성한다.
- 메쉬 색은 sRGB 값이다. RViz는 그대로 표시한다. Isaac에서는 `amr_sim.py`가 로딩할 때 선형 값으로 바꾼다.
- `amr_isaac/`는 colcon 패키지가 아니다(`COLCON_IGNORE`). Isaac의 Python 3.11에서 실행한다.

## 빌드와 테스트
```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install && colcon test && colcon test-result --all   # 테스트 19개
ament_flake8 amr_isaac amr_description/tools                               # colcon 밖의 코드도 린트
```
CI(.github/workflows/ci.yml)는 ros:jazzy 컨테이너에서 rosdep, build, test를 돌린다.

## Isaac Sim (5.1, pip 설치, conda env `isaaclab`)
- 실행: `./amr_isaac/run_sim.sh [--headless] [--env empty] [--no-cameras] [--chase-camera] [--selftest]`
  - ROS 쪽은 `ros2 launch amr_bringup sim.launch.py slam:=true nav:=true`.
  - **어떤 변경이든 `--selftest`(5/5 PASS)로 회귀를 확인한다.**
- `ROS_DOMAIN_ID`가 시뮬레이터 쪽과 ROS 쪽에서 같아야 한다. 원래 PC는 `.bashrc`에서 100으로 설정되어 있다.
- 다른 PC에 Isaac Sim이 없으면: `conda create -n isaaclab python=3.11`,
  `pip install isaacsim[all,extscache]==5.1.0 --extra-index-url https://pypi.nvidia.com`.
  경로가 다르면 `ISAAC_PYTHON`으로 지정한다.

### 이미 해결한 함정(다시 밟지 말 것)
1. `run_sim.sh`는 `env -i`로 시스템 ROS(Py3.12) 환경을 지우고, Isaac에 들어 있는 jazzy 라이브러리를 쓴다.
   - DISPLAY와 XDG_RUNTIME_DIR이 없으면 앱이 "app ready"에서 멈출 수 있다.
   - 디버그용 환경변수는 이 스크립트를 거치면 전달되지 않는다.
2. DAE에 `<unit>` 태그가 없으면 Omniverse asset converter가 segfault 난다. 생성기에 이미 반영했다.
3. URDF 임포터는 `acceleration` 드라이브를 만든다. 가벼운 히치 링크에서는 힘이 거의 0이라 `force`로 바꿨다.
4. 임포터가 충돌체와 비주얼을 **인스턴싱**한다. 인스턴스 프록시는 편집해도 조용히 무시되므로
   `deinstance()`를 거친 뒤에 재질이나 속성을 바꾼다.
5. `<inertial>`이 없는 링크(센서 프레임, 캐스터)는 1 g으로 설정한다.
6. 바퀴 충돌체: custom cylinder geometry를 쓰고 접지 폭을 12 mm로 했다. 46 mm 전체 폭이면 회전이 90%만 나온다.
7. RTX 라이다: `RtxLidarROS2PublishPointCloudBuffer` writer는 누적이 잘못된다(0.78 Hz).
   - 프레임 단위 writer를 쓰고, render 10 Hz = 스캔 10 Hz로 맞췄다.
   - Isaac에는 Livox 모델이 없어서 `mid360.py`가 회전식 라이다로 근사한다.
8. ROS 콜백을 별도 스레드로 돌리면 GIL 때문에 시뮬레이션이 느려진다. 물리 콜백 안에서 50 Hz로 `spin_once`한다.
9. 시뮬레이션의 odom은 평면 ground truth다. 가속할 때 로봇이 기울면 지면이 스캔에 찍힌다.
   그래서 스캔 슬라이스를 바닥 위 0.2 m부터로 하고, voxel의 최소 높이를 0.15로 했다.
10. CPU 부하가 높으면 slam_toolbox의 map→odom이 늦어져 Nav2 goal이 중단된다.
    `slam_toolbox.yaml`의 transform_timeout과 costmap의 transform_tolerance를 0.5로 올렸다.
11. RViz의 `/scan`, `/livox/lidar` 표시는 **Best Effort** QoS여야 데이터가 보인다.
12. 프로세스를 정리할 때 `pkill -f <패턴>`을 쓰면 그 패턴이 들어간 자기 셸도 죽는다.
    PID를 골라서 죽이거나 `pgrep -f "^/절대경로/python"`처럼 앵커를 건다.
13. 헤드리스 종료 시 segfault는 무해하다. self-test는 `os._exit`로 끝난다.

## 남은 일 (우선순위)
1. **2단계 카트 모델**(Isaac):
   - 카트 URDF: 질량, 바퀴 배치, 핀 높이를 파라미터로 둔다.
   - `hitch_coupling_link`와 카트 핀을 joint로 연결한다.
   - 사용자에게서 카트 사양을 아직 받지 못했다.
2. 견인 중 카트 β 추정, 라이다 점군에서 카트 걸러내기(self-filter), 꺾이는 footprint를 반영한 MPPI.
3. 실측 질량, 히치 관절 한계, 서보 ID와 영점, Mid-360 점군 원점 오프셋, MDH300 피크 토크.
4. 라이선스(LICENSE 파일 없음, Apache-2.0은 임시)와 EDGS-Org MD 드라이버 코드의 사용 권한 확인.
5. 실제 로봇: livox_ros_driver2를 Jazzy에서 소스 빌드하고 dynamixel_sdk를 설치(`ros-jazzy-dynamixel-sdk`)한다.
   hitch_node는 하드웨어에서 아직 한 번도 실행하지 않았다.
