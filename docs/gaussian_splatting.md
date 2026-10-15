# 실제 현장 → Gaussian Splatting → Isaac Sim 트윈

실제 공장을 카트에 실은 센서 리그로 촬영해, Isaac Sim 안에 **보이는 것은 Gaussian Splatting(GS),
부딪히고 라이다에 잡히는 것은 라이다 지도 메시**인 디지털 트윈을 만드는 방법이다.
논문 2(GS 트윈에서 학습한 마커 없는 결합)의 기반이고, 논문 1·3의 현장 시뮬레이션에도 쓴다.

```
촬영(카트 리그)                처리                                   시뮬레이션
ZED 2  ──┐                    FAST-LIO ─→ 궤적 + 라이다 지도(pcd)
Mid-360 ─┼→ rosbag(mcap) ─→   export_dataset ─→ COLMAP / nerfstudio ─→ 3DGRUT ─→ splats.usdz ─┐
T265  ───┘   + SVO            pcd_to_mesh    ─→ collision.usda ────────────────────────────────┤
                                                                                              ↓
                                       run_sim.sh --env-usd splats.usdz --env-collision collision.usda
```

## 1. 리그 구성(카트 탑재)
- **ZED 2, Mid-360, T265를 한 장의 플레이트에 고정하고, 높이는 플레이트째 조절한다.**
  센서 사이 외부 보정이 모든 높이에서 그대로 유지된다. 카메라만 움직이면 높이마다 다시 보정해야 한다.
- 카메라는 진행 방향 정면을 본다. Mid-360은 플레이트 앞쪽 높은 곳에 둬서 손잡이나 미는 사람에 덜 가리게 한다.
- 미는 사람은 Mid-360에 찍힌다. `crop_operator`가 리그 뒤쪽 상자 영역(`config/capture.yaml`의
  `crop_operator`)을 지운다. 상자 크기는 실제 손잡이 위치에 맞춘다.
- 바닥 진동이 롤링 셔터 왜곡을 키운다. 우레탄·고무 바퀴 카트를 쓰고, 플레이트 아래에 방진 고무를 넣는다.
  높이 조절 마스트가 흔들리지 않게 한다.
- 기록 PC(노트북 또는 Jetson)와 배터리를 카트에 싣는다. 모든 센서를 한 PC에서 기록해야 시간이 맞는다.

## 2. 보정(촬영 당일, 촬영 전)
1. **ZED 내부 파라미터**: 공장 보정값을 쓰고, 정류된(rectified) 영상을 기록한다. export는 `camera_info`의
   K를 PINHOLE 모델로 쓴다.
2. **카메라-라이다 외부 보정**(`T_rig_camera`): 리그 좌표 = Mid-360 IMU 프레임(FAST-LIO의 body)이다.
   - 권장 도구: [direct_visual_lidar_calibration](https://github.com/koide3/direct_visual_lidar_calibration)
     (타깃 없이 정적 장면에서 보정, Livox 지원).
   - 결과를 `capture.yaml`의 `T_rig_camera`(x y z roll pitch yaw, 카메라 광학 프레임: x 오른쪽,
     y 아래, z 앞)에 넣는다. 기본값은 "라이다 위 0.10 m, 정면"인 임시값이다.
3. **시간 오프셋**(`time_offset`): 한 PC에서 기록하면 보통 수 ms 이내다. 리그를 빠르게 좌우로 흔든
   구간에서 ZED IMU와 Livox IMU의 각속도를 교차상관하면 오프셋을 잴 수 있다.

## 3. 촬영 방법
| 패스 | 높이 | 목적 |
|---|---|---|
| **A (필수)** | **약 0.2 m** | 로봇 D435 높이(전방 0.20 m, 후방 0.22 m). 결합 정책이 보는 시점이다 |
| B | 0.6~0.8 m | 시점 보간, 바닥과 선반 하단 |
| C | 1.2~1.5 m | 위에서 보는 시점, 전체 구조 |

- **모든 패스를 bag 하나에 연속으로 기록한다.** 높이는 멈춘 상태에서 천천히 바꾼다. FAST-LIO가 한 궤적,
  한 좌표계로 모든 패스를 처리하므로 패스 사이 정합이 필요 없다.
- 속도 **0.2~0.3 m/s 이하**를 유지한다. 통로 끝에서는 멈추고 천천히 돌며, 제자리 회전은 피한다.
- 경로는 고리 모양으로 만들어 루프를 닫는다. 통로는 왕복한다(카메라가 정면 하나라서).
- 충전 스테이션과 카트 대기 위치(결합 실험 장소)는 패스 A로 여러 각도에서 천천히, 촘촘하게 찍는다.
- 조명은 모두 켜고 사람, 지게차, 열리는 문이 없는 시간에 찍는다.
- **노출, 게인, 화이트밸런스는 고정**한다(`config/zed2_capture.yaml`). 셔터는 조명이 허락하는 한 짧게 한다.
- 정합 검증용 기준점: 바닥 몇 곳에 ArUco 마커를 붙이거나 indoor GPS 비컨 위치를 기록해 둔다.
  나중에 GS 좌표계, SLAM 지도, GPS 좌표를 맞추고 시뮬레이션-실물 오차를 잴 때 쓴다.
- **처음에는 10 m 정도만 시험 촬영해서 아래 처리 과정을 끝까지 한 번 돌려 본다.**

```bash
ros2 launch amr_capture capture.launch.py bag:=~/gs_capture/site_a      # t265:=true 선택
# Ctrl+C로 끝낸다. ~/gs_capture/site_a/{bag/, zed2.svo2}
```
1080p JPEG(15 Hz) 기록은 대략 5~10 MB/s다. SVO는 전체 30 fps 원본 백업이다.

## 4. 처리
### 4-1. FAST-LIO로 궤적과 지도 만들기(오프라인)
```bash
# 터미널 1: FAST-LIO (config: amr_capture/config/fast_lio_mid360.yaml)
ros2 launch fast_lio mapping.launch.py config_path:=$(ros2 pkg prefix amr_capture)/share/amr_capture/config \
  config_file:=fast_lio_mid360.yaml rviz:=false
# 터미널 2: 사람 제거
ros2 run amr_capture crop_operator
# 터미널 3: 궤적 기록
ros2 bag record -s mcap -o ~/gs_capture/site_a/odom /Odometry
# 터미널 4: 재생 (crop_operator가 Python이라 0.5배속 권장)
ros2 bag play ~/gs_capture/site_a/bag -r 0.5
```
끝나면 FAST-LIO가 `PCD/scans.pcd`(지도)를 저장한다(fast_lio 패키지의 PCD 폴더 또는 실행 폴더).
launch 인자 이름은 사용하는 FAST_LIO ROS 2 포트 버전의 `launch/mapping.launch.py`에서 확인한다.

### 4-2. 데이터셋 내보내기
```bash
ros2 run amr_capture export_dataset ~/gs_capture/site_a/bag ~/gs_capture/site_a/dataset \
  --poses ~/gs_capture/site_a/odom --pcd <FAST_LIO>/PCD/scans.pcd
```
- 5 cm 또는 3°마다 프레임을 고르고, 흐린 프레임(최근 30장 선명도 중앙값의 절반 미만)은 버린다.
- 지도에서 바닥 평면을 찾아 **바닥이 z = 0이 되도록 좌표계를 맞춘다**(`alignment.txt`).
  시뮬레이터의 바닥이 z = 0이라서 GS와 메시가 그대로 맞는다.
- 출력: `images/`, `sparse/0/`(COLMAP 텍스트: 3DGRUT, gsplat, Inria 3DGS용), `transforms.json`(nerfstudio용).
- 포즈가 부족하면 COLMAP `point_triangulator`나 `bundle_adjuster`로 정제할 수 있다(선택).

### 4-3. Gaussian Splatting 학습(3DGRUT)
```bash
# 3DGRUT 환경에서 (https://github.com/nv-tlabs/3dgrut)
python train.py --config-name apps/colmap_3dgut.yaml path=~/gs_capture/site_a/dataset \
  out_dir=~/gs_capture/site_a/gs experiment_name=site_a export_usdz.enabled=true
```
- 위 명령은 예시다. 설정 파일과 옵션 이름은 3DGRUT README에서 확인한다(버전마다 다르다).
- 3DGUT는 왜곡 카메라와 **롤링 셔터 카메라 모델**을 지원하는 것으로 알려져 있다. ZED 2는 글로벌 셔터가
  아니므로, 지원 옵션과 정확한 설정 이름은 사용하는 3DGRUT 버전의 문서를 확인한다.
- USDZ 내보내기(`export_usdz`)로 Isaac Sim 5.x(NuRec)에서 바로 읽을 수 있는 파일이 나온다.

### 4-4. 충돌·라이다용 메시
```bash
pip install open3d
ros2 run amr_capture pcd_to_mesh <FAST_LIO>/PCD/scans.pcd ~/gs_capture/site_a/collision.usda \
  --alignment ~/gs_capture/site_a/dataset/alignment.txt          # --voxel 0.03 --depth 10
```
Poisson 재구성 결과를 스캔 범위로 잘라 USDA로 쓴다. 6만 삼각형 기준 몇 초 걸린다.

## 5. Isaac Sim에 넣기
```bash
./amr_isaac/run_sim.sh --env-usd ~/gs_capture/site_a/gs/site_a.usdz \
  --env-collision ~/gs_capture/site_a/collision.usda --no-cart --no-charger
```
- `--env-usd`: 화면용(GS USDZ). 여러 개를 줄 수 있다.
- `--env-collision`: 메시에 정적 충돌체를 붙이고 `primvars:hideForCamera`로 **카메라에서는 숨긴다**.
  RTX 라이다는 이 메시를 그대로 본다(테스트로 확인했다). 보이지 않게(invisible) 만들면 라이다도 못 본다.
  메시를 확인하려면 `--env-collision-visible`을 준다.
- 물리용 바닥(z = 0)은 숨겨진 GroundPlane이다.
- 카트와 충전기는 `--cart-pose`, `--charger-pose`로 현장 좌표에 배치한다(지도 좌표 = 데이터셋 좌표).
- **확인할 점**: GS(NuRec)가 로봇 D435의 깊이 영상에 어떻게 나오는지, 조명 없이도 색이 맞는지는
  첫 데이터로 확인해야 한다. 깊이 영상이 필요하면 메시 기반 깊이로 대체할 수 있다.

## 6. 품질 체크리스트
- [ ] export 로그의 blurred 비율이 20% 이하인가(넘으면 촬영 속도를 줄이거나 셔터를 짧게)
- [ ] FAST-LIO 지도에 사람 잔상이 없는가(있으면 crop 상자를 키운다)
- [ ] 패스 A 높이(0.2 m)에서 렌더링한 영상이 실제 D435 영상과 비슷한가(논문 2의 핵심 지표)
- [ ] 기준점(바닥 마커) 위치가 GS 좌표와 실측에서 몇 cm 이내로 맞는가
