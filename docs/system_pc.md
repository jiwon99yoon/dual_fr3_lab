# Dual FR3 배포 환경 — PC 구성과 workspace 설계

> 갱신: **2026-08-12**
>
> 역할: sim2sim·sim2real 실행을 위한 PC 환경, 역할 분리, 외부 저장소, workspace 구조, 진행 로드맵을 관리한다.
> 태스크별 I/O 계약과 sim2real gap 분석은 [`sim2real_gap.md`](sim2real_gap.md) 및 각 태스크 문서를 기준으로 한다.

---

## 0. 문서 라우팅

| 궁금한 내용 | 기준 문서 |
|---|---|
| 공통 sim2real 원칙, 외란·timing 처리 | [`sim2real_gap.md`](sim2real_gap.md) |
| Cube-v3 inference I/O 계약, 배포 체크리스트 | [`DUAL_FR3_LIFT_CUBE_SIM2REAL.md`](DUAL_FR3_LIFT_CUBE_SIM2REAL.md) |
| Plate box-lift 배포 | [`DUAL_FR3_LIFT_PLATE_SIM2REAL.md`](DUAL_FR3_LIFT_PLATE_SIM2REAL.md) |
| 실행·평가·실측 결과 | [`log_per_task.txt`](log_per_task.txt) |

---

## 1. PC 구성

### 1-1. 개발 PC (학습·inference 개발)

| 항목 | 값 |
|---|---|
| OS | Ubuntu 24.04 LTS |
| ROS2 | Jazzy |
| 역할 | Isaac Lab 학습, policy export, ONNX inference 개발, ArUco 노드, shadow mode logging |
| 주요 저장소 | `~/dual_fr3_lab` (Isaac Lab sim) |
| 계획 workspace | `~/dual_fr3_real_ws` (inference + perception ROS2 패키지) |

### 1-2. 로봇 PC (하드웨어 제어)

| 항목 | 값 |
|---|---|
| OS | Ubuntu 22.04 LTS |
| ROS2 | Humble |
| 역할 | fr3_husky controller 빌드·실행, FR3 hardware interface, MuJoCo sim2sim |
| 주요 저장소 | `~/ros2_ws/src/fr3_husky` (예정) |
| 연결 | 개발 PC와 DDS 네트워크 통신 |

---

## 2. 역할 분리 원칙

```
개발 PC (Jazzy)                         로봇 PC (Humble)
───────────────────────────             ────────────────────────────
dual_fr3_lab  (Isaac Lab 학습)          fr3_husky (controller + MJCF)
ArUco node    (perception)    ←──DDS──  joint_states
policy inference node          ──DDS──→ joint command topic
shadow mode logger                      FR3 hardware interface (실물)
ONNX model, obs/action 계약              MuJoCo sim (sim2sim)
```

**분리 근거:**
- fr3_husky는 Humble 전용 의존성(libfranka, franka_ros2)을 가지므로 로봇 PC에서만 빌드한다.
- policy inference는 순수 Python + ONNX이므로 Jazzy에서 문제 없이 실행된다.
- ROS2 DDS는 Jazzy ↔ Humble 네트워크 통신을 지원한다 (DDS vendor 일치 필요, 아래 §4 참조).

**개발 PC에서 빌드·실행하는 것:**
- `dual_fr3_policy` (ONNX inference, 36D obs 조립, 20 Hz timer)
- `dual_fr3_perception` (ArUco, RealSense)
- `dual_fr3_bringup` (launch 모음, config YAML)
- shadow mode / evaluation logger

**로봇 PC에서 빌드·실행하는 것:**
- `fr3_husky_controller` (1000 Hz ros2_control plugin)
- `dual_fr3_policy_controller` (fr3_husky generator로 생성한 position-mode 컨트롤러)
- MuJoCo launch (sim2sim)
- FR3 hardware launch (실물)

---

## 3. 외부 저장소

### 3-1. fr3_husky (제어 프레임워크)

| 항목 | 값 |
|---|---|
| URL | `https://github.com/JunHeonYoon/fr3_husky` |
| 브랜치 | `devel` |
| 확인한 commit | `5e255a93` |
| 로봇 PC 경로 (예정) | `~/ros2_ws/src/fr3_husky` |
| ROS2 | Humble |

**구조 요약:**
- `fr3_husky_description/mjcf/` — `dual_fr3_husky.xml.xacro` 포함 (MuJoCo용)
- `fr3_husky_controller/` — ros2_control plugin, controller generator (`generate_fr3_controller.py`)
- `fr3_husky_moveit_config/` — MoveIt2 설정
- launch 플래그: `use_mujoco:=true`, `use_fake_hardware:=true`, `load_gripper:=true`
- 기본 command interface: **effort (토크)** → position-mode 컨트롤러는 generator로 생성

**주의:**
- `use_mujoco:=true` 시 gripper launch가 자동으로 비활성화된다.
  sim2sim에서는 gripper를 별도로 처리해야 한다.
- 팔 장착 geometry:
  - left:  `[0, +0.215, 0]`, yaw `+0.5236 rad` (≈ +30°)
  - right: `[0, -0.215, 0]`, yaw `-0.5236 rad` (≈ -30°)
  - 우리 Isaac sim URDF와 일치함 (검증 완료).
- joint 이름: `left_fr3_joint1` ~ `left_fr3_joint7`, `right_fr3_joint1` ~ `right_fr3_joint7`
- controller manager 주기: **1000 Hz**
- action server: `GravityCompensation`, `MoveToJoint`, `MobileMove` 등

### 3-2. aruco_realsense (perception)

| 항목 | 값 |
|---|---|
| URL | `https://github.com/iamjoseph1/aruco_realsense` |
| 브랜치 | `ros2` |
| 확인한 commit | `dc38102f` |
| 개발 PC 경로 (예정) | `~/dual_fr3_real_ws/src/aruco_realsense` 또는 독립 |

**구조 요약:**
- 핵심 스크립트: `6_publish_object_multiple_qr.py`
- 출력: `/object_pose` (PoseStamped), TF `base → object_frame`
- 카메라: pyrealsense2 직접 열기 (RealSense ROS driver와 동시 사용 불가)
- 외부 viewpoint 카메라 기준 (손목 카메라 아님)
- 기본 필터: position N개 평균 → **N=1로 시작 권장** (지연 최소화)

**실사용 전 수정 필요 사항:**
- `--camera-serial` 옵션 추가 (카메라 2대 이상 시 필수)
- publish timestamp를 node now()가 아닌 capture time 기준으로 변경
- marker 미검출 시 stale 표시 메시지 발행 (현재는 발행 중단)
- ament_python 패키지 구조 (package.xml, setup.py, launch/) 추가

---

## 4. DDS 네트워크 연결

ROS2는 DDS 기반이므로 **Jazzy ↔ Humble 간 topic/service 통신이 가능하다.**
단, 양 PC의 DDS vendor가 일치해야 한다.

```bash
# 각 PC에서 확인
echo $RMW_IMPLEMENTATION
# 또는
ros2 doctor --report | grep rmw
```

- 기본값이 다를 경우 양쪽 모두 동일 vendor로 설정한다.
  예: `export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`
- 같은 서브넷에 있으면 별도 설정 없이 DDS auto-discovery가 동작한다.
- 보안이 필요하면 `ROS_DOMAIN_ID`를 양쪽 동일하게 설정한다.

---

## 5. 계획된 ROS2 workspace 구조

### 개발 PC: `~/dual_fr3_real_ws`

```
~/dual_fr3_real_ws/
├── dual_fr3_real.repos          # vcs manifest (commit hash 고정)
└── src/
    ├── aruco_realsense/         # 팀 fork 또는 직접 clone
    ├── dual_fr3_msgs/           # custom message (PolicyAction.msg 등)
    ├── dual_fr3_policy/         # ONNX inference, 36D obs 조립, 20 Hz
    ├── dual_fr3_perception/     # ArUco ROS 패키지화, calibration YAML
    └── dual_fr3_bringup/        # launch, config YAML 모음
```

**`dual_fr3_msgs/PolicyAction.msg` 예시:**
```
builtin_interfaces/Time stamp
uint64 sequence
float64[7] arm_action    # raw policy output (clamp 전)
bool gripper_close
bool valid
string stop_reason
```

**`dual_fr3_policy` 역할:**
- `/joint_states`, `/object_pose` 구독
- joint name 재정렬 → ready pose 상대값 → 36D obs 조립
- ONNX inference @ 20 Hz
- previous raw action 관리
- ArUco stale 검사
- `/dual_fr3_lift/raw_action` 발행

### 로봇 PC: `~/ros2_ws/src` (기존 workspace에 추가)

```
~/ros2_ws/src/
├── fr3_husky/                   # devel, commit 고정
└── dual_fr3_policy_controller/  # fr3_husky generator로 생성한 컨트롤러
```

**`dual_fr3_policy_controller` 역할:**
- 개발 PC의 `/dual_fr3_lift/raw_action` 구독
- 오른팔: 20→100 Hz rebase (`q_target = q_current + 0.1 * action`)
- 왼팔: ready pose hold
- 1000 Hz tracking + joint/rate/workspace limit
- command watchdog (100~150 ms timeout)

---

## 6. sim2sim → sim2real 로드맵

### Phase 1: Offline (로봇 PC 불필요)

- [ ] Isaac Lab에서 obs/action 저장 (`obs_000.npy`, `action_000.npy`)
- [ ] ONNX parity test: `max |action_onnx - action_sim| < 1e-5`
- [ ] 36D observation builder 단위 테스트
  - joint name 재정렬
  - absolute q → ready pose 상대값 변환
  - goal quaternion 순서 `(w, x, y, z)` vs `(x, y, z, w)` 확인
  - previous action 초기화 (0 벡터)
  - ArUco stale 시 inference 중단 로직
- [ ] PolicyAction message 정의
- [ ] `dual_fr3_policy` node skeleton (subscribe → inference → publish)

### Phase 2: sim2sim (MuJoCo)

- [ ] 로봇 PC에 fr3_husky 빌드 (devel 브랜치)
- [ ] `use_mujoco:=true`로 dual arm MuJoCo 실행 확인
- [ ] joint_states topic 이름·Hz·joint 순서 확인
- [ ] `dual_fr3_policy_controller` 생성 (position mode)
  - `generate_fr3_controller.py DualFr3LiftPolicyController --control_mode position`
  - `update()` 내 20→100 Hz rebase 구현
  - 왼팔 ready hold, 오른팔 policy target
- [ ] 개발 PC ↔ 로봇 PC DDS 연결 확인
- [ ] shadow mode: inference 결과 로그만, 명령 미전송
- [ ] 작은 수동 action으로 controller 동작 확인 (`a = [0.01, 0, ...]`)
- [ ] policy + MuJoCo 연동 (gripper 없이 팔 7축만)
- [ ] gripper 처리 방침 결정 (sim2sim에서는 open 고정 또는 fake node)

**MuJoCo에서 확인할 것:**
- 100 Hz rebase가 Isaac sim의 substep 동작과 동일하게 동작하는가
- 팔이 합리적인 궤적을 그리는가
- command watchdog이 정상 동작하는가

### Phase 3: sim2real 전환

- [ ] ArUco standalone 측정 (publish rate, jitter, dropout, latency)
- [ ] base-to-camera calibration → `TF_base2cam.yaml`
- [ ] marker layout YAML 작성 (실물 부착 위치 확인)
- [ ] frame 검증: `policy_base` frame에서 nominal cube 위치 비교
  - 기대값: `(x, y, z) ≈ (0.45, -0.25, 0.395)` in policy robot-root
- [ ] arm command latency/tracking 측정 (object 없이)
- [ ] gripper timing·width·success flag 측정
- [ ] safety supervisor 구현 (§6-1 참조)
- [ ] real ladder 순차 실행

### Phase 4: 후속 학습 (필요 시)

zero-shot 실패 근거가 있을 때만, [`DUAL_FR3_LIFT_CUBE_SIM2REAL.md §5`](DUAL_FR3_LIFT_CUBE_SIM2REAL.md) 참조.

---

## 6-1. Safety Supervisor 항목 (실물 필수)

policy action timeout: 100~150 ms
ArUco stale timeout:   100~150 ms  
episode timeout:       8 s
joint position hard limit
joint velocity/acceleration/jerk limit
workspace floor/table guard
start-pose deviation limit
tracking-error sustained timeout
command watchdog (controller가 명령을 못 받으면 hold)
physical E-stop 준비

---

## 7. 기타 확인 사항

### Frame 주의

Isaac sim에서 robot-root frame은 URDF의 `base` link다.  
fr3_husky URDF에서 `base_link`가 같은 위치인지 **TF chain으로 반드시 확인**해야 한다.  
ArUco publisher의 `--base-frame` 인수를 바로 policy frame으로 설정하지 말고,  
ArUco는 실제 calibration base frame으로 발행 → policy bridge에서 TF 변환하는 것이 안전하다.

### Action semantics 재현

sim의 action semantics:
```
policy  @ 20 Hz: delta = 0.1 * raw_action
physics @ 100 Hz: q_target[k] = q_current[k] + delta  (매 10 ms 갱신)
```

실물 controller는 이 100 Hz rebase를 재현해야 한다.  
**잘못된 구현**: policy tick 시 snapshot target 한 번 계산 → 50 ms hold  
이 경우 action semantics가 달라지며 재학습이 필요하다.  
자세한 내용은 [`DUAL_FR3_LIFT_CUBE_SIM2REAL.md §1-3`](DUAL_FR3_LIFT_CUBE_SIM2REAL.md) 참조.

### Previous action

policy observation의 `previous action`은 raw policy output이다 (clamp 전).  
실제 실행된 clamped/executed action과 반드시 분리해서 관리한다.
