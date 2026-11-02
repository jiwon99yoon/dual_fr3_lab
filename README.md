# dual_fr3_lab

DYROS dual-FR3 (Husky 베이스 + 양팔 Franka FR3) 로봇의 RL 태스크 모음.
[Isaac Lab](https://github.com/isaac-sim/IsaacLab) **external project** 방식 — Isaac Lab 본체는 import로만 사용하고,
로봇 cfg·태스크·에셋·실험 기록을 이 repo에서 독립 관리한다.

> **최신 버전 브랜치: Isaac Lab 3.0.0 Beta 2 / Isaac Sim 6.0.1** (conda env `isaac_601_300b2`, Python 3.12, Ubuntu 24.04)

---

## Demo

| Cube Lift v3 (오른팔 단독, 배포 후보) | Plate Box Lift P47 (양팔 판형 EE, 현재 최선) | Reach (양팔 EE, sim2real gap 분석용) |
|:---:|:---:|:---:|
| ![cube lift v3](demos/dual_fr3_lift_v3.gif) | ![plate lift P47](demos/dual_fr3_plate_p47.gif) | ![reach](demos/dual_fr3_reach.gif) |

---

## Installation

> **기준 환경** (Ubuntu 24.04, Python 3.12, Isaac Sim 6.0.1 / Isaac Lab 3.0.0 Beta 2, conda + pip 설치)
> PyPI 패키지는 호환 패치가 포함된 `isaacsim 6.0.1.0` / `isaaclab 3.0.0b2.post1`로 고정한다.
> NVIDIA 드라이버는 **580 이상**이어야 한다(PyTorch cu130). 검증 PC: RTX 5070 Ti, RTX 5070 12 GB(드라이버 595).

### 0. clone (브랜치 `3.0.0/6.0` + git-lfs)

GitHub 기본 브랜치 `main`은 Isaac Lab 2.2.1 시절 코드다. Isaac Lab 3 작업은 **`3.0.0/6.0` 브랜치**에 있다.
로봇 USD(`source/dual_fr3_lab/dual_fr3_lab/assets/*.usd`)는 git-lfs라서 `git lfs pull`을 안 하면 포인터 파일만 받는다.

```bash
git lfs install                     # 계정당 1회. git-lfs가 없으면 sudo apt install git-lfs
git clone -b 3.0.0/6.0 https://github.com/jiwon99yoon/dual_fr3_lab.git
cd dual_fr3_lab
git lfs pull
ls -lh source/dual_fr3_lab/dual_fr3_lab/assets/*.usd   # 수 MB여야 한다 (100 B대면 LFS 포인터)

# 이미 main으로 clone한 경우
git fetch origin && git checkout 3.0.0/6.0 && git lfs pull
```

### 1. conda 환경 생성

```bash
conda create -n isaac_601_300b2 python=3.12 -y      # mamba도 같은 명령으로 가능
conda activate isaac_601_300b2
```

### 2. PyTorch 2.11.0 (CUDA 13.0) 설치

```bash
python -m pip install --upgrade pip

python -m pip install \
  torch==2.11.0 \
  torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu130
```

### 3. Isaac Sim 6.0.1 + Isaac Lab 3.0.0 Beta 2 설치

```bash
python -m pip install \
  "isaaclab[isaacsim,all]==3.0.0b2.post1" \
  --extra-index-url https://pypi.nvidia.com
```

> `isaacsim[all,extscache] 6.0.1.0`과 Isaac Lab의 RL 의존성이 함께 설치된다. `rsl-rl-lib`는 isaaclab이 **5.0.1**로
> 고정해서 설치한다(다른 버전이면 학습 cfg가 맞지 않는다). 디스크는 env 약 27 GB + Omniverse 캐시
> (`~/.local/share/ov`, `~/.cache/ov`) 약 10 GB가 든다(RTX 5070 PC 실측).

### 4. dual_fr3_lab 설치

```bash
cd <이 repo 루트>
python -m pip install -e source/dual_fr3_lab

# 설치 버전 및 의존성 확인
python -m pip show isaacsim isaaclab torch torchvision rsl-rl-lib | grep -E "^(Name|Version)"
python -m pip check

# 정상 설치 확인 (Isaac Sim EULA 동의는 환경 변수로)
export OMNI_KIT_ACCEPT_EULA=YES
python scripts/list_envs.py

# 짧은 학습 시험. 첫 실행은 셰이더·확장 캐시로 몇 분 걸리고, 바닥 에셋을 NVIDIA 서버에서 받으므로 인터넷이 필요하다
python scripts/rsl_rl/train.py --task Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-v0 \
    --viz none --num_envs 64 --max_iterations 2 --run_name smoke
```

---

## Usage

Isaac Lab 3에서 `--headless`는 deprecated다. 창 없이 돌릴 때는 `--viz none`, 창을 띄울 때는 `--viz kit`.

```bash
conda activate isaac_601_300b2
cd <이 repo 루트>
export OMNI_KIT_ACCEPT_EULA=YES

# 학습 (rsl_rl PPO, 4096 envs 기본)
python scripts/rsl_rl/train.py --task Isaac-Lift-Cube-DualFR3-v0 --viz none --max_iterations 5000

# 장시간 학습은 터미널과 분리해서 (로그는 repo 루트에)
setsid nohup python scripts/rsl_rl/train.py --task Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-v0 \
    --viz none --max_iterations 3000 --run_name pregrasp_box410 > box_pregrasp_box410.log 2>&1 &

# 이어 학습 (logs/rsl_rl/<experiment>/<run>/model_XXXX.pt)
python scripts/rsl_rl/train.py --task <task> --viz none --resume --load_run <run> --checkpoint model_XXXX.pt

# 재생 (checkpoint 절대경로 지정)
python scripts/rsl_rl/play.py --task Isaac-Lift-Cube-DualFR3-Play-v0 --num_envs 16 \
    --checkpoint logs/rsl_rl/dual_fr3_lift/<run>/model_XXXX.pt --viz kit

# 영상 녹화 (cube/box lift, 창 없이 1 env, overview/front/side 3시점 + 들림·기울기·판 높이차 HUD)
# -> <run>/videos/model_XXXX_views_ep3.mp4 + 에피소드 요약 .json
python scripts/rsl_rl/record_views.py --task Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Offset03-Play-v0 \
    --checkpoint logs/rsl_rl/dual_fr3_plate_lift_sim2real_100hz_offset03/<run>/model_XXXX.pt --episodes 3

# Reach: 명시적으로 지정한 goal sequence 재생
python scripts/rsl_rl/play.py --task Isaac-Reach-DualFR3-Play-v0 --num_envs 1 \
    --checkpoint logs/rsl_rl/dual_fr3_reach/<run>/model_XXXX.pt \
    --reach_goal_sequence config/reach_goal_sequence.example.json --viz kit

# Reach Sim2Real v1 학습 (tanh-bounded actor + FR3 torque/rate limits)
python scripts/rsl_rl/train.py --task Isaac-Reach-DualFR3-Sim2Real-v1 \
    --viz none --max_iterations 5000

# Reach Sim2Real 학습 (권장 배포 조합: 1 kHz + MuJoCo-matched friction)
python scripts/rsl_rl/train.py --task Isaac-Reach-DualFR3-Sim2Real-1kHz-Friction-v0 \
    --viz none --max_iterations 5000

# Lift Sim2Real 학습 (Reach Sim2Real v2와 같은 제어 계약, 플래그 불필요)
python scripts/rsl_rl/train.py --task Isaac-Lift-Cube-DualFR3-Sim2Real-1kHz-v0 --viz none
python scripts/rsl_rl/train.py --task Isaac-Lift-Box-DualFR3-Plate-Sim2Real-1kHz-v0 --viz none

# 강건성 평가 (baseline vs noisy, 축 분리 스위치 --no_obs_noise / --no_physics_dr)
python scripts/tools/eval_dual_fr3_lift_robustness.py \
    --task Isaac-Lift-Cube-DualFR3-Play-Noisy-v0 --checkpoint <ckpt> --viz none

# 배치 미리보기 (테이블/스폰존/goal존 시각화 + 수치 마진 체크)
python scripts/tools/preview_dual_fr3_table.py --mjcf
```

로그는 실행 위치 기준 `logs/`에 쌓인다 (git 미추적).

---

## Repository structure

```
dual_fr3_lab/
├── source/dual_fr3_lab/dual_fr3_lab/   # pip 설치 패키지 (pip install -e source/dual_fr3_lab)
│   ├── assets/          # 완성품 로봇 USD (git-lfs): dual_fr3_no_handeye, dual_fr3_plate_ee
│   │   └── pregrasp/    # box Pregrasp task의 시작 자세 표 (.npz, rule-based 접근으로 생성)
│   ├── robots/          # ArticulationCfg (USD 경로는 패키지 상대 경로)
│   └── tasks/manager_based/
│       ├── lift_cube/   # cube lift (mdp/: grasp gating, hold-still 등 커스텀 reward)
│       ├── lift_plate/  # plate box lift (lift_cube의 mdp·EMA curriculum을 import)
│       ├── handover_cube/ # 양팔 큐브 핸드오버
│       └── reach/       # sim2real gap 분석용 양팔 EE reach (obs 58D = plate와 동일 구조)
│
├── scripts/
│   ├── rsl_rl/ | rl_games/ | skrl/ | sb3/   # 학습/재생 (Isaac Lab 템플릿 기반)
│   └── tools/           # 강건성 평가, 배치 preview, rule-based 데모 (아래 표 참조)
│
├── content/             # 에셋 원본 및 생성 도구 (git-lfs: *.usd, *.stl, *.step)
│   ├── env/             # 환경 오브젝트 에셋: box_50x50x100.usd (v3 cube), .step
│   ├── plate_ee/        # plate EE 에셋: plate_ee_140x140.usd/stl/step + CAD 생성 스크립트
│   ├── urdf/            # 로봇 URDF 원본 (dual_fr3.urdf, dual_fr3_abs.urdf)
│   └── make_usd.md      # URDF → USD 변환 파이프라인 문서
│
├── demos/               # README용 데모 GIF (git-lfs)
│
├── docs/                # 실험 분석·설계 문서 및 run 기록
│   ├── DUAL_FR3_LIFT_ENV_ANALYSIS.md        # cube lift 환경 설계 이력 (v0~v3, 판단 근거)
│   ├── DUAL_FR3_LIFT_CUBE_SIM2REAL.md       # cube v3 실물 배포 계약·calibration·체크리스트
│   ├── DUAL_FR3_LIFT_PLATE_ENV_ANALYSIS.md  # plate lift 환경 설계 이력 (v2~P-series)
│   ├── DUAL_FR3_LIFT_PLATE_SIM2REAL.md      # plate 실물 배포 계약
│   ├── dual_fr3_lift_v1.md                  # cube v1 스펙 (50Hz, 히스토리)
│   ├── dual_fr3_lift_v2.md                  # cube v2 스펙 (20Hz 정육면체, 히스토리)
│   ├── dual_fr3_lift_v3.md                  # cube v3 스펙 (20Hz 세로 cube — 현재 배포 후보)
│   ├── dual_fr3_reach_v1.md                 # Reach Sim2Real v1 액션·액추에이터·배포 계약
│   ├── sim2real_gap.md                      # 공통 sim2real 원칙
│   ├── system_pc.md                         # PC 환경·역할 분리·ROS2 workspace·로드맵
│   └── log_per_task.txt                     # 전체 run 기록 (학습 결과·판정·설계 메모)
│
└── logs/                # 학습 로그 (git 미추적, rsl_rl/<task>/<timestamp>/ 구조)
```

---

## Tasks

Cube lift는 sim2real 목표로 v0→v3까지 진화했다. **각 버전은 한 축씩만 바꿔 ablation으로 검증** (v0 보존, 로그 분리).

| Gym ID | 내용 |
|---|---|
| `Isaac-Lift-Cube-DualFR3-v0` | 원형. grasp gating(양 finger pad 접촉) + 충돌 페널티 + 측정 joint 토크(τ_ext) obs + obs noise/물성 DR. 테이블 상판 world 0.75m(base 위 0.345m) |
| `Isaac-Lift-Cube-DualFR3-v1` | v0 − τ_ext obs (ablation: 기여 없음·오히려 잡음 → 제거 확정). **obs 43→36** |
| `Isaac-Lift-Cube-DualFR3-v2` | v1 + decimation 5(정책 50→**20Hz**) + episode 8s. 실물 perception 주기 정합 |
| `Isaac-Lift-Cube-DualFR3-v3` | v2 + cube **50×50×100mm 세로 직육면체** (실물 정합). **← 현재 실물 배포 후보** |
| `Isaac-Lift-Cube-DualFR3-Friction-v3` | v3 + 팔 관절 passive dynamics 명시 (armature 0.1 / friction 0.2 / viscous 0.003, Reach `*-Friction`과 동일). implicit PD 유지. 로그 `dual_fr3_lift_v3_friction` |
| `Isaac-Lift-Cube-DualFR3-Sim2Real-1kHz-v0` / `-100Hz-v0` | v3 + **Reach Sim2Real v2 제어 계약**: explicit PD + 토크 87/12 Nm·1000 Nm/s 제한 + passive dynamics, 1 kHz(또는 100 Hz) 물리, tanh actor + ±0.1 rad bounded action, policy-step target, PD 게인 DR. **← 실물 배포용 재학습 대상** ([docs/dual_fr3_lift_sim2real.md](docs/dual_fr3_lift_sim2real.md)) |
| `...-Play-v{N}` / `...-Play-Noisy-v{N}` | 각 버전 play용(corruption off) / 강건성·성공률 평가용(noise on) — `eval_dual_fr3_lift_robustness.py`와 세트 |
| `Isaac-Lift-Box-DualFR3-Plate-v0/v1/v2` | (plate 세션) 양팔 판형 EE(140×140mm)로 바닥 박스 squeeze 파지·들기. v2=외란+마찰DR, 기준선 |
| `Isaac-Lift-Box-DualFR3-Plate-P39-v0` ~ `P52-v0` | P-series 단일변수 reward 실험. **현재 최선: P47** (SR 0.984, tie 83.6%, one-sided 0.85 steps) |
| `Isaac-Lift-Box-DualFR3-Plate-P47-Friction-v0` | P47 + 팔 관절 passive dynamics 명시 (cube `Friction-v3`와 동일 값). reward/curriculum 불변. 로그 `dual_fr3_plate_lift_p47_friction` |
| `Isaac-Lift-Box-DualFR3-Plate-Sim2Real-1kHz-v0` / `-100Hz-v0` | P47 + cube Sim2Real과 같은 제어 계약. 정책 50→**20 Hz**(success hold 25→10 step), 관절 순서 left1~7/right1~7 고정 |
| `Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Offset03-v0` → `-Level` → `-PlateOnly` | box Sim2Real 100 Hz: action ±0.3 rad, 4 kg 박스 단계 실험 |
| `Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-v0` | PlateOnly + action ±0.1 rad(배포 노드 기준) + 0.5~1.0 kg 박스(555×475×275 mm) |
| `Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Margin-v0` | Light + 관절 한계를 FCI 한계보다 0.05 rad 안쪽으로 |
| `Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-v0` / `-Upright-v0` | **실물 2차 박스 410×310×280 mm, 0.4~0.6 kg.** 접근은 rule-based/MoveIt이 맡고, 판이 박스 면 4 cm 바깥에 있는 자세(`assets/pregrasp/` 표)에서 시작해 조이기~들기만 학습. Upright는 자세 페널티를 처음부터 켬. ([docs/dual_fr3_lift_sim2real.md](docs/dual_fr3_lift_sim2real.md)) |
| `...-P{N}-Play-v0` / `...-P{N}-Play-Noisy-v0` | 각 P-series play/강건성 평가용. `eval_dual_fr3_plate_robustness.py`와 세트 |
| `Isaac-Handover-Cube-DualFR3-v0` / `-v1` | 양팔 큐브 핸드오버 ([docs/dual_fr3_handover_cube.md](docs/dual_fr3_handover_cube.md)) |
| `Isaac-Reach-DualFR3-v0` | **sim2real gap 분석용 양팔 EE reach.** 물체 없이 goal center ± 0.2m(y축) 목표점으로 양팔 동시 도달. **obs 58D = plate task와 완전 동일 구조** (plate_forces·object_orientation 슬롯은 0으로 채움). action semantics(delta joint pos × 0.1, 20Hz)도 plate와 동일 — 두 정책의 관절 궤적 특성 비교로 제어 지연·tracking 오차 분리 가능 |
| `Isaac-Reach-DualFR3-Play-v0` | reach play용 (50 envs, corruption off) |
| `Isaac-Reach-DualFR3-Sim2Real-v1` | 100 Hz/no-passive-override 호환 ID. Reach v0의 unbounded Gaussian을 **tanh-squashed bounded actor**로 교체하고, explicit PD 뒤 FR3 **torque saturation(87/12 Nm)** 및 **torque-rate limit(1000 Nm/s)** 적용 |
| `Isaac-Reach-DualFR3-Sim2Real-v2` | 1 kHz/MuJoCo-matched-friction 호환 ID. v1의 obs/action/actor와 20 Hz policy 주기를 유지하고 physics + explicit PD를 **1 kHz**로 변경 |
| `Isaac-Reach-DualFR3-Sim2Real-100Hz-NoFriction-v0` | 100 Hz physics/PD, armature/friction을 명시적으로 override하지 않는 과거 v1/nofriction 재현 환경 |
| `Isaac-Reach-DualFR3-Sim2Real-100Hz-Friction-v0` | 100 Hz physics/PD + armature 0.1, static/dynamic friction 0.2, viscous friction 0.003 |
| `Isaac-Reach-DualFR3-Sim2Real-1kHz-NoFriction-v0` | 1 kHz physics/PD, armature/friction을 명시적으로 override하지 않는 `2026-09-10_19-33-09` 재현 환경 |
| `Isaac-Reach-DualFR3-Sim2Real-1kHz-Friction-v0` | 1 kHz physics/PD + 동일 passive dynamics. 현재 MuJoCo 배포 정합용 권장 환경 |
| `...-NoFriction-Play-v0` / `...-Friction-Play-v0` | 각 dynamics별 inference ID. checkpoint 학습에 사용한 rate/friction 조합과 같은 Play ID를 선택해야 함 |

### Reach 고정 goal sequence inference

play.py의 --reach_goal_sequence 옵션을 사용하면 seed 기반 random sampling 대신
JSON에 적힌 goal center를 순서대로 사용한다. 각 goal은 base frame 기준 위치와 유지
시간을 가진다.

~~~json
{
  "goals": [
    {"label": "first", "position": [0.50, 0.00, 0.15], "duration_s": 4.0},
    {"label": "second", "position": [0.45, 0.05, 0.25], "duration_s": 3.0}
  ]
}
~~~

~~~bash
python scripts/rsl_rl/play.py \
  --task Isaac-Reach-DualFR3-Play-v0 \
  --num_envs 1 \
  --checkpoint /absolute/path/to/model_4999.pt \
  --reach_goal_sequence config/reach_goal_sequence.example.json \
  --viz kit
~~~

sequence 모드에서는 random goal resampling과 8초 episode timeout이 비활성화된다.
마지막 goal의 duration_s가 끝나면 inference가 종료된다. 계속 반복하려면
--reach_goal_sequence_loop를 추가한다. 모든 position은 학습 범위
x=[0.40,0.60], y=[-0.10,0.10], z=[0.10,0.35] 안에 있어야 하며, 범위를
벗어나면 environment 생성 전에 오류를 낸다.

각 버전의 **배포 스펙**(obs 순서·ready pose·action 규약·배포 루프·TODO)은 `docs/dual_fr3_lift_v{1,2,3}.md`.
**현재 배포 후보 = v3** (`docs/dual_fr3_lift_v3.md`). 설계 이력·판단 근거는
`docs/DUAL_FR3_LIFT_ENV_ANALYSIS.md`(cube), run별 실험 기록은 `docs/log_per_task.txt`.

> **⚠ 크기 DR 교훈 (2026-08-03)**: `randomize_rigid_body_scale`은 `mode="prestartup"`으로 써야 한다.
> `mode="usd"`는 IsaacLab 2.2.1에서 아무도 호출하지 않아 **에러 없이 침묵** — v0~v2에서 cube 크기 DR이
> 내내 무효(실제 4.8cm 고정)였다. instanceable USD보다 `MeshCuboidCfg` primitive가 크기 검증에 유리.
> **새 크기/scale DR은 반드시 sim 실측(env별 AABB)으로 per-env 랜덤 여부를 확인할 것.**

## scripts/tools 목록

RL 학습과 별개로 로봇·환경을 검증/평가하는 도구들. 전부 `dual_fr3_lab.robots`에서 로봇 cfg를
가져오므로 Isaac Lab 본체 수정 없이 동작한다. 실행: `python scripts/tools/<이름>.py [옵션]`
(main 브랜치의 5090-v3는 `isaaclab.sh -p` 래퍼로).

| 스크립트 | 용도 | 주요 옵션 |
|---|---|---|
| `preview_dual_fr3_table.py` | 테이블/스폰존/goal존 배치 미리보기 + 수치 마진 체크 (학습 전 배치 검증) | `--mjcf` (0.75 테이블), `--table_x` |
| `run_dual_fr3_diff_ik.py` | differential IK 추종 데모 — TCP 오프셋/워크스페이스 검증 (HIGH_PD cfg 사용) | — |
| `run_dual_fr3_pick.py` | 단일팔(오른팔) rule-based pick — cube 파지 파이프라인 검증 | — |
| `run_dual_fr3_pick_handover.py` | 양팔 handover 데모 (직교 grip) — 양팔 협조·TCP 정합 검증 | — |
| `run_dual_fr3_plate_lift.py` | 판 EE squeeze 파지 rule-based 데모 — plate task의 기준 궤적 | — |
| `eval_dual_fr3_lift_robustness.py` | **cube 정책 강건성 + 에피소드 성공률 평가**: hold/precise-hold/drop + 성공률(5cm/2.5cm)/time-to-success. Play vs Play-Noisy 비교로 sim2real 취약점 진단, yaw 민감도 sweep | `--checkpoint`(필수), `--task`, `--no_obs_noise`, `--no_physics_dr`(축 분리), `--yaw_deg`, `--success_eps`, `--success_hold_steps` |
| `eval_dual_fr3_plate_robustness.py` | plate 정책 강건성 평가 (위의 plate 버전) | 동일 패턴 |
| `eval_gripper_toggle.py` | **그리퍼 이진 명령 toggle 빈도 측정** (sim2real 배포 진단) — 채터링 유무·히스테리시스 N 결정. env별 명령 시퀀스 덤프 | `--checkpoint`(필수), `--task`, `--num_envs`, `--episodes` |
| `dump_obs_layout.py` | **obs 벡터 구성·관절 순서 실측** — 실물 배포 시 obs 조립 배선 검증. 결과는 `/tmp/obslayout_result.txt`에 기록 | `--task` |
| `compute_dual_fr3_workspace.py` | 양팔 도달영역(workspace) 몬테카를로 FK 추정 — plate 파지 영역·goal 범위 설계용 | (plate 세션 도구) |
| `eval_dual_fr3_plate_rule_based.py` | box rule-based 조이기-들기 평가(배포 계약 그대로) + Pregrasp 시작 자세 표 생성 + 3시점 영상 | `--box_size`, `--mass`, `--box_dx`, `--disturb`, `--pregrasp_gap`/`--save_table`, `--video`, `--log`, `--gain_free` |
| `dump_boxlift_deploy_trace.py` | box 정책의 obs·action·원시 상태를 step별로 저장(실물 노드 obs 재구성 검증용) | `--task`(Play), `--checkpoint` |
| `export_boxlift_actor.py` | box actor를 배포용 NPZ(가중치 + `params/env.yaml` 기반 배포 manifest)로 내보내기 | run 폴더 / checkpoint |

주의: rule-based 데모들은 GUI 확인용 (창 없이 돌리면 의미 없음). eval 계열은 `--viz none` 권장이며,
종료 시 Isaac Sim `close()`가 행에 걸릴 수 있으니 배치 실행 시 `timeout -s KILL`로 감쌀 것.

---

## Multi-PC 셋업

### `3.0.0/6.0` 브랜치 (Isaac Lab 3.0.0 Beta 2 / Isaac Sim 6.0.1)

모든 PC에서 위 Installation 그대로 같은 pip 고정 버전(`isaacsim 6.0.1.0`, `isaaclab 3.0.0b2.post1`, `torch 2.11.0+cu130`,
`rsl-rl-lib 5.0.1`)과 같은 env 이름 `isaac_601_300b2`를 쓴다.
- 공용 PC에 다른 프로젝트(소스 설치 IsaacLab 등)가 있어도 env를 따로 만든다. 그쪽 업데이트가 이 repo 결과에 섞이지 않게 하기 위해서다.
- 옮기는 것은 git(코드, 로봇 USD, `assets/pregrasp/` 표)뿐이다. `logs/`(체크포인트)와 `data/`는 git에 없다. 이어 학습할 run만 `rsync`로 복사한다:
  `rsync -av logs/rsl_rl/<experiment>/<run> <user>@<host>:<repo>/logs/rsl_rl/<experiment>/`
- 4096 env 학습 1개는 대략 VRAM 3.7 GB, RAM 7 GB를 쓴다(RTX 5070 기준). 공용 PC에서는 `nvidia-smi`로 다른 사용자를 확인하고, GPU가 여러 장이면 `CUDA_VISIBLE_DEVICES`로 지정한다.
- 같은 버전이어도 GPU가 다르면 결과가 비트 단위로 같지 않다. 엄밀한 run 비교는 같은 PC끼리 하고, 기록에 PC를 적는다.

### `main` 브랜치 (Isaac Lab 2.2.1 / 2.3.2, 2026-07-28 검증, 참고용)

| | **5090-v2** (주력 학습 PC) | **5090-v3** |
|---|---|---|
| Isaac Lab / Isaac Sim | **2.2.1 / 5.0.0** (conda pip 설치, Python 3.11) | 2.3.2 / 5.1.0 (바이너리 설치, `~/Isaac/IsaacLab`) |
| 실행 방법 | `conda activate env_isaaclab` 후 맨 `python` | **반드시 `~/Isaac/IsaacLab/isaaclab.sh -p` 래퍼** (맨 python은 X/omni 환경 문제로 크래시) |
| **rsl-rl-lib** | **3.0.1 (양쪽 필수 고정)** | 5.0.1이 기본 설치됨 → `isaaclab.sh -p -m pip install rsl-rl-lib==3.0.1`로 다운그레이드 |

버전 차이 흡수 장치 (scripts에 내장, 양쪽 무해):
- `dump_pickle` fallback — Isaac Lab 2.3에서 `isaaclab.utils.io`로부터 제거됨
- PPO cfg 키 자동 필터 — isaaclab_rl 2.3이 만드는 신형 필드(`optimizer`, `share_cnn_encoders`)를
  rsl_rl 3.0.1 PPO 시그니처 기준으로 제거 (`[compat]` 로그로 확인 가능)

주의: 물리 엔진이 다르므로(5.0 vs 5.1) **엄밀한 run 비교는 같은 PC끼리**. `docs/log_per_task.txt`에
run 기록 시 PC 표기(`[2.2.1/5.0]` / `[2.3.2/5.1]`)를 붙일 것.

---

## Notes

- **USD 재생성**: 빌드 소스(xacro/ROS ws, 메시)는 이 repo 밖 `~/fr3_ws`에 있다. 절차는 `content/make_usd.md`.
  재생성 시 링크 구조(merge-joints, link7에 hand 병합)와 robots/ cfg의 TCP 오프셋 정합을 반드시 확인할 것.
- **git-lfs**: `*.usd`, `*.stl`, `*.step`, `*.gif`는 LFS로 추적한다 (`git lfs track` 설정 후 커밋).
- 실물 FR3에는 F/T·촉각 센서가 없다는 전제로 설계됨 — 접촉 정보는 reward 전용(sim-only),
  actor obs는 실로봇 획득 가능 신호(관절 상태, 내장 토크 τ_ext, 물체 pose)만 사용.
