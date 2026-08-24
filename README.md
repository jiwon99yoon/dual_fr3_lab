# dual_fr3_lab

DYROS dual-FR3 (Husky 베이스 + 양팔 Franka FR3) 로봇의 RL 태스크 모음.

## Demo

| Cube Lift v3 (오른팔 단독, 배포 후보) | Plate Box Lift P47 (양팔 판형 EE, 현재 최선) |
|:---:|:---:|
| ![cube lift v3](demos/dual_fr3_lift_v3.gif) | ![plate lift P47](demos/dual_fr3_plate_p47.gif) |
[Isaac Lab](https://github.com/isaac-sim/IsaacLab) **external project** 방식 — Isaac Lab 본체는 import로만 사용하고,
로봇 cfg·태스크·에셋·실험 기록을 이 repo에서 독립 관리한다.

> **버전 고정: Isaac Lab 2.2.1 / Isaac Sim 5.0.0** (conda env `env_isaaclab`, Python 3.11, Ubuntu 24.04)
> Isaac Lab 3.0(beta)/Isaac Sim 6.0은 Python 3.12 + 물리 백엔드 변경이 있으므로 마이그레이션 시 이 기준 코드임을 참고.

## Multi-PC 셋업 (2026-07-28 검증)

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

## Tasks

Cube lift는 sim2real 목표로 v0→v3까지 진화했다. **각 버전은 한 축씩만 바꿔 ablation으로 검증** (v0 보존, 로그 분리).

| Gym ID | 내용 |
|---|---|
| `Isaac-Lift-Cube-DualFR3-v0` | 원형. grasp gating(양 finger pad 접촉) + 충돌 페널티 + 측정 joint 토크(τ_ext) obs + obs noise/물성 DR. 테이블 상판 world 0.75m(base 위 0.345m) |
| `Isaac-Lift-Cube-DualFR3-v1` | v0 − τ_ext obs (ablation: 기여 없음·오히려 잡음 → 제거 확정). **obs 43→36** |
| `Isaac-Lift-Cube-DualFR3-v2` | v1 + decimation 5(정책 50→**20Hz**) + episode 8s. 실물 perception 주기 정합 |
| `Isaac-Lift-Cube-DualFR3-v3` | v2 + cube **50×50×100mm 세로 직육면체** (실물 정합). **← 현재 실물 배포 후보** |
| `...-Play-v{N}` / `...-Play-Noisy-v{N}` | 각 버전 play용(corruption off) / 강건성·성공률 평가용(noise on) — `eval_dual_fr3_lift_robustness.py`와 세트 |
| `Isaac-Lift-Box-DualFR3-Plate-v0/v1/v2` | (plate 세션) 양팔 판형 EE(140×140mm)로 바닥 박스 squeeze 파지·들기. v2=외란+마찰DR, 기준선 |
| `Isaac-Lift-Box-DualFR3-Plate-P39-v0` ~ `P51-v0` | P-series 단일변수 reward 실험. **현재 최선: P47** (SR 0.984, tie 83.6%, one-sided 0.85 steps) |
| `...-P{N}-Play-v0` / `...-P{N}-Play-Noisy-v0` | 각 P-series play/강건성 평가용. `eval_dual_fr3_plate_robustness.py`와 세트 |

각 버전의 **배포 스펙**(obs 순서·ready pose·action 규약·배포 루프·TODO)은 `docs/dual_fr3_lift_v{1,2,3}.md`.
**현재 배포 후보 = v3** (`docs/dual_fr3_lift_v3.md`). 설계 이력·판단 근거는
`docs/DUAL_FR3_LIFT_ENV_ANALYSIS.md`(cube), run별 실험 기록은 `docs/log_per_task.txt`.

> **⚠ 크기 DR 교훈 (2026-08-03)**: `randomize_rigid_body_scale`은 `mode="prestartup"`으로 써야 한다.
> `mode="usd"`는 IsaacLab 2.2.1에서 아무도 호출하지 않아 **에러 없이 침묵** — v0~v2에서 cube 크기 DR이
> 내내 무효(실제 4.8cm 고정)였다. instanceable USD보다 `MeshCuboidCfg` primitive가 크기 검증에 유리.
> **새 크기/scale DR은 반드시 sim 실측(env별 AABB)으로 per-env 랜덤 여부를 확인할 것.**

## Installation

> **기준 환경: 5090-v2** (Ubuntu 24.04, Python 3.11, Isaac Sim 5.0.0 / Isaac Lab 2.2.1, conda pip 방식)
> 5090-v3(Isaac Sim 5.1.0 바이너리)는 아래 Multi-PC 셋업 참조.

### 1. conda 환경 생성

```bash
conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab
```

### 2. Isaac Sim 5.0.0 + Isaac Lab 2.2.1 pip 설치

```bash
# Isaac Sim 5.0.0 (NVIDIA PyPI)
pip install isaacsim==5.0.0.0 --extra-index-url https://pypi.nvidia.com

# Isaac Lab (설치 당시 최신 버전 = 2.2.1, pip 패키지명 기준 0.46.2)
pip install isaaclab --extra-index-url https://pypi.nvidia.com
```

> 설치 후 `pip show isaacsim isaaclab`으로 버전 확인: `isaacsim 5.0.0.0`, `isaaclab 0.46.2`

### 3. rsl-rl-lib 버전 고정 (필수)

Isaac Lab 기본 설치 시 rsl-rl-lib 최신 버전이 딸려 올 수 있다. **반드시 3.0.1로 맞춰야 한다.**

```bash
pip install rsl-rl-lib==3.0.1
```

### 4. dual_fr3_lab 설치

```bash
cd ~/dual_fr3_lab   # 이 repo 루트
pip install -e source/dual_fr3_lab

# 정상 설치 확인
python scripts/list_envs.py   # Isaac-Lift-Cube-DualFR3-v3 등이 보이면 성공
```

## Usage

```bash
conda activate env_isaaclab
cd <이 repo 루트>

# 학습 (rsl_rl PPO, 4096 envs 기본)
python scripts/rsl_rl/train.py --task Isaac-Lift-Cube-DualFR3-v0 --headless --max_iterations 5000

# 재생 (checkpoint 절대경로 지정)
python scripts/rsl_rl/play.py --task Isaac-Lift-Cube-DualFR3-Play-v0 --num_envs 16 \
    --checkpoint logs/rsl_rl/dual_fr3_lift/<run>/model_XXXX.pt

# 강건성 평가 (baseline vs noisy, 축 분리 스위치 --no_obs_noise / --no_physics_dr)
python scripts/tools/eval_dual_fr3_lift_robustness.py \
    --task Isaac-Lift-Cube-DualFR3-Play-Noisy-v0 --checkpoint <ckpt> --headless

# 배치 미리보기 (테이블/스폰존/goal존 시각화 + 수치 마진 체크)
python scripts/tools/preview_dual_fr3_table.py --mjcf
```

로그는 실행 위치 기준 `logs/`에 쌓인다 (git 미추적).

## Repository structure

```
dual_fr3_lab/
├── source/dual_fr3_lab/dual_fr3_lab/   # pip 설치 패키지 (pip install -e source/dual_fr3_lab)
│   ├── assets/          # 완성품 로봇 USD (git-lfs): dual_fr3_no_handeye, dual_fr3_plate_ee
│   ├── robots/          # ArticulationCfg (USD 경로는 패키지 상대 경로)
│   └── tasks/manager_based/
│       ├── lift_cube/   # cube lift (mdp/: grasp gating, hold-still 등 커스텀 reward)
│       └── lift_plate/  # plate box lift (lift_cube의 mdp·EMA curriculum을 import)
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
├── docs/                # 실험 분석·설계 문서 및 run 기록
│   ├── DUAL_FR3_LIFT_ENV_ANALYSIS.md        # cube lift 환경 설계 이력 (v0~v3, 판단 근거)
│   ├── DUAL_FR3_LIFT_CUBE_SIM2REAL.md       # cube v3 실물 배포 계약·calibration·체크리스트
│   ├── DUAL_FR3_LIFT_PLATE_ENV_ANALYSIS.md  # plate lift 환경 설계 이력 (v2~P-series)
│   ├── DUAL_FR3_LIFT_PLATE_SIM2REAL.md      # plate 실물 배포 계약
│   ├── dual_fr3_lift_v1.md                  # cube v1 스펙 (50Hz, 히스토리)
│   ├── dual_fr3_lift_v2.md                  # cube v2 스펙 (20Hz 정육면체, 히스토리)
│   ├── dual_fr3_lift_v3.md                  # cube v3 스펙 (20Hz 세로 cube — 현재 배포 후보)
│   ├── sim2real_gap.md                      # 공통 sim2real 원칙
│   ├── system_pc.md                         # PC 환경·역할 분리·ROS2 workspace·로드맵
│   └── log_per_task.txt                     # 전체 run 기록 (학습 결과·판정·설계 메모)
│
└── logs/                # 학습 로그 (git 미추적, rsl_rl/<task>/<timestamp>/ 구조)
```

## scripts/tools 목록

RL 학습과 별개로 로봇·환경을 검증/평가하는 도구들. 전부 `dual_fr3_lab.robots`에서 로봇 cfg를
가져오므로 Isaac Lab 본체 수정 없이 동작한다. 실행: `python scripts/tools/<이름>.py [옵션]`
(5090-v3는 `isaaclab.sh -p` 래퍼로).

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

주의: rule-based 데모들은 GUI 확인용 (headless 무의미). eval 계열은 `--headless` 권장이며,
종료 시 Isaac Sim `close()`가 행에 걸릴 수 있으니 배치 실행 시 `timeout -s KILL`로 감쌀 것.

## Notes

- **USD 재생성**: 빌드 소스(xacro/ROS ws, 메시)는 이 repo 밖 `~/fr3_ws`에 있다. 절차는 `content/make_usd.md`.
  재생성 시 링크 구조(merge-joints, link7에 hand 병합)와 robots/ cfg의 TCP 오프셋 정합을 반드시 확인할 것.
- **git-lfs**: `*.usd`, `*.stl`, `*.step`은 LFS로 추적한다 (`git lfs track` 설정 후 커밋).
- 실물 FR3에는 F/T·촉각 센서가 없다는 전제로 설계됨 — 접촉 정보는 reward 전용(sim-only),
  actor obs는 실기 획득 가능 신호(관절 상태, 내장 토크 τ_ext, 물체 pose)만 사용.
