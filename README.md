# dual_fr3_lab

DYROS dual-FR3 (Husky 베이스 + 양팔 Franka FR3) 로봇의 RL 태스크 모음.
[Isaac Lab](https://github.com/isaac-sim/IsaacLab) **external project** 방식 — Isaac Lab 본체는 import로만 사용하고,
로봇 cfg·태스크·에셋·실험 기록을 이 repo에서 독립 관리한다.

> **버전 고정: Isaac Lab 2.2.1 / Isaac Sim 5.0.0** (conda env `env_isaaclab`, Python 3.11, Ubuntu 24.04)
> Isaac Lab 3.0(beta)/Isaac Sim 6.0은 Python 3.12 + 물리 백엔드 변경이 있으므로 마이그레이션 시 이 기준 코드임을 참고.

## Multi-PC 셋업 (2026-07-28 검증)

| | PC1 (주력) | PC2 |
|---|---|---|
| Isaac Lab / Isaac Sim | 2.2.1 / 5.0.0 (conda pip 설치) | 2.3.2 / 5.1.0 (바이너리 설치, `~/Isaac/IsaacLab`) |
| 실행 방법 | `conda activate env_isaaclab` 후 맨 `python` | **반드시 `~/Isaac/IsaacLab/isaaclab.sh -p` 래퍼** (맨 python은 X/omni 환경 문제로 크래시) |
| **rsl-rl-lib** | **3.0.1 (양쪽 필수 고정)** | 5.0.1이 기본 설치되어 있었음 → `isaaclab.sh -p -m pip install rsl-rl-lib==3.0.1`로 다운그레이드 |

버전 차이 흡수 장치 (scripts에 내장, 양쪽 무해):
- `dump_pickle` fallback — Isaac Lab 2.3에서 `isaaclab.utils.io`로부터 제거됨
- PPO cfg 키 자동 필터 — isaaclab_rl 2.3이 만드는 신형 필드(`optimizer`, `share_cnn_encoders`)를
  rsl_rl 3.0.1 PPO 시그니처 기준으로 제거 (`[compat]` 로그로 확인 가능)

주의: 물리 엔진이 다르므로(5.0 vs 5.1) **엄밀한 run 비교는 같은 PC끼리**. `docs/log_per_task.txt`에
run 기록 시 PC 표기(`[2.2.1/5.0]` / `[2.3.2/5.1]`)를 붙일 것.

## Tasks

| Gym ID | 내용 |
|---|---|
| `Isaac-Lift-Cube-DualFR3-v0` | 오른팔로 테이블(상판 0.75m) 위 DexCube 파지·목표 이동. grasp gating(양 finger pad 접촉) + 충돌 페널티 + 측정 joint 토크 obs + obs noise/물성 DR |
| `Isaac-Lift-Cube-DualFR3-Play-v0` | 위 태스크 play용 (obs corruption off) |
| `Isaac-Lift-Cube-DualFR3-Play-Noisy-v0` | 강건성 평가용 (noise 활성) — `scripts/tools/eval_dual_fr3_lift_robustness.py`와 세트 |
| `Isaac-Lift-Box-DualFR3-Plate-v0` | 양팔 판형 EE(140×140mm)로 바닥 박스 squeeze 파지·들기 |
| `Isaac-Lift-Box-DualFR3-Plate-Play-v0` | 위 태스크 play용 |

설계 이력·판단 근거는 `docs/DUAL_FR3_LIFT_ENV_ANALYSIS.md`(cube), `docs/DUAL_FR3_LIFT_PLATE_ENV_ANALYSIS.md`(plate),
run별 실험 기록은 `docs/log_per_task.txt` 참조.

## Installation

```bash
# 전제: Isaac Lab 2.2.1이 설치된 conda env (env_isaaclab)
conda activate env_isaaclab
python -m pip install -e source/dual_fr3_lab
python scripts/list_envs.py   # DualFR3 태스크들이 보이면 성공
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
source/dual_fr3_lab/dual_fr3_lab/
├── assets/          # 완성품 USD (git-lfs): dual_fr3_no_handeye, dual_fr3_plate_ee
├── robots/          # ArticulationCfg (USD 경로는 패키지 상대)
└── tasks/manager_based/
    ├── lift_cube/   # cube lift (mdp/: grasp gating, 측정 토크 obs, hold-still 등 커스텀)
    └── lift_plate/  # plate box lift (lift_cube의 mdp·EMA curriculum을 import)
scripts/
├── rsl_rl | rl_games | skrl | sb3   # 학습/재생 (Isaac Lab 템플릿 제공)
└── tools/           # 강건성 평가, 배치 preview, rule-based 데모 (아래 표)
content/             # URDF 원본 + USD 생성 파이프라인 문서(make_usd.md) + plate EE CAD 스크립트
docs/                # 환경 정밀 분석 2편 + run별 실험 로그(log_per_task.txt)
```

## scripts/tools 목록

RL 학습과 별개로 로봇·환경을 검증/평가하는 도구들. 전부 `dual_fr3_lab.robots`에서 로봇 cfg를
가져오므로 Isaac Lab 본체 수정 없이 동작한다. 실행: `python scripts/tools/<이름>.py [옵션]`
(PC2는 `isaaclab.sh -p` 래퍼로).

| 스크립트 | 용도 | 주요 옵션 |
|---|---|---|
| `preview_dual_fr3_table.py` | 테이블/스폰존/goal존 배치 미리보기 + 수치 마진 체크 (학습 전 배치 검증) | `--mjcf` (0.75 테이블), `--table_x` |
| `run_dual_fr3_diff_ik.py` | differential IK 추종 데모 — TCP 오프셋/워크스페이스 검증 (HIGH_PD cfg 사용) | — |
| `run_dual_fr3_pick.py` | 단일팔(오른팔) rule-based pick — cube 파지 파이프라인 검증 | — |
| `run_dual_fr3_pick_handover.py` | 양팔 handover 데모 (직교 grip) — 양팔 협조·TCP 정합 검증 | — |
| `run_dual_fr3_plate_lift.py` | 판 EE squeeze 파지 rule-based 데모 — plate task의 기준 궤적 | — |
| `eval_dual_fr3_lift_robustness.py` | **cube 정책 강건성 정량 평가**: hold/precise-hold/drop 측정. Play vs Play-Noisy 비교로 sim2real 취약점 진단 | `--checkpoint` (필수), `--task`, `--no_obs_noise`, `--no_physics_dr` (축 분리) |
| `eval_dual_fr3_plate_robustness.py` | plate 정책 강건성 평가 (위의 plate 버전) | 동일 패턴 |

주의: rule-based 데모들은 GUI 확인용 (headless 무의미). eval 계열은 `--headless` 권장이며,
종료 시 Isaac Sim `close()`가 행에 걸릴 수 있으니 배치 실행 시 `timeout -s KILL`로 감쌀 것.

## Notes

- **USD 재생성**: 빌드 소스(xacro/ROS ws, 메시)는 이 repo 밖 `~/fr3_ws`에 있다. 절차는 `content/make_usd.md`.
  재생성 시 링크 구조(merge-joints, link7에 hand 병합)와 robots/ cfg의 TCP 오프셋 정합을 반드시 확인할 것.
- **git-lfs**: `*.usd`, `*.stl`, `*.step`은 LFS로 추적한다 (`git lfs track` 설정 후 커밋).
- 실물 FR3에는 F/T·촉각 센서가 없다는 전제로 설계됨 — 접촉 정보는 reward 전용(sim-only),
  actor obs는 실기 획득 가능 신호(관절 상태, 내장 토크 τ_ext, 물체 pose)만 사용.
