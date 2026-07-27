# dual_fr3_lab

DYROS dual-FR3 (Husky 베이스 + 양팔 Franka FR3) 로봇의 RL 태스크 모음.
[Isaac Lab](https://github.com/isaac-sim/IsaacLab) **external project** 방식 — Isaac Lab 본체는 import로만 사용하고,
로봇 cfg·태스크·에셋·실험 기록을 이 repo에서 독립 관리한다.

> **버전 고정: Isaac Lab 2.2.1 / Isaac Sim 5.0.0** (conda env `env_isaaclab`, Python 3.11, Ubuntu 24.04)
> Isaac Lab 3.0(beta)/Isaac Sim 6.0은 Python 3.12 + 물리 백엔드 변경이 있으므로 마이그레이션 시 이 기준 코드임을 참고.

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
└── tools/           # 강건성 평가, 배치 preview, rule-based 데모
content/             # URDF 원본 + USD 생성 파이프라인 문서(make_usd.md) + plate EE CAD 스크립트
docs/                # 환경 정밀 분석 2편 + run별 실험 로그(log_per_task.txt)
```

## Notes

- **USD 재생성**: 빌드 소스(xacro/ROS ws, 메시)는 이 repo 밖 `~/fr3_ws`에 있다. 절차는 `content/make_usd.md`.
  재생성 시 링크 구조(merge-joints, link7에 hand 병합)와 robots/ cfg의 TCP 오프셋 정합을 반드시 확인할 것.
- **git-lfs**: `*.usd`, `*.stl`, `*.step`은 LFS로 추적한다 (`git lfs track` 설정 후 커밋).
- 실물 FR3에는 F/T·촉각 센서가 없다는 전제로 설계됨 — 접촉 정보는 reward 전용(sim-only),
  actor obs는 실기 획득 가능 신호(관절 상태, 내장 토크 τ_ext, 물체 pose)만 사용.
