# Dual-FR3 Sim-to-Real SysID 도구

## 이 도구가 하는 일

`run_dual_fr3_sysid.py`는 실로봇에서 수행한 관절별 probe 동작을 Isaac Lab에서
같이 재생하는 도구입니다. 실물과 Isaac의 관절 위치, 속도, 토크 응답을 나란히
비교하여 sim-to-real 차이를 확인하는 데 사용합니다.

대상 모델은 **D435i와 gripper가 없고, 양쪽 link7 끝에 plate EE가 달린 dual-FR3**입니다.
학습환경과 같은 USD 및 A/B/friction 물성을 사용하지만, SysID 실행 중에는 학습용
implicit position drive를 끄고 실물 제어 경로를 코드로 재현합니다.

```text
실물과 같은 관절 목표값을 100 Hz로 유지
  -> 매 1 ms마다 현재 관절각/속도로 torque 계산
  -> 관절별 안전 torque 한계 적용
  -> torque가 너무 급격히 변하지 않도록 제한
  -> PhysX에 최종 torque 입력
```

`Kp`, `Kd`, torque 한계, torque-rate 한계는 real controller와 맞춰 코드에 고정되어
있습니다. **일반적인 SysID 실행에서 사용자가 이 값을 바꿀 필요는 없습니다.**

학습 모델과 SysID 모델은 현재 아래 물성값을 공유합니다: armature A = `0.1`, dry
friction = `0.2`, viscous friction B = `0.003`. 이 값도 새 SysID 결과가 나오기 전에는
바꾸지 않습니다.

## 이 작업을 하는 이유와 현재 범위

시뮬레이터에서 학습한 정책이 실물에서 다르게 움직이는 이유는 로봇의 물성만이
아니라 controller, 토크 한계, command/update timing, 관측 지연도 simulation과 다를
수 있기 때문입니다. 이 도구는 그 차이를 추측하지 않고 같은 명령을 Isaac과 실물에
적용해 관절 응답으로 측정하기 위해 만들었습니다.

현재 runner가 할 수 있는 일은 **replay와 검증**입니다.

- real CSV의 정확한 q0에서 Isaac trial을 시작
- 실물과 같은 probe profile 및 controller path를 Isaac에서 재생
- Isaac의 q, qd, torque를 CSV로 기록
- real/Isaac overlay를 통해 tracking error, phase lag, torque 차이를 비교
- A/B/friction 같은 후보 plant 값이 실물 응답에 더 가까워지는지 검증

즉, 현재 단계만으로도 “이 후보 물성값이 실물에 맞는가?”를 공정하게 평가할 수
있습니다. 그러나 현재 runner는 아직 파라미터를 스스로 찾는 자동 최적화기는
아닙니다.

완전한 Isaac-real SysID는 다음 단계에서 진행합니다. ETRI 측에서 rich SysID 로그
(`q_ref`, `q`, `qd`, 신뢰 가능한 `tau_meas`, timestamp)를 제공하면, 이를 Isaac Sim
6.1 System Identification backend에 연결합니다. 그러면 A/B/friction/delay 후보를
자동으로 탐색하고, 별도로 남겨 둔 validation trajectory에서 정말 개선됐는지 평가할
수 있습니다. 현재 runner와 overlay 도구는 그 자동 fitting 결과를 독립적으로
검증하는 기준으로 계속 사용합니다.

## 실물 로그를 받은 뒤 실행 순서

1. 실로봇에서 consistency probe 또는 SysID probe를 실행해 CSV를 받습니다.
2. CSV 파일을 `data/sysid/.../sysid_logs/real/` 같은 한 폴더에 둡니다.
3. CSV에는 14개 관절의 `q0_*`, `q_*`, `qd_*`가 있어야 합니다. 토크 기반 분석까지
   하려면 `tau_meas_*`도 유한한 값으로 기록되어야 합니다.
4. 아래 명령의 `--ref-dir` 경로를 실제 CSV 폴더로 맞춘 뒤 repository root에서 실행합니다.

```bash
conda run -n env_isaaclab python scripts/sim2real/run_dual_fr3_sysid.py \
  --headless --sweep --profile ramp \
  --amplitude-rad 0.05 --amplitude-wrist-rad 0.03 \
  --ref-dir data/sysid/sysid_logs_260911_1422/sysid_logs/real
```

위 명령은 양팔 14관절을 순서대로 실행합니다. 각 관절에 대해 runner는 다음을
자동으로 처리합니다.

- 같은 관절 이름, `ramp` profile, 진폭에 맞는 최신 real CSV를 찾음
- 그 파일의 14개 `q0_*`를 Isaac의 해당 trial 시작 자세로 정확히 사용
- Isaac에서 같은 ramp를 실행하고 q, qd, torque CSV를 기록
- 어떤 real CSV와 q0를 사용했는지 `run_metadata.json`에 기록

따라서 사용자가 매번 q0, Kp, Kd, torque limit을 수동으로 입력할 필요가 없습니다.
다만 real 로그의 진폭과 명령의 진폭은 같아야 합니다. 예를 들어 실물 wrist를
`0.03 rad`로 기록했다면 `--amplitude-wrist-rad 0.03`을 유지합니다.

결과는 `logs/sysid/isaaclab/<날짜_시간>/`에 저장되며 Git에는 포함되지 않습니다.
그 폴더의 `run_metadata.json`에서 실행 성공 여부와 사용한 real CSV를, 각 CSV에서
Isaac의 응답을 확인합니다.

## 처음 실행할 때: 단일 관절 확인

새 PC, 새 Isaac 환경, 또는 새 실물 로그를 처음 연결할 때는 전체 14관절 sweep 전에
아래처럼 left joint1 하나만 짧게 확인하는 것이 좋습니다.

```bash
conda run -n env_isaaclab python scripts/sim2real/run_dual_fr3_sysid.py \
  --headless --target-joint 0 --profile ramp --amplitude-rad 0.05 \
  --t-start 0.1 --ramp-time 0.1 --hold-time 0.1 \
  --initial-settle 0.1 --home-settle 0.1 --home-timeout 10 \
  --ref-dir data/sysid/sysid_logs_260911_1422/sysid_logs/real
```

실행 전 runner는 PhysX에 A/B/friction이 실제로 적용됐는지 읽어 확인합니다. 값이
현재 학습 plant 설정과 다르면 잘못된 조건으로 실행하지 않고 즉시 실패합니다.

## Action semantics ablation: controller-1 snapshot target

기존 `Isaac-Reach-DualFR3-v0`는 Isaac Lab 기본 relative action을 사용합니다. 즉 policy가
20 Hz로 낸 delta를 유지하고, 매 10 ms physics step마다 최신 `q_meas`에 더하는 controller-3
방식입니다. 기존 task와 기존 checkpoint는 기준선으로 보존합니다.

새 `Isaac-Reach-DualFR3-Snapshot-v0`는 **controller-1** 전용 task입니다. policy tick `t_k`에만
`q_cmd = q_meas(t_k) + 0.1 * action[k]`을 계산하고, 다음 50 ms policy tick까지 그 absolute
target을 고정합니다. robot/plant, observation 58D, reward, 20 Hz policy, 100 Hz physics, PD gain,
effort limit, domain randomization은 v0와 같습니다. 따라서 target 생성 semantics만 비교합니다.

새 정책은 기존 v0 checkpoint를 재사용하지 말고 처음부터 학습합니다.

```bash
conda run -n env_isaaclab python scripts/rsl_rl/train.py \
  --task Isaac-Reach-DualFR3-Snapshot-v0 --headless
```

학습 전에는 새 action term이 정말 50 ms hold를 만드는지 아래 검증을 한 번 실행합니다.

```bash
conda run -n env_isaaclab python scripts/sim2real/reach_snapshot_action_check.py --headless
```

공정한 비교는 v0와 Snapshot을 같은 seed뿐 아니라 최소 3개 seed로 각각 학습한 뒤 수행합니다.
실물 비교 전에는 free-space reach에서 1 kHz `q_cmd`, `q_meas`, `qd`, applied torque를 기록합니다.
Snapshot에서는 각 50 ms window 안에서 `q_cmd-q_meas`가 줄어드는지를 평가할 수 있습니다. plate
contact/squeeze 구간의 position error는 의도된 접촉력일 수 있으므로 이 수렴 지표에 섞지 않습니다.

## 관련 스크립트

- `sysid_isaac_recorder.py`: 이전 로그 형식과 바로 비교하기 위한 호환 recorder입니다.
- `sysid_overlay_plot.py`: real/MuJoCo/Isaac 응답을 겹쳐 그립니다.
- `sysid_inertia_dump.py`: Isaac 관절 물성 및 mass matrix를 확인합니다.
- `sysid_inertia_probe.py`: 상수 torque로 유효 관성을 점검합니다.
- `reach_snapshot_action_check.py`: controller-1 task에서 policy-tick `q_cmd`가 다섯 physics
  substep 동안 고정되는지 확인합니다.
