# Dual FR3 큐브 pick-and-place (오른팔, UMI 팁, 75 cm 책상)

`Isaac-PickPlace-Cube-DualFR3-UMI-v0` / `-Play-v0` (2026-11-07 작성, 11-08 실물 그리퍼·DR 반영). 코드: `tasks/manager_based/pick_place_cube/`.
rule-based 확인: `scripts/tools/eval_dual_fr3_pick_place_rule_based.py`.

책상 위에 세워 둔 큐브(5×5×10 cm)를 오른손으로 위에서 집어요. 무작위 목표 위치에 세워 놓고, 손을 뗀 뒤 위로 올리면 성공이에요.
왼팔은 ready 자세로 서 있어요(action 없음). 제어 계약은 lift Sim2Real과 같아서 배포 노드 구조도 그대로 써요.

## 지금 상태와 할 일 (2026-11-09)

| PC | 학습 | 상태 |
|---|---|---|
| **5070** | 기준 run `Isaac-PickPlace-Cube-DualFR3-UMI-v0` (들기 보상 추가) | **11-09 00:32 다시 시작.** 로그 `logs/rsl_rl/dual_fr3_pick_place_cube/2026-11-09_00-32-42_pick_place_lift_5070`. 4096 env, 5000 iteration, 1.7~1.9 s/it(약 2.5시간) |
| **4090** | **실험 A (`-GripperBusy-v0`)와 B (`-CriticPlus-v0`)** | **할 일: 들기 보상 커밋을 pull한 뒤 둘 다 다시 학습.** 들기 보상 전에 시작한 run은 멈추고 지워도 돼요. 아래 명령 참고 |

### 첫 run (들기 보상 없음, 11-09 00:03~00:30, 멈춤)

`2026-11-09_00-03-44_pick_place_dr_5070`, 900 iteration에서 멈췄어요.
- 들기(held) 도달률이 250 iteration부터 0.2~0.7%에서 정체됐어요. progress 보상도 3.4에서 그대로였고, 넘어짐 종료는 7%에서 13%로 늘었어요.
- model_700 결정적 평가(64 env, 종료 끔): 다가가기 64/64(파지점까지 0.7 cm), 양쪽 패드 파지 64/64, **들기 0/64**. 쥔 채 14 s 동안 가만히 있었어요. 쥐는 순간 큐브가 15~30° 기울어요.
- 영상: `videos/model_850_oblique4.mp4` (위 run 폴더)
- 원인 추정: 4 cm를 들어야 다음 단계 보상이 시작돼서, 쥐고만 있어도 받는 보상과 들다 넘어뜨리면 받는 −100 사이에서 "쥐고 버티기"가 국소 최적이 됐어요. 지연 그리퍼 없는 handover는 같은 형태의 보상으로 300 iteration에 들기 99%였어요.
- 수정: 파지 단계 shaping을 다가가기 0.4 + 방향 0.2 + 파지 0.2 + **쥔 채 들어 올린 높이 0.2**(0~4 cm → 0~1)로 바꿨어요(이전 0.5 + 0.25 + 0.25). 1 cm만 들어도 보상이 늘어요.

### 4090에서 실행

5070에서 들기 보상 수정을 커밋·push한 뒤 4090에서 pull해요.

```bash
cd ~/dual_fr3_lab && git pull
# 터미널 1: 실험 A
python scripts/rsl_rl/train.py --task Isaac-PickPlace-Cube-DualFR3-UMI-GripperBusy-v0 --headless --num_envs 4096 --run_name gripper_busy_4090
# 터미널 2: 실험 B
python scripts/rsl_rl/train.py --task Isaac-PickPlace-Cube-DualFR3-UMI-CriticPlus-v0 --headless --num_envs 4096 --run_name critic_plus_4090
```

- 4096 env 하나가 GPU 메모리를 약 5 GB 써서, 4090(24 GB)에서 둘을 동시에 돌릴 수 있어요. 대신 각각 느려져요.
- 로그 폴더는 `logs/rsl_rl/dual_fr3_pick_place_cube_gripper_busy/`와 `logs/rsl_rl/dual_fr3_pick_place_cube_critic_plus/`예요.
- seed는 세 run 모두 runner 기본값이에요.

### 실험 A, B (기준 run과 한 가지씩만 다름)

| | 바꾸는 것 | 관측 차원 | 실물 배포 영향 | 기대 |
|---|---|---|---|---|
| 기준 (5070) | - | actor 36D, critic 23D | - | - |
| **A GripperBusy** | actor 관측 맨 뒤에 그리퍼 명령 진행 중 비트(`mdp.gripper_busy`, 1 = 아직 안 끝남) | actor **37D**, critic 23D | 노드에서 관측 끝에 1칸 추가. 그리퍼 goal을 보낸 뒤 result를 받기 전까지 1이에요(노드의 send·result 이벤트 사이) | 지연(0.45~0.9 s + 대기 0.3~0.4 s)을 손가락 폭(0.27 s마다 갱신)만으로 추측하지 않아도 돼요. 기다리는 법을 더 빨리 배울 것으로 봐요 |
| **B CriticPlus** | critic에만 그리퍼 내부 상태 8D(단계 one-hot, 타이머, 요청·보냄·목표 비트)와 DR 값 7D(큐브 pose 관측 오프셋 4, 손가락 강성·속도 배율 2, 비전 지연 1) | actor 36D, critic **38D** | 없음(actor가 기준과 같음) | 무작위 지연·오프셋 때문에 생기는 value 오차가 줄어 학습이 안정될 것으로 봐요 |

**비교할 것** (TensorBoard, 같은 iteration끼리)
- `Curriculum/progress/reach_held`, `reach_placed`, `reach_released`, `reach_success`: 단계별 도달률(EMA). placed가 0.3을 넘는 iteration이 빠른 쪽이 좋아요.
- `Episode_Reward/progress`, `Episode_Reward/success`
- `Episode_Termination/cube_toppled`, `cube_dropped`
- GPU와 동시 실행 여부가 달라 iteration당 시간은 비교하지 말고, 같은 iteration끼리 비교해요.

**다음 후보 (A·B 결과를 본 뒤)**
- C: gamma 0.99 → 0.995. 작업이 약 9 s(180 step)이고 그리퍼 대기가 두 번 있어요. 효과 있는 쪽에 더해요.
- D: actor 관측 이력(최근 3 step 쌓기). 배포 노드 변경이 커서 나중에 해요.

## 배치 (base frame, base는 world z 0.405)

| 항목 | 값 |
|---|---|
| 책상 | 실물 600(x) × 1000(y), 앞 모서리 x 0.40, 윗면 z 0.345 (캐비닛 Table75와 같은 위치·높이) |
| 큐브 | 5×5×10 cm, 실측 90 g |
| 집는 곳·놓는 곳 | x 0.45~0.65, y −0.45~0.05, 오른팔 link0 축에서 수평 0.62 m 안. yaw ±45°. 목표는 집는 곳에서 10 cm 이상 떨어뜨려요 |
| 성공 | 목표 xy **2.5 cm** 안(11-08 1 cm에서 완화), 기울기 5° 이내, 손 뗌, TCP가 큐브 윗면보다 3 cm 위 |

## 제어와 그리퍼 (실측 반영)

- 물리/PD 100 Hz, 정책 20 Hz, 에피소드 **14 s (280 step)**
- 팔: 오른팔 7관절 `q_target = q_measured + clip(0.1·a, ±0.1 rad)`, FR3 datasheet 한계, 팔 PD 게인 DR ±15%
- 그리퍼(`mdp/delayed_gripper.py`): binary. 실측(fr3_husky_etri `data/gripper_latency/20261108_*`)에 맞췄어요.
  - 명령 → 손가락 움직임 시작: 0.45~0.9 s, 명령마다 새로 뽑아요
  - 손가락 하나 0.05 m/s (실물 0.1 m/s = 두 손가락 폭 합)
  - 손가락이 멈춘 뒤 0.3~0.4 s 동안 다음 명령을 기다려요(실물 action result까지)
  - 손가락 폭 관측은 0.25~0.38 s마다만 갱신돼요(실물 joint_states 값 갱신 약 0.27 s)
  - 조이는 힘 30 N(실물 `grasp_force_n`). 손가락 PD 강성 1282 N/m로 맞췄어요. 5 cm 큐브에서 손가락 q가 23.4 mm라 30 N이 나와요
  - effort 상한으로 힘을 맞추면 안 돼요. 상한 10/20/30 N에서 실제 힘이 8/22/41 N이었고, 두 손가락이 같이 포화되면 큐브가 옆으로 밀렸어요

## DR

| 항목 | 범위 | 시점 |
|---|---|---|
| 큐브 마찰 / 질량 | 정지 0.4~1.2, 동 0.3~1.0 / 70~100 g | startup |
| 팔 PD 게인 | ±15% | startup |
| 조이는 힘 | 25~35 N (손가락 강성 배율) | startup |
| 손가락 속도 | ±10% | startup |
| 그리퍼 명령 지연·대기 | 위 그리퍼 항목 | 명령마다 |
| 시작 자세 | 오른팔 관절마다 ready ±0.05 rad | reset |
| 책상 높이 | ±1 cm. 상판이 kinematic이고, 판정·큐브 시작 위치는 env별 실제 높이를 써요. 목표 관측 z는 실물 노드처럼 명목 높이 | reset |
| 큐브 pose 관측 고정 오프셋 | 위치 축마다 ±5 mm, yaw ±3° (손-눈 보정 오차) | 에피소드마다 |
| 큐브 비전 지연·끊김 | 지연 0~150 ms, 끊김 0.1~1.0 s, 에피소드 10%는 도중부터 끝까지 끊김 | 에피소드·step |
| 관측 잡음 | 관절 0.005 rad, 속도 0.05, 손가락 1 mm, 큐브 위치 1 cm(실측 2~5 mm), 자세 0.02 | step |

큐브 크기 DR은 넣지 않았어요. env마다 크기를 다르게 하려면 `replicate_physics`를 꺼야 하는데, 실물 큐브 크기는 실측값이라 얻는 게 적어요.

## 관측과 보상

- actor 36D: 오른팔 관절 위치 7, 손가락 폭 1(갱신 지연), 관절 속도 7, 큐브 위치 3·자세 4(ArUco, 지연·끊김·오프셋), 큐브 − TCP 3, 목표 위치 3, 이전 action 8
- critic +23D: 실제 큐브 위치·자세·속도, TCP, 단계 one-hot 5, 파지·접촉·들었음·비전 끊김, 책상 높이 오프셋
- 단계: reach → held → placed → released → success. 보상은 단계 기본값 + 단계 안 shaping에 가중치 5를 곱해요
- reach 단계 shaping: 파지점까지 거리 0.4 + 손 방향 0.2 + 양쪽 패드 파지 0.2 + 쥔 채 들어 올린 높이 0.2 (11-09 추가)
- 그 밖의 보상: 넘어짐·떨어뜨림·책상 밖 −100(종료), 들고 있을 때 기울기 −2, 관절 한계 여유 −1. action rate·관절 속도·팔 접촉 벌점은 placed 도달률이 0.3을 넘으면 커리큘럼으로 강화해요

## rule-based 결과 (256 env, 실제 큐브 pose, 0.25 m/s)

| 날짜 | 조건 | 성공 | 비고 |
|---|---|---|---|
| 11-07 | 그리퍼 지연 없음, 1 cm, 10 s | 249/256 | 실패는 손을 펼 때 큐브가 0.7~1.1 cm 밀린 경우 |
| 11-08 | 지연 그리퍼, 1 cm, 12 s | 251/256 | 성공 중앙 9.1 s, 최대 11.8 s. 4개가 시간 부족 |
| 11-08 | + 30 N, 14 s | 253/256 | 들고 있을 때 패드 힘 30.4 N |
| 11-08 | + 성공 2.5 cm | 255/256 | |
| 11-08 | + DR 전부 | 254/256 | 실패 2개는 스크립트 한계(4 mm 안까지 내리다 시간 초과, 관절 한계 근처 자세에서 IK가 3 cm 못 미침) |

닫기는 명령부터 끝까지 1.25~2.0 s, 열기는 1.35~1.9 s 걸려요.
영상: `logs/rule_based/pick_place_cube/rule_based_pick_place_delay_30N_14s.mp4` (DR 넣기 전).

## 레퍼런스와 비교 (11-08 조사)

- 벤치마크(Isaac Lab Lift, ManiSkill PickCube, Fetch PickAndPlace, robosuite PickPlace)는 그리퍼가 즉시 움직이고 정확한 상태를 관측하며, 대부분 공중의 점에 큐브를 갖다대면 성공이에요(2.5~5 cm).
- 실물에서 놓기까지 한 sim2real은 많지 않아요.
  - RGB-Stacking(DeepMind, CoRL 2021): Sawyer, 성공 기준 3 cm, sim2real 67.9% → 실물 offline RL로 81.6%. binary 그리퍼가 전이에 가장 중요했어요(연속이면 20.7%).
  - Pinto et al.(RSS 2018): Fetch, asymmetric actor-critic + DR, Pick 5/5.
  - MuJoCo Playground(2025): Franka 큐브 들기 12/12, 그리퍼 지연을 최대 250 ms까지 무작위로 넣었어요.
- 우리는 그리퍼 지연을 실측값(최대 0.9 s + 대기)으로 넣었는데, 이렇게 한 레퍼런스는 찾지 못했어요. 성공 2.5 cm는 레퍼런스 수준이에요.
