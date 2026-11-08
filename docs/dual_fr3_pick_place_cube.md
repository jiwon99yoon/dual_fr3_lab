# Dual FR3 큐브 pick-and-place (오른팔, UMI 팁, 75 cm 책상)

`Isaac-PickPlace-Cube-DualFR3-UMI-v0` / `-Play-v0` (2026-11-07 작성, 11-08 실물 그리퍼·DR 반영, 11-09 첫 학습 결과로 보상 수정). 코드: `tasks/manager_based/pick_place_cube/`.
rule-based 확인: `scripts/tools/eval_dual_fr3_pick_place_rule_based.py`.

책상 위에 세워 둔 큐브(5×5×10 cm)를 오른손으로 위에서 집어요. 무작위 목표 위치에 세워 놓고, 손을 뗀 뒤 위로 올리면 성공이에요.
왼팔은 ready 자세로 서 있어요(action 없음). 제어 계약은 lift Sim2Real과 같아서 배포 노드 구조도 그대로 써요.

## 지금 상태와 할 일 (2026-11-09 01:40)

11-09 새벽에 첫 학습 4개(5070 2개, 4090 A·B)를 보고 task를 고쳤어요. 고친 내용은 아래 "11-09 수정"에 있어요.
**지금 할 일: 수정 커밋을 pull하고, 세 run을 새로 시작해요.** 세 run 모두 actor는 A(그리퍼 진행 중 비트, 37D)예요.

| PC | run | task | 보는 것 |
|---|---|---|---|
| **5070** | ① 기준 | `Isaac-PickPlace-Cube-DualFR3-UMI-GripperBusy-v0` | 11-09 수정 전부. **11-09 01:27 시작** (`2026-11-09_01-27-06_fix_5070`, 약 1.6 s/it) |
| **4090** | ② 기울기 벌점 | `Isaac-PickPlace-Cube-DualFR3-UMI-GripperBusy-Tilt-v0` | ① + 기울기 15° 초과분 −5 (캐비닛 책 -Tilt와 같음). **11-09 01:28 시작** (`2026-11-09_01-28-37_tilt_4090`, ③과 동시에 2.6~2.9 s/it) |
| **4090** | ③ critic 정보 | `Isaac-PickPlace-Cube-DualFR3-UMI-GripperBusy-CriticPlus-v0` | ① + B의 critic 정보 (그리퍼 내부 상태 8D, DR 값 7D). **11-09 01:29 시작** (`2026-11-09_01-28-57_critic_plus_4090`) |

새로 시작할 때 멈출 것:
- 5070의 `pick_place_lift_5070` (들기 보상만, 960 iteration에 들기 1%)
- 4090의 A·B (수정 전 코드)

```bash
# 5070
python scripts/rsl_rl/train.py --task Isaac-PickPlace-Cube-DualFR3-UMI-GripperBusy-v0 --headless --num_envs 4096 --run_name fix_5070
# 4090 (git pull 뒤, 터미널 두 개에서 동시에)
python scripts/rsl_rl/train.py --task Isaac-PickPlace-Cube-DualFR3-UMI-GripperBusy-Tilt-v0 --headless --num_envs 4096 --run_name tilt_4090
python scripts/rsl_rl/train.py --task Isaac-PickPlace-Cube-DualFR3-UMI-GripperBusy-CriticPlus-v0 --headless --num_envs 4096 --run_name critic_plus_4090
```

- 로그 폴더: `logs/rsl_rl/dual_fr3_pick_place_cube_gripper_busy/`, `..._gripper_busy_tilt/`, `..._gripper_busy_critic_plus/`
- 4096 env 하나가 GPU를 약 5 GB 써요. 4090에서 둘을 같이 돌리면 각각 약 2.9 s/it이에요.
- 측정: `scripts/tools/eval_dual_fr3_pick_place_grasp.py` (4090, 11-09). 세 run 모두 이걸로 같은 기준으로 재요. 기본 128 env, 결정적, 종료 끔(`--keep_terminations`로 켬). 손가락-책상 접촉은 `mdp.finger_table_force`(패드 센서 Table 필터)로 재요. --task는 그 run 변형의 -Play task예요(관측 차원이 맞아야 해요).
```bash
python scripts/tools/eval_dual_fr3_pick_place_grasp.py --headless --task Isaac-PickPlace-Cube-DualFR3-UMI-GripperBusy-Tilt-Play-v0 \
  --checkpoint logs/rsl_rl/dual_fr3_pick_place_cube_gripper_busy_tilt/<run>/model_N.pt
```

**비교할 것** (같은 iteration끼리)
- ① vs ②: 기울기 벌점의 효과. 옮기는 동안 기울기, 성공률, 배우는 속도
- ① vs ③: critic 정보의 효과. 쥐는 순간·옮기는 동안 기울기, 손가락-책상 접촉, 배우는 속도
- 손 놓기와 성공 사이: 이전 B는 1423 it에 손 놓기 84%, 성공 0%였어요(손 올리기에서 막힘). 새 코드의 놓은 뒤 유지 범위(5 cm·15°)는 성공 조건(2.5 cm·5°)보다 넓어서 이 차이가 더 커질 수 있어요. `reach_released`와 `reach_success`를 같이 봐요.
- TensorBoard: `Curriculum/progress/reach_*`, `Episode_Reward/success`, `finger_table`, `tip_floor`, `cube_tilt`, `cube_tilt_excess`(②)
- 평가: 종료를 끄고 결정적으로 128 env. 잡는 깊이, 쥐는 순간·들고 있을 때 기울기, 손가락-책상 접촉, 놓을 위치에서 열기 명령 비율

**실물 배포** (세 run 모두 같아요)
- 노드 관측 끝에 그리퍼 진행 중 비트를 추가해요. 그리퍼 goal을 보낸 뒤 result를 받기 전까지 1이에요.
- 노드의 `gripper_hysteresis`(0.1)를 0으로 둬요. sim은 행동 부호(0 기준)로 열고 닫아요.

## 11-09 수정 (첫 학습들의 결과)

### 지금까지의 run (모두 멈추거나 멈출 예정)

| run | 코드 | 결과 |
|---|---|---|
| 5070 첫 run `2026-11-09_00-03-44_pick_place_dr_5070` | 처음 코드 | 900 it에서 멈춤. 들기 0.2~0.7%에서 정체. model_700(64 env): 파지 64/64, **들기 0/64**, 쥔 채 14 s 가만히 있음. 영상 `videos/model_850_oblique4.mp4` |
| 5070 `2026-11-09_00-32-42_pick_place_lift_5070` | + 쥔 채 들기 높이 shaping (d69f771) | 960 it에 들기 1~3%, 놓을 위치 0%. model_300(확률적 행동, 256 env): 4 cm 든 4개 모두 **정책이 열기 명령을 보내 떨어뜨림**. 들고 있는 동안 그리퍼 행동 평균 0.03(닫기·열기 사이에서 못 정함), 들고 있는 step의 57%가 열기 요청 |
| 4090 A `-GripperBusy-v0` | 처음 코드 + actor 그리퍼 진행 중 비트 | **270 it에 들기 85%**, 500 it 놓을 위치 90%, 손 놓기 4% → 800 it 0.6%. 성공 0.1% 이하. **1444 it에서 멈춤: 놓을 위치 91%, 손 놓기 1.2%, 성공 0.9%** |
| 4090 B `-CriticPlus-v0` | 처음 코드 + critic 정보 | 620~650 it에 스스로 들기를 넘음. 800 it 들기 92%, 놓을 위치 37%, 손 놓기 0.3%, 성공 0. **1423 it에서 멈춤: 놓을 위치 86%, 손 놓기 84%, 성공 0%** — 손 놓기는 스스로 배웠지만 손 올리기(TCP가 큐브 윗면 + 3 cm)에서 막혔어요(깊게 잡아 약 10 cm를 올려야 함, 원인은 측정 전) |

- 그리퍼 진행 중 비트는 들기를 약 2.4배 빨리 배우게 했어요(270 vs 620~650 it). 비트가 없으면 actor는 그리퍼 행동이 실제로 먹히는 순간(그리퍼가 쉴 때)과 무시되는 순간을 구분하지 못해서, 그리퍼 출력의 학습 신호가 흐려지는 걸로 봐요.
- critic 정보(B)는 늦게 배웠지만 파지가 더 조심스러웠어요(아래 표).

### A·B model_800 평가 (4090, 128 env, 결정적, 종료 끔)

| | A GripperBusy | B CriticPlus |
|---|---|---|
| 놓을 위치에 머문 시간 | 11.75 s | 10.5 s |
| 놓을 위치에서 열기 명령을 보낸 비율 | 0% | 0% |
| 놓을 위치에서 목표와 xy 오차 | 0.83 cm | 2.13 cm |
| 잡는 깊이 (큐브 윗면에서 손가락 끝, 의도 3 cm) | 7.5 cm | 6.9 cm |
| 쥐는 순간 기울기 | 8° 안팎 (model_700, 깊이에 따라 6~29°) | 3.1° |
| 들고 있을 때 최대 기울기 | 28~33° | 8~12° |
| 손가락이 책상에 닿은 env | 80/128 (최대 103 N) | 14/128 |
| 도달한 단계 (종료 끔) | 놓을 위치 118/128, 손 놓기 0/128 | 놓을 위치 11/128, 손 놓기 0/128 |

위 접촉 값은 패드 접촉력에서 큐브와의 접촉을 뺀 값이에요. A model_800을 `eval_dual_fr3_pick_place_grasp.py`(Table 필터, 현재 코드의 단계 판정)로 다시 재면: 쥘 때 손가락 끝 책상 위 2.8 cm, 잡는 깊이 7.5 cm, 쥐는 순간 기울기 중앙 8.6°(깊이에 따라 4~24°), 손가락-책상 접촉 86/128(최대 138 N), 놓을 위치 106/128, 손 놓기 0/128, 놓을 위치에서 열기 명령 0%예요.

**A가 놓을 위치에서 손을 놓지 않는 이유 (코드 확인)**
- placed 단계 shaping이 "손가락 벌린 정도"였어요. 쥔 손가락은 23.4 mm이고 25 mm를 넘으면 바로 패드가 떨어져 다음 단계로 넘어가요. 그래서 이 shaping은 사실상 늘 0이었고, 열기를 시작할 신호가 없었어요.
- 손을 놓은 뒤 큐브가 2.5 cm·5°를 조금만 벗어나도 단계가 0으로 떨어졌어요. 보상이 초당 20에서 약 3으로 급락해요. 열지 않는 게 안전한 선택이었어요.

### A model_700 파지 깊이 (4090 측정, 128 env)

- 손가락 끝이 책상 위 중앙값 2.3 cm(최소 0.7 cm)에서 쥐었어요. 보상이 의도한 높이는 7 cm였어요. 잡는 깊이 중앙값 7.7 cm(3.3~9.3)예요.
- 손가락이 책상에 닿은 env 103/128, 최대 136 N, env당 약 0.2 s예요. 거의 전부 다가가서 쥐는 순간이에요.
- 손가락-책상 접촉은 벌점이 없었어요. `arm_contact`는 link3~7(손 포함)만 보고, 손가락은 따로 된 body라 빠져 있었어요.

| 잡는 깊이 | env | 쥐는 순간 기울기 | 들고 있을 때 최대 기울기 | 놓을 위치 도달 |
|---|---|---|---|---|
| 5 cm 미만 | 7 | 29° | 33° | 6/7 |
| 5~7 cm | 25 | 18° | 32° | 20/25 |
| 7~8.5 cm | 87 | 8° | 33° | 76/87 |
| 8.5 cm 이상 | 9 | 6° | 35° | 6/9 |

깊게 잡을수록 쥐는 순간 덜 기울지만, 옮기는 동안 기울기는 깊이와 관계없이 32~35°예요.

### rule-based 파지 높이 비교 (256 env)

정책과 같은 큐브 관측을 썼어요(`--obs_pose`: 에피소드 오프셋 ±5 mm·±3°, 잡음, 지연·끊김). 다가가는 동안 평균한 값이고, 닫기 시작할 때 오차는 xy 중앙값 4.2 mm, p90 6.6 mm예요.

| 손가락 끝 높이 (책상 위) | 성공 | 닫는 동안 최대 기울기 (중앙값 / p90) | 쥔 순간 기울기 중앙값 | 손가락-책상 접촉 env |
|---|---|---|---|---|
| 7 cm (이전 파지점) | 251 | 0.8° / 2.8° | 0.4° | 0 |
| **5 cm (큐브 중심, 새 파지점)** | **255** | 0.3° / 1.0° | 0.2° | 7 (최대 61 N) |
| 4 cm | 254 | 0.2° / 0.9° | 0.2° | 36 (86 N) |
| 2.3 cm (A와 같은 깊이) | 232 | 0.1° / 0.3° | 0.1° | 95 (133 N) |
| 7 cm, 실제 pose | 254 | 0.4° / 0.8° | 0.3° | 0 |

- 천천히 다가가 멈춘 뒤 닫으면, 관측 오차가 있어도 어느 높이에서든 쥔 순간 기울기가 2° 이하예요. 정책의 쥐는 순간 기울기(A 8~29°, B 3°)는 관측 오차보다 정책 자신의 움직임(움직이면서 닫기, 손가락으로 치기)에서 온다고 봐요.
- 5 cm가 성공률이 가장 높고, 7 cm보다 닫을 때 덜 기울어요. 더 깊으면 책상 접촉이 빠르게 늘어요.
- 5 cm의 접촉 7개와 하한 아래 step은 스크립트 동작 때문이에요. 0.25 m/s로 내려가다 목표보다 2 cm 넘게 더 내려갔다 돌아와요(200/256 env가 하한 아래 평균 5.5 step). 정책은 하한 벌점 때문에 천천히 내려가는 법을 배워야 해요.

### 수정 내용

| | 내용 | 코드 |
|---|---|---|
| 1 | 쥔 채 들어 올린 높이 shaping. 파지 단계 shaping을 다가가기 0.4 + 방향 0.2 + 파지 0.2 + 들기 0.2로 (이전 0.5 + 0.25 + 0.25) | d69f771 |
| 2 | 파지점을 큐브 중심으로 (손가락 끝 책상 위 7 → 5 cm, `GRASP_OFFSET` 0.02 → 0) | `mdp/pick_place.py` |
| 3 | 손가락-책상 접촉 1 N 넘으면 −5 (`finger_table`). 패드 센서 filter에 Table을 추가했어요(filter[1]) | `mdp.finger_table_contact` |
| 4 | 손가락 끝이 책상 위 3 cm보다 낮은 만큼 −10 (`tip_floor`, 책상 높이에서 1). 손이 책상 위(앞 모서리에서 3 cm 앞까지)일 때만이에요. ready TCP는 책상 앞(x 0.30)이에요 | `mdp.tip_below_floor` |
| 5 | placed 단계 shaping을 열기 요청 0.5 + 손가락 벌림 0.5로. 열기를 요청한 순간 보상해요 | `PickPlaceState.update` |
| 6 | 한 번 placed였으면, 손을 놓은 뒤 큐브가 목표 5 cm·기울기 15° 안에 서 있는 동안 released를 유지해요. 성공 판정은 그대로 2.5 cm·5°예요 | `RELEASE_XY_TOL`, `RELEASE_TILT_DEG` |
| ② | `-GripperBusy-Tilt`: 기울기 15° 초과분(rad) × −5, 언제나 | `mdp.cube_tilt_excess` |
| ③ | `-GripperBusy-CriticPlus`: A의 actor + B의 critic | `_add_critic_plus` |

- 3·4는 처음부터 켜요. 파지점(5 cm)이 하한(3 cm)보다 높아서 다가가기만 하면 벌점이 없어요.
- 6을 placed를 한 번 거친 경우로 묶은 이유: 목표 근처에서 대충 떨어뜨리는 행동은 보상받지 않게 하려고요. 목표 5 cm 밖에 놓으면 지금처럼 −100 종료예요.
- 확인:
  - rule-based(5 cm, 정책 관측) 255/256
  - 놓기 유지: 큐브를 직접 옮겨 놓고 확인했어요. placed를 거친 뒤 1 cm·3 cm는 released, 6 cm와 placed 없이 3 cm는 reach예요.
  - smoke: 세 변형 각 2 iteration, 256 env. actor 37D, critic 60 / 60 / 75D

### 처음 계획한 실험 A, B (11-09 00시)

| | 바꾸는 것 | 관측 차원 |
|---|---|---|
| A GripperBusy | actor 관측 맨 뒤에 그리퍼 명령 진행 중 비트(`mdp.gripper_busy`, 1 = 아직 안 끝남) | actor 37D, critic 23D |
| B CriticPlus | critic에만 그리퍼 내부 상태 8D(단계 one-hot, 타이머, 요청·보냄·목표 비트)와 DR 값 7D(큐브 pose 관측 오프셋 4, 손가락 강성·속도 배율 2, 비전 지연 1) | actor 36D, critic 38D |

다음 후보: gamma 0.99 → 0.995(작업 약 9 s, 그리퍼 대기 두 번), actor 관측 이력(최근 3 step, 배포 노드 변경이 커서 나중에).

## 배치 (base frame, base는 world z 0.405)

| 항목 | 값 |
|---|---|
| 책상 | 실물 600(x) × 1000(y), 앞 모서리 x 0.40, 윗면 z 0.345 (캐비닛 Table75와 같은 위치·높이) |
| 큐브 | 5×5×10 cm, 실측 90 g |
| 집는 곳·놓는 곳 | x 0.45~0.65, y −0.45~0.05, 오른팔 link0 축에서 수평 0.62 m 안. yaw ±45°. 목표는 집는 곳에서 10 cm 이상 떨어뜨려요 |
| 파지점 | 큐브 중심 (손가락 끝이 책상 위 5 cm, 11-09에 7 cm에서 내림). TCP = UMI 손가락 끝 중앙, 손가락 길이 11 cm |
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

- actor 36D: 오른팔 관절 위치 7, 손가락 폭 1(갱신 지연), 관절 속도 7, 큐브 위치 3·자세 4(ArUco, 지연·끊김·오프셋), 큐브 − TCP 3, 목표 위치 3, 이전 action 8. `-GripperBusy` 계열은 맨 뒤에 그리퍼 진행 중 비트 1 (37D)
- critic +23D: 실제 큐브 위치·자세·속도, TCP, 단계 one-hot 5, 파지·접촉·들었음·비전 끊김, 책상 높이 오프셋. `-CriticPlus` 계열은 그리퍼 내부 상태 8 + DR 값 7 (+38D)
- 단계: reach → held → placed → released → success. 보상은 단계 기본값(0, 2, 4, 6, 8) + 단계 안 shaping(0~1)에 가중치 5를 곱해요

| 단계 | shaping |
|---|---|
| reach | 파지점까지 거리 0.4 + 손 방향 0.2 + 양쪽 패드 파지 0.2 + 쥔 채 들어 올린 높이 0.2 (11-09) |
| held | 큐브 → 목표 거리 (넓게 0.5 + 좁게 0.5) |
| placed | 열기 요청 0.5 + 손가락 벌림 0.5 (11-09, 전에는 손가락 벌림만이라 늘 0) |
| released | 손 올리기. 한 번 placed였으면 목표 5 cm·15° 안에서 유지 (11-09) |
| success | 1 |

- 그 밖의 보상:
  - 넘어짐·떨어뜨림·책상 밖: −100 (종료)
  - 들고 있을 때 기울기: −2
  - 관절 한계 여유: −1
  - 손가락-책상 접촉: −5 (11-09)
  - 손가락 끝 높이 하한 3 cm: −10 (11-09)
- action rate·관절 속도·팔 접촉 벌점은 placed 도달률이 0.3을 넘으면 커리큘럼으로 강화해요.

## rule-based 결과 (256 env, 실제 큐브 pose, 0.25 m/s)

| 날짜 | 조건 | 성공 | 비고 |
|---|---|---|---|
| 11-07 | 그리퍼 지연 없음, 1 cm, 10 s | 249/256 | 실패는 손을 펼 때 큐브가 0.7~1.1 cm 밀린 경우 |
| 11-08 | 지연 그리퍼, 1 cm, 12 s | 251/256 | 성공 중앙 9.1 s, 최대 11.8 s. 4개가 시간 부족 |
| 11-08 | + 30 N, 14 s | 253/256 | 들고 있을 때 패드 힘 30.4 N |
| 11-08 | + 성공 2.5 cm | 255/256 | |
| 11-08 | + DR 전부 | 254/256 | 실패 2개는 스크립트 한계(4 mm 안까지 내리다 시간 초과, 관절 한계 근처 자세에서 IK가 3 cm 못 미침) |
| 11-09 | 정책과 같은 큐브 관측 (`--obs_pose`), 파지점 7 cm | 251/256 | 위 "rule-based 파지 높이 비교" |
| 11-09 | + 파지점 5 cm (큐브 중심), 새 보상 | 255/256 | 손가락-책상 접촉 7/256 (스크립트가 내려갈 때 2 cm 넘게 더 내려감) |

닫기는 명령부터 끝까지 1.25~2.0 s, 열기는 1.35~1.9 s 걸려요.
영상: `logs/rule_based/pick_place_cube/rule_based_pick_place_delay_30N_14s.mp4` (DR 넣기 전).

## 레퍼런스와 비교 (11-08 조사)

- 벤치마크(Isaac Lab Lift, ManiSkill PickCube, Fetch PickAndPlace, robosuite PickPlace)는 그리퍼가 즉시 움직이고 정확한 상태를 관측하며, 대부분 공중의 점에 큐브를 갖다대면 성공이에요(2.5~5 cm).
- 실물에서 놓기까지 한 sim2real은 많지 않아요.
  - RGB-Stacking(DeepMind, CoRL 2021): Sawyer, 성공 기준 3 cm, sim2real 67.9% → 실물 offline RL로 81.6%. binary 그리퍼가 전이에 가장 중요했어요(연속이면 20.7%).
  - Pinto et al.(RSS 2018): Fetch, asymmetric actor-critic + DR, Pick 5/5.
  - MuJoCo Playground(2025): Franka 큐브 들기 12/12, 그리퍼 지연을 최대 250 ms까지 무작위로 넣었어요.
- 우리는 그리퍼 지연을 실측값(최대 0.9 s + 대기)으로 넣었는데, 이렇게 한 레퍼런스는 찾지 못했어요. 성공 2.5 cm는 레퍼런스 수준이에요.
