# 박스 들기: rule-based vs RL 5499, 같은 조건 비교 (2026-11-13)

"왜 RL인가"에 답하기 위한 sim 비교. nominal sim에서는 rule-based 조임-들기도 손 닿는 범위에서 98 %라
(docs/dual_fr3_lift_sim2real.md "box Light RL 결과와 rule-based 비교") 성공률만으로는 RL을 정당화할 수 없었다.
여기서는 두 제어기를 **같은 env, 같은 시작 표, 같은 제어 계약** 아래 두고 실물에서 본 gap(낮은 마찰, 약한 조임, 무거운 박스,
관측 오염)을 하나씩 넣어 비교한다. 결과는 KRoC 시스템 논문의 "왜 RL" 표 후보.

## 1. 설정

| 항목 | 값 |
|---|---|
| env | `Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip-BoxDrop-FlatSqueeze-LowMuSoftLift-GoalXY-Play-v0` (5499가 학습한 env의 Play) |
| 시작 | 둘 다 env의 pregrasp 표(판이 면 4 cm 밖, 박스 중심 +5.5 cm, 모서리 정렬). 실물 배포의 MoveIt 접근 뒤 상태 |
| 제어 계약 | Δq = clip(0.1·a, ±0.1 rad) @ 20 Hz, explicit PD Kp 80 / Kd 4, FR3 한계 |
| 공통 DR(기본 조건) | 질량 0.4~0.6 kg, 박스 μ 0.1~0.8 / 판 0.3~0.5, Kp ±15 %, 박스 크기 ±1 cm, 외란 1.0, 목표 x/y 박스와 따로 |
| 성공 | env success 항: 목표 10 cm 이내 + 기울기 15° 이내 + 양 판 ≥ 1 N을 0.5 s 유지 (10 s 안에 한 번) |
| 규모 | 512 env, seed 0, 200 step(10 s) |
| RL | GoalXY model_5499 (실물 8/10), `scripts/tools/eval_dual_fr3_box_policy.py` |
| rule-based | `scripts/tools/eval_dual_fr3_plate_rule_based.py --from_pregrasp --carry_xy --roll_align --gain_free 1.0 --damp 0.05 --v_contact 0.05` (11-03 권장 설정). pregrasp 높이에서 면 안쪽 5 cm로 조이고(SETTLE 0.3 s) 목표 x/y/z로 운반 |
| 관측 오염 조건 | 정책: 관절 σ 0.005 rad, 박스 위치 σ 1 cm / 자세 σ 2°, 지연 0~150 ms, 끊김. rule-based도 같은 오염된 박스 pose(`--obs_pose --corrupt`) |

조건별 명령: scratchpad `cmp/run_matrix.sh`(이 문서 끝에 복사).

## 2. 결과

| 조건 | rule-based | RL 5499 | 성공까지 중앙값 s (rule / RL) | 들 때 최대 기울기 중앙값 ° (rule / RL) | 조임 N (rule settle / RL 끝) |
|---|---|---|---|---|---|
| 기본 (학습 DR) | 89.3 % | **99.0 %** | 4.05 / 2.00 | 3.7 / 9.2 | 18.7 / 18.2 |
| + 관측 오염 | 86.7 % | **99.6 %** | 4.25 / 2.00 | 3.9 / 9.3 | 18.2 / 18.2 |
| 낮은 마찰 (박스 μ 0.1~0.3, 판 0.3) | 38.3 % | **82.4 %** | 4.10 / 2.05 | 6.5 / 10.1 | 18.4 / 18.2 |
| Kp × 0.7 | 22.1 % | **75.2 %** | 4.10 / 2.55 | – / 10.0 | 13.2 / 13.2 |
| Kp × 0.5 | 0.0 % | 23.8 % | – / 3.95 | – / 25.1 | 9.6 / 9.7 |
| 무거운 박스 0.8~1.2 kg | 16.4 % | 44.7 % | 4.10 / 2.20 | – / 9.7 | 17.8 / 18.2 |
| 오염 + 낮은 마찰 + Kp × 0.7 | 0.4 % | 9.6 % | 4.60 / 2.90 | – / 20.5 | 12.8 / – |

rule-based 실패 내역(512 env 중): 기본 = 못 듦 10, 빠짐 34, 기울기 9; 낮은 마찰 = 못 듦 133, 빠짐 152, 기울기 29;
Kp 0.7 = 못 듦 281, 빠짐 106; 무거움 = 못 듦 346, 빠짐 75. "못 듦"은 판이 올라가는데 박스가 2 cm도 안 따라온 것(= 조이자마자 빠짐).

파지 위치: rule-based는 판 중심이 박스 중심 +5.5 cm(시작 표 높이), RL은 −3 cm. 조임 힘은 둘 다 18 N(PD 포화 0.1 rad × Kp 80에서 나오는 값이라 같다).

## 3. 해석

1. **조임 힘이 같은데 결과가 다르다.** 차이는 힘이 아니라 어디를 잡고 어떻게 드는가다. RL은 중심 아래 3 cm를 잡고(미끄러져도 무게가 회전
   토크를 덜 만든다) 들기 가속을 제한한 채 2 s에 끝낸다. rule-based는 중심 위를 잡고 일정 속도로 올리며, 조임과 들기가 분리돼 있어
   미끄럼에 반응하지 못한다.
2. **낮은 마찰 행이 실물 gap에 대응한다.** 실물 3998은 sim 100 %였지만 12회 중 11회 미끄러졌고, 5499는 낮은 마찰 DR로 학습해 8/10이었다.
   이 표에서 rule-based는 낮은 마찰에서 절반 넘게 빠지고 RL은 82 %를 지킨다. "왜 RL"의 1차 근거.
3. **Kp·질량은 둘 다 떨어지지만 RL이 3배 버틴다.** 조임이 PD 포화에서 나오는 구조는 같아서 Kp 0.5에서는 RL도 24 %다. 이건 제어 계약의
   한계(±0.1 rad)이지 정책 차이가 아니다.
4. **관측 오염은 둘 다 거의 영향 없다**(−2.6 %p / +0.6 %p). 이 행은 "왜 RL"이 아니라 "비전 지연·끊김은 이 계약 안에서 문제가 아니다"의 근거다.
5. **rule-based가 나은 것은 기울기뿐**(4° vs 9°). RL 정책의 기울기는 실물 5499에서도 6.7~7.8°였다.
6. 한계: rule-based는 11-03에 사용자 요청으로 튜닝을 멈춘 상태의 설정이다. 더 튜닝하면 기본 조건은 오르겠지만, 낮은 마찰·약한 Kp에서의
   차이는 파지 높이·들기 속도를 상황에 따라 바꾸는 능력에서 오므로 고정 규칙으로는 좁히기 어렵다. 파지 높이를 RL과 같게 하려는 시도는
   같은 시작 자세에서는 관절 한계 때문에 IK로 불가능했다(§4).

## 4. 파지 높이를 맞춘 변형 (rule-based, 박스 중심 −3 cm) — 같은 시작에서 rule-based는 못 내려간다

`--pregrasp_grip_dz -0.03`(pregrasp에서 DOWN으로 −3 cm까지 내려간 뒤 APPROACH)을 넣었더니 판이 내려가지 않고 앞·바깥·위로
10 cm 움직인 뒤 박스를 밀었다(16 env 진단, `cmp/diag_*.log`). null-space(ready 당김 0, 한계 밀기 0)·roll 정렬을 꺼도 같고,
`--limit_aware`(clamp된 관절을 빼고 IK)로도 판 높이는 +2.8~+3.5 cm에 머물고 조임 3.5~5.8 N, 성공 38~56 %(16 env)다.
원인은 pregrasp 표 자세가 FR3 한계 inset(0.05 rad)에 붙어 있어 DLS-IK가 요구하는 관절이 env에서 clamp되기 때문이다
(에피소드 내내 관절 여유 0.050 rad). RL 5499도 같은 여유 0.050에서 동작하지만 판 중심을 −3.1 cm까지 내려서 잡는다.
즉 **같은 시작 자세에서 "RL처럼 잡는" 규칙은 이 IK 제어기로는 만들 수 없었다.** 이 변형은 표에 넣지 않고, 파지 높이 차이 자체를
RL이 찾은 해로 기록한다. rule-based에 다른 시작 표(−3 cm용 `--pregrasp_gap --grip_dz -0.03 --save_table`)를 주는 것은
시작 조건이 달라지므로 이 비교의 범위 밖이다.

## 5. 다음

- hover(ready) 시작 Curriculum 정책이 나오면 같은 행렬을 `--from_pregrasp` 없이 한 번 더 돌린다(rule-based는 HIGH → DOWN → APPROACH).
- 실물 확인은 박스 rule-based 10회(같은 MoveIt 접근 뒤 `--from_pregrasp` 제어기)면 된다.
- 큐브·캐비닛도 같은 틀(rule-based 스크립트에 정책 관측·그리퍼 지연·DR 조건)로 돌리면 세 작업 표가 된다.

## 6. 명령

```bash
T=Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip-BoxDrop-FlatSqueeze-LowMuSoftLift-GoalXY-Play-v0
CK=logs/rsl_rl/dual_fr3_plate_lift_sim2real_100hz_light_pregrasp_wide_rollalign_facegrip_boxdrop_flatsqueeze_lowmusoftlift_goalxy/2026-11-05_05-20-47_goalxy_from_sl4500/model_5499.pt
COMMON="--task $T --headless --num_envs 512 --seed 0 --steps 200 --disturb 1.0"
RULE="--from_pregrasp --carry_xy --roll_align --gain_free 1.0 --damp 0.05 --v_contact 0.05"
# 조건: "" | "--corrupt (--obs_pose)" | "--box_mu 0.1 0.3 --plate_mu 0.3 0.3" | "--kp_scale 0.7" | "--kp_scale 0.5" | "--mass 0.8 1.2"
python scripts/tools/eval_dual_fr3_plate_rule_based.py $COMMON $RULE [--obs_pose] <조건>
python scripts/tools/eval_dual_fr3_box_policy.py $COMMON --checkpoint $CK <조건>
```
