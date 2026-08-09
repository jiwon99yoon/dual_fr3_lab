# Dual FR3 Plate-EE Box Lift — 환경 · Reward 정밀 분석

> 대상: `Isaac-Lift-Box-DualFR3-Plate-v0/v1/v2/P39` (`tasks/manager_based/lift_plate/joint_pos_env_cfg.py`, dual_fr3_lab repo). **P38 동결 기준은 Plate-v2, 현재 개발 task는 Plate-P39-v0**
> 알고리즘: PPO (rsl_rl), 기본 4096 envs
> 작성 기준: **2026-08-08** — P38 학습·3-seed full-disturbance 평가 완료, `model_9850.pt` 선택.
> run 이력·결과 상세는 `docs/log_per_task.txt` 참조
> Plate-v2 실물 전이 계약·문헌·P39 이후 계획: [`DUAL_FR3_LIFT_PLATE_SIM2REAL.md`](DUAL_FR3_LIFT_PLATE_SIM2REAL.md)
> 이력 (v0, 정육면체 30~50cm): P1 모서리 걸침 → P2 성공(모서리 몰림) → P3 실패(허공 정렬 국소해)
> → P4 성공(오른판 포기) → P5a/b 실패(+20cm 절벽) → P6b 성공·중력보상 채택 → P7 실패(상시 페널티)
> → P8 절반(쐐기 파지) → **P9 성공 77%** (excessive_squeeze 100N 픽스)
> 이력 (v1, 실물 박스 475×555×275·4kg·obs56): **P10 성공 82%** (P9 zero-shot 접촉 0 → v1 학습 필수 확정)
> → P11~P12(정렬 게이트 P→G→×upright: "기울인 세트-그립"/"어중간" 시소) → P13 수직리프트(발견 실패)
> → **P14** lift_progress로 발견 해결(정렬은 붕괴 지속) → **P15-v2 success 75%**(정렬 분리+거리게이트, but 정렬 0.005)
> → **P16** 정렬 w6+게이트제거 → P3 재발 → **P17** reaching_aligned로 통합(6·min[dist×align], 덧셈3항→곱셈1항)
> → **P18** 접촉품질 q(법선비율)를 lifting/tracking에 곱셈 도입 (success 12%로 하락, 파지 품질↑)
> → **P19** reaching 커리큘럼 6→12 (과잉 지배: reaching이 success 압도, 막판 붕괴)
> → **P20 (5k) reaching 6→9 = success 64%·reach kernel 0.33 (파지 최고)** → **P21 resume 10k / P22 scratch 10k**
> (P20~P22 관찰: 긴 학습이 success↑ 대신 파지 품질↓ 0.33→0.03 — 같은 리워드가 sloppy 파지 해를 허용)
> → **v2 신설 (외란)**: "좋은 파지를 reward로 명세하지 말고 물리로 필연화" — 박스에 랜덤 wrench.
> → **P23** (10N/2Nm, lifting 기준 즉시 full) = 품질↑(0.49) but success 0 + "쐐기 브레이싱" 국소해
>   (⭐ 교훈: 외란은 '안정'만 강제 — force closure는 평행의 필요조건이지 충분조건 아님)
> → **P24** (B+A: 외란 5N/1Nm success-기반 점진 ramp + plate_height_symmetry 페널티) = ⭐ **success 74%
>   (외란 full 속) + reach kernel 0.44 유지 — 품질↔성공 트레이드오프 첫 해소.** 잔존: "세트째 기울인 파지"
>   (판끼리 평행하나 박스 면과 안 평행 = flush 미달, q가 힘 '방향'만 보고 판 '자세'를 못 보는 사각지대)
> → **q' = q×align (soft ramp, [08032])**: flush 신호를 곱셈 자리에 — 지배 보상(36)의 스케일을 빌림.
>   weight 재조정 축은 기각 완료 (품질 w: 2 무시 / 9 부분 / 12 붕괴 — 덧셈은 task와 '경쟁', 곱셈은 '조건').
> → **P25** q'+크기DR 동시 변경(단일변수 위반) → 크기DR이 궤적을 나쁜 basin으로 밀어 success 0, q' 미작동 (폐기)
> → **P26** q' 격리(크기DR off): success ~65%·q' 첫 작동·play flush 개선, but q' 늦게 켜져(it4000) 미수렴.
>   eval: **drop 0/128(파지 안정)**·upright 0.54·box tilt ~13° → 남은 약점=기울기
> → **P27 resume 10k / P28 scratch 10k** (q' 수렴): **P28(scratch) 승 — success ~94% 안정** (P27 resume은 막판 하락).
>   ⚠️ **play(P27/P28 공통 발견)**: 접근은 평행 → **모서리만 접촉+진동** (에피소드 대부분) → **끝 1/8에 딱 정렬**.
>   해석: q'가 **방향(평행)은 해결**했으나 **위치(중심성)는 못 잡음** (align은 평행이면 모서리도 만점) +
>   success가 **1회성**이라 "끝에 0.5초만 정렬해 훔치기" 게이밍. **다음 = ① 중심성 ② success per-step 지속**.
> → **P29** (②: success 1회→**per-step 지속**, weight 1500→20, [08041]): eval **precise<10cm 0.71·upright 0.63·
>   tilt 11.6°** (P26 0.62/0.54/13°서 전반 개선 — 정렬 유지 인센티브 역전) + **drop 0/128**. 잔존: play서
>   **박스 yaw drift**(box_upright는 tilt만, yaw 자유) — 원인=두 판 접촉이 면 중심 아닌 COM 옆 → 커플.
> → **P30** (①: 중심성, [0804]): **q''=q×align×centrality** + yaw DR ±0.3→±0.1. **❌ 실패**: eval yaw 평균 **35°**·
>   aligned 0.03(목표 미달) + precise-hold **0.71→0.31 붕괴**(q''가 goal_tracking까지 곱해 인질) + 커리큘럼 지연.
>   ⭐진단: centrality는 **박스 로컬** 측정이라 박스 world yaw에 **불변** → yaw 구조적으로 못 잡음. tilt만 개선(8°).
> → **P31** (A: yaw 직접 페널티, [0804b]): centrality **폐기**(✅precise 0.70·tilt 8° 회복) + box_yaw 페널티(-3).
>   **❌ A 무효**: yaw 37°(P30 35°와 동일). ⭐**grip-axis 진단**: grip축 yaw=이상180°에서 **35° 벗어남**=박스yaw와
>   일치 → **팔이 grip을 35° 비틀어 놓고 박스가 따라감**(placement). reaching/align/q'가 박스상대라 base앵커 無.
> → **P32** (A': yaw goal_tracking 곱셈, [0805]): A 페널티 폐기 + **goal_tracking×(1−s·tanh(yaw_err/0.35))**
>   (`_yawtrack_scale` ramp, qprime 미러). yaw를 **지배보상(36)에 곱셈 앵커**(q'가 additive 이긴 논리).
>   **❌ 죽은 run**: yawtrack을 disturbance/q'와 같은 lo1/hi3로 걸어 success 조기(EMA1.9)에 yaw 켜짐 →
>   취약 success 붕괴(0 영구)·tilt 43°. ⭐**but yaw 37°→20.6°·aligned 0.03→0.28 개선** = A'(곱셈)는 yaw를
>   실제로 움직임(P31 A 페널티는 요지부동) → **H1 방식 맞음, 타이밍만 틀림**.
> → **P33** (yawtrack lo/hi 3/6, [0805b]): ⭐⭐**H1 확증** — eval **yaw 37°→3.1°·aligned 0.03→0.79**(거의 완벽)!
>   A'(곱셈)가 yaw를 실제로 정렬 → **H2(kinematic 한계) 기각**. **but precise 0.71→0.004 붕괴**: success 늦게 배워
>   (it9490) EMA가 3→6을 13iter만에 넘어 yawtrack이 **순간 full**(급발진) → 막판 position 버림.
> → **P34** (yawtrack **iteration ramp**, [0805c]): ✅**급발진 붕괴 차단 성공** — eval **precise 0.70·upright 0.78/
>   tilt 8°**(P33 0.004/19° 완전 회복!) + yaw 소폭(22.9°). but seed42가 또 discovery 늦어(it9490 success 폭발)
>   yaw scale 0.17까지만 참(ramp 미완). 끝 상태 이상적(success 8.72·tilt~0·yaw 막 시작).
> → **P35** (P34 **resume** +5000, [0806]): ramp 교과서적 완벽(발동→full 2994iter 완만)! **but yaw 안 잡힘**
>   (30.8°, P34보다 악화). ⭐⭐진단: yaw full 가는데 track 6.86→0.81 붕괴해도 **success 9.48 멀쩡**·precise 0.66 →
>   **success(지배 sustained)가 yaw-BLIND**(near&upright&grasp만) → 정책이 success만 파밍, yaw-penalized track 버림.
>   A'(yaw를 부차 보상 tracking에)는 원천 무력. **position↔yaw 이분법**: P33=yaw만/P35=position만, 둘다 X.
> → **P36** (yaw를 **success에 soft 곱셈**, [0806b]): 지배 보상에 yaw 앵커. **but inconclusive** — seed42가 또
>   discovery 늦어(it9490) yaw scale 0.06까지만, yaw 31°(안 켜짐). P34/P35/P36은 yaw 발동 전까진 동일 rollout.
> → **P37** (접촉 타이밍 대칭, [0806c]): yaw를 reward로 되돌리기(P31~36) 대신 **origin 차단**.
>   play 관찰(양팔이 하나씩 닿아 yaw 튐) → `plate_approach_symmetry`=근접 시 |d_L−d_R| 페널티(동시 터치다운
>   유도). eval: **yaw 31.0°→24.1°**, yaw-aligned 0.059→0.103, drop 0. transient 원인은 줄였지만
>   grip-axis도 약 24° off라 kinematic placement 잔차 확인. precise 0.616/upright 0.645로 소폭 비용.
> → **P38 설계** ([0808]): P37 cause 차단 유지 + success에 **soft yaw outcome** 결합.
>   `yaw_factor=1−0.5·s·tanh(yaw_err/0.35)`, success EMA≥3에서 발동 후 3000 iter ramp(`s:0→1`).
>   full에서도 factor≥0.5라 success 50% floor 보존. P37의 이른 discovery로 P36의 ramp 미작동 문제 해소 기대.
> 실물 후보 checkpoint = **P34 model_9999(precise 0.70·upright 0.78) / P33(yaw 3.1°·precise↓) / P29(precise 0.71)**.
> 교훈 모음: ① 제자리에서 딸 수 있는 shaping + 움직임 페널티 = 정지 국소해 (P3) ② mean 형태의 양팔 품질
> 보상은 "한쪽 포기" 허용 — min 강제 (P4) ③ 새 페널티 항은 −1e-4 시작 + 성능 커리큘럼 (P7) ④ 이진 문턱은
> 탐색 도달 범위 내 (P5/P13) ⑤ 발견 문제는 상태 기반 dense 사다리로 (P14 lift_progress)
> ⑥ 힘 기반 reward/obs는 clamp+nan_to_num 필수 (P11 크래시) ⑦ 정렬은 reward가 아니라 gate여야 (P16)
> ⑧ 파지 품질을 task 보상에 곱셈으로 섞으면 품질↔성공 트레이드오프 → 품질은 gate로 분리 (P20~P22, cube 비교)
> ⑨ 정렬 목표를 **물체 로컬 프레임**서 측정하면 그 물체의 world 회전에 불변 → world/base ori 목표는 반드시
> world/base 기준 측정 (P30 centrality가 box-local이라 yaw 못 잡음 — 로컬 게이트는 "어디를 잡나"용)
> ⑩ 목표가 grip 기하와 **양립**하면(box_upright: world-up, 중력 도움) 약한 페널티로 되지만, grip placement와
> **싸우면**(yaw: 팔이 grip을 35°로 놓음) 같은 페널티도 무효 → 지배보상에 **곱셈** 앵커라야 스케일36을 이김
> (P31 A 실패 vs P32 A'). 논문(2310.03191)은 box goal-pose 추종으로 yaw 앵커 — 우리 tracking은 position-only가 구멍
> ⑪ 붕괴 위험 큰 ramp(외란·q'·yaw)를 **여러 개 같은 스케줄로 겹치면** 취약 마일스톤을 죽임 → 새 압력은
> 기존 압력이 안정된(EMA로 확인) 뒤 **순차 발동**(lo/hi를 뒤로) (P32 yaw 조기발동 success 사망 → P33 lo3/hi6)
> ⑫ success EMA는 급등 가능 → EMA(값)기반 ramp는 "순간 full"이 되어 붕괴 유발 → ramp **속도**가 중요하면
> iteration(시간)으로 고정하라. 발동 "조건"은 성능, "속도"는 시간 (P33 yaw 13iter 급발진 → P34 iteration ramp)
> ⑬ 정렬을 곱셈 앵커해도 **지배 보상이 그 축을 안 보면 무력**(정책이 지배항만 파밍) → 정렬 요구는 반드시
> **지배 보상 자체**에 넣어야(부차 보상=무시) (P35 yaw in tracking 무력, success yaw-blind → P36 yaw in success)
> ⑭ 나쁜 자세를 reward로 **되돌리기(outcome)**보다 그게 **생기는 순간을 막기(origin/cause)**가 근본 — 원인이
> 물리적으로 명확하면(비동시 접촉→yaw) 그쪽이 깔끔 (P31~36 yaw-reward 난항 → P37 접촉 대칭으로 origin 차단)
> 자매 문서: `DUAL_FR3_LIFT_ENV_ANALYSIS.md` (오른팔 cube lift — gating/curriculum 교훈의 출처)

---

## 1. Task Description

- **작업**: 바닥(z=0) 위 박스를 **그리퍼 없이 양팔의 판(plate) EE로 조여(squeeze)** 마찰력만으로
  들어올려 목표 위치로 운반. v0는 30~50cm 범용 박스 연구, **현재 V1/V2는 555×475×275mm 실물 박스**
- **로봇**: Husky 베이스 + FR3 양팔 + **140×140×25mm 판 EE** (`DUAL_FR3_PLATE_EE_CFG`, USD `dual_fr3_plate_ee.usd`). 그리퍼 없음 → **양팔 14 joint 전부 제어**
- **EE 정의**: `left/right_fr3_link7` + 판 작업면 오프셋 `PLATE_TCP_OFFSET_POS = (0,0,0.132)` (플랜지 0.107 + 판 두께 0.025). 판 collider는 link7 body 소속 (자기 링크와 비충돌, 외부 물체와만 접촉)
- **파지 = 힘 제어**: 그리퍼가 없으므로 "잡기"는 **양 판의 법선력 × 마찰**로만 성립.
  힘은 명시적 action이 아니라 **PD 위치 오차(침투 깊이)로 암묵 제어** — 임피던스 제어와 동일 원리.
  rule-based 데모(`scripts/tutorials/05_controllers/run_dual_fr3_plate_lift.py`)로 2~3cm 침투 명령이 2kg 박스를 드는 것 검증 완료
- **에피소드**: 5.0 s × 50 Hz 제어 (sim 100 Hz, decimation 2) = **250 step**

---

## 2. MDP 구조 (2026-07-27 기준)

### 2-1. Observation (policy, v0 = 총 58 dims / **v1·v2 = 56 dims (plate_forces 제외)**, history 없음)

| Term | 내용 | Dims | 비고 |
|------|------|------|------|
| `joint_pos` | 양팔 14 joint, default 대비 상대값 | 14 | |
| `joint_vel` | 위와 동일 | 14 | |
| `object_position` | 박스 위치, robot root frame | 3 | |
| `target_object_position` | goal pose command (pos 3 + quat 4) | 7 | quat 4 dims는 identity → dead dims (cube task와 동일) |
| `actions` | 직전 action | 14 | |
| `plate_forces` | **양 판 접촉력 크기 (L, R)** × scale 0.02 | 2 | **v0 only**. V1/V2 actor에서는 제거 — §3-2 |
| `object_orientation` | 박스 quat (root frame, w≥0 고정), noise σ0.02 | 4 | P5 추가 — yaw DR 관측성. 실물은 ArUco 6DoF (§3-2 유사 논리) |

**obs noise (v3, 2026-07-27 추가)**: joint_pos σ0.005 rad / joint_vel σ0.05 / object_position σ0.01 m.
v0의 plate_forces에만 σ0.04(2N 상당)를 사용하며 V1/V2 actor에는 해당 term 자체가 없다. Orientation은
quaternion component에 σ0.02를 직접 더하는 현재 구현이라, 실물 전이 branch에서는 SO(3) noise+normalize로 바꿀 후보
(noise 단독·물성 DR 단독은 drop 0, "조합"만 슬립 드랍 발생하는 상호작용 확인).

### 2-2. Action (14 dims) — **delta position**

| | 구현 | Scale |
|---|---|---|
| arms (14) | policy tick에서 `delta=0.1·a`; 각 100 Hz physics substep에서 `target=q_current+delta`로 재계산 | 0.1 |

- **delta 채택** (2026-07-27 합의). 근거:
  1. contact-rich task에서 current-q 기준 delta가 조임 방향을 계속 제공 (공식 dexsuite와 동일 설정)
  2. (원래 cube가 쓰던) 절대+offset은 ready ±0.5rad hard-bound라 **바닥 박스 접촉 자세에 도달 못 할 위험**
  3. **주의:** 0.1은 rate limit이 아니다. policy 50 Hz에서 target을 1회 만들지 않고, 같은 delta를 두
     physics substep 각각의 최신 q에 다시 더한다. 실물 action adapter 계약은
     [`DUAL_FR3_LIFT_PLATE_SIM2REAL.md`](DUAL_FR3_LIFT_PLATE_SIM2REAL.md) §1-3을 기준으로 한다.
  - ※ cube task도 이후 Run C 3차(2026-07-28)에 절대+offset → delta(0.1)로 전환해 **양 task delta 통일**
    (근거: "절대 action × obs noise" 조합에서 noise_std 폭주 — cube 문서 참조). 지금은 두 task 모두 delta.
- 부작용: `action_rate` 페널티가 delta 변화율을 제한하는 성격으로 변하며, 실제 이동 속도는 PD·관성·effort
  limit에 좌우된다. Drift는 obs의 joint_pos로 policy가 보정한다.

### 2-3. Sensors (squeeze 판정 + v0 obs)

```python
# 판별로 분리 (PhysX 제약: filter는 sensor body당 1:1 — cube task에서 확인된 것과 동일)
left_plate_object_contact  = ContactSensorCfg(prim_path=".../left_fr3_link7",
                                              filter_prim_paths_expr=[".../Object"])
right_plate_object_contact = ContactSensorCfg(prim_path=".../right_fr3_link7",
                                              filter_prim_paths_expr=[".../Object"])
```

- 판 collider가 link7 body에 속하므로 link7 센서가 판-박스 접촉력을 그대로 보고
- `filter_prim_paths_expr`로 **박스와의 접촉력만** 측정 (바닥·자기충돌 제외)
- 모든 버전의 reward gate/품질 계산에 사용. actor obs(`plate_forces`)로는 **v0에서만** 사용

### 2-4. Reward

```
G      = 𝟙[min(‖F_left‖, ‖F_right‖) > 1.0 N]        # squeeze gate: 양 판이 동시에 박스 접촉
C_L/R  = 𝟙[‖F_left/right,Object‖ > 1.0 N]
X      = C_L ⊕ C_R                                      # P39: actual one-sided-contact indicator
L      = 𝟙[z_obj > half_env + 0.10]                  # 높이 gate — env별. P5의 +0.20은 발견 실패(무보상 절벽) → P6에서 +0.10 원복
P      = (1−tanh(d_L/0.15))·(1−tanh(d_R/0.15))       # [P11부터 미사용] 근접 factor — v5~P10의 품질 항 게이트

r_total = 6.0  · min_i[(1 − tanh(d_i/0.3)) · align_i]      # reaching_aligned [07302019] 접근×정렬 통합 (덧셈 3항→곱셈 1항)
          align_i = clamp(∓n_i·ŷ_box, 0, 1)              # 판 법선 vs 면 법선 반평행도. 곱셈=가깝고AND평행만 보상, std0.3 넓어 파지 중 생존
          # 이전(P16까지): reaching(w2,mean→min)+reaching_fine(w3,std0.08)+plate_alignment(w6) 3항 덧셈
          # → 덧셈이라 정렬만 허공서 따먹음(P16 스크린샷). 곱셈 통합으로 동시 달성 강제. reaching_fine 제거(커리큘럼 카드)
        − 4.0  · G · |Δ_pair − Δ_ideal|_{x,z}             # plates_misalignment (P6 yaw-aware, P11: 게이트 P→G)
q      = min_i[ |F_i·n̂_face|/|F_i| ]                 # [07302119] 접촉 품질: 면-면 접촉=1, 모서리 걸침=낮음
q      = q · (1 − s + s·align)   (v2 전용, [08032])   # q': 판 자세 blend. align=min_i[판법선·면법선],
        # s=_qprime_scale (success EMA 1→3 구간서 0→1 ramp). s=1이면 q·align — flush 파지만 만점.
        # v0/v1은 s=0 고정(무영향). 근거: 곱셈 신호만 생존 (P20~P24), q는 힘 방향만 봐서 기울인 세트 통과
        + 2.0  · G · q                                    # squeezing ([07302119] q 곱: 잘 붙을수록 더)
        + 8.0  · G · clamp((z_box−half_z)/0.10, 0, 1)     # lift_progress (P14, q 미적용=발견 안전밸브)
        + 15.0 · L · G · q                                # lifting        ([07302119] ×q: 면-면 파지 유도)
        + 16.0 · L · G · q · (1 − tanh(d_obj-goal / 0.3))  # goal_tracking  ([07302119] ×q)
        +  5.0 · L · G · q · (1 − tanh(d_obj-goal / 0.05)) # fine_grained   ([07302119] ×q)
        − 2.0  · (1 − ẑ_box·ẑ_world)   (curriculum 발동 후 −3.0)   # box_upright (P8: 상시 −3은 발견 차단 — P7 실패)
        − 1e-4 · Σ_i relu(‖F_i‖ − 100N) (curriculum 발동 후 −0.05)  # excessive_squeeze (P9: 60→100N — 60은 정당 파지 힘 norm(~58N 정적, ~87N 과도)보다 낮아 쐐기 파지 유발)
S       = 𝟙[goal<10cm ∧ tilt<15° ∧ G, 0.5s 연속 후 지속]
Y       = 1 − 0.5·s_yaw·tanh(yaw_err/0.35)                 # P38: base-yaw soft factor, Y≥0.5
        + 20.0 · S · Y                                     # v2 sustained success; s_yaw는 EMA≥3 후 3000 iter ramp
        − 1e-4 · N_{link[1-6] 접촉}  (curriculum 발동 후 −1.0)   # undesired_contact_penalty (v3)
        − 1e-4 · ‖a_t − a_{t−1}‖²   (curriculum 발동 후 −0.1)   # action_rate
        − 1e-4 · ‖q̇‖²              (curriculum 발동 후 −0.1)   # joint_vel
        − 1e-4 · 𝟙[min(d_L,d_R)<0.05]·|d_L−d_R| (발동 후 −20) # P37~P39 distance approach symmetry
        + 0.0 · X                    (P39만, 발동 후 −5.0·X)   # full-episode actual unilateral duration
```

- **항별 현황 (P39 설계 기준)**:
  - `reaching_object`(w6, 커리큘럼으로 **6→9** [07311b]): `reaching_aligned` = min_i[(1−tanh(d_i/0.3))·align_i].
    양 판→박스 **양 측면 중심점**(yaw·env별 반너비 반영)까지 거리 × 정렬(판 법선 vs 면 법선)의 **곱셈 통합**.
    P17에서 reaching(w2)+reaching_fine(w3)+plate_alignment(w6) 3항 덧셈을 이 1항으로 통합 (덧셈=허공서 정렬만
    따먹음 → 곱셈=가깝고 AND 평행만 보상). **둘 다 min** (P4 한쪽 포기 방지). env별 반너비 `_box_halves` 캐싱.
    ⚠ **align_i = clamp(∓n̂_i·ŷ_box, 0, 1) = raw cos** → 평행 근처 기울기 평평 → 마지막 몇 도 짜낼 유인 약함
    (오른팔 "덜 평행" 관찰의 원인). 커리큘럼 6→9는 P19의 6→12 과잉지배(success 붕괴) 완화값.
  - `q` = `_contact_normal_ratio` = min_i[|F_i·n̂_face|/|F_i|]: 접촉힘의 면-법선 정렬도 (면-면=1, 모서리=낮음).
    **squeezing/lifting/goal_tracking/fine_grained에 곱셈**. ⚠ q는 접촉힘 **방향**만 보고 판 **자세**(flush)와
    **위치(중심성)는 못 봄** → 판이 기울어 모서리로 닿아도 힘이 면수직이면 만점 (P24 "세트째 기울임"의 원인).
    **[08032] q' 확장 (v2 전용)**: `q_eff = q·(1−s+s·align)`, align=min(양판 법선 vs 면 법선), s=`_qprime_scale`
    (success EMA lo1~hi3 soft ramp, `ramp_qprime_on_success`). s=1이면 q·align → 지배 보상(36)이 flush를 요구.
    v1/v0은 s=0 고정이라 무영향. 근거: 곱셈 신호는 생존·덧셈 신호는 침식 (P20~P24 실증) — flush를 곱셈 자리에.
  - **weight 지형 요약 (08-03 논의)**: 실효(/s)로 보면 사다리 1.6~11.4, reaching 4.0, 페널티 합 −3.3,
    success 4.4(1500은 1회성 착시, 만점 6/s). 사다리 내부 격차 = 단조증가 필수(발견), 페널티 −1e-4 시작 = P7.
    **유일한 진짜 불균형 = 품질(1~9) vs 사다리(36)** — weight 재조정으로는 균형점 없음(2/9/12 데이터),
    해법은 곱셈(q')로 품질이 사다리 스케일을 빌리는 것. 부수 발견: fine_grained 실효 0.4/s 거의 사망(정리 후보),
    기하 페널티 3종(symmetry/misalignment/upright) 역할 중복(통합 후보).
  - `plates_misalignment`(w−4, G-gate): 이상 쌍 벡터 대비 편차 = **위치** 정렬 (박스 회전 방지, P2).
  - `squeezing`(w2): `2·G·q`. reaching→lifting 사이 "양팔 접촉 품질" 사다리 (P18부터 ×q)
  - `lift_progress`(w8, P14): `8·G·clamp((z−half)/0.10,0,1)`. q **미적용**(발견 안전밸브)
  - `success`: goal 10cm + 수평(tilt<15°) + G. **[08041] v2는 per-step 지속(weight 20, sustained=True)**:
    0.5s 진입(hold_steps=25) 후 zone 유지하는 매 스텝 지급(~20/s, dense lifting 15/s 우위) → "일찍 정렬·오래
    유지"가 최대 이득 = 정렬 유지가 최고 reward로 역전. **v0/v1은 기존 1회 latch(weight 1500, ÷6=성공률)**.
    배경: P27/P28 게이밍(평소 모서리, 끝 0.5s만 정렬해 1회 훔침) — 중앙 정렬 요구하는 유일 항이 1회성이라
    모서리 파지 방치됐음. termination 없음(out_of_bounds만)이 "오래 유지=최대이득"을 성립시킴.
    ⚠ v2에선 `Episode_Reward/success`가 "지속 유지량"이라 **÷6=성공률 해석 무효** → 진짜 성공률은 eval로.
    **[P38] base-yaw soft factor 추가**: `1−0.5·s_yaw·tanh(yaw_err/0.35)`. success EMA≥3에서 yaw ramp를
    시작해 3000 iter 동안 `s_yaw:0→1`; 최악 yaw에서도 factor≥0.5라 위치·수평·파지 success의 절반은 보존.
    P37의 접촉 동시성은 yaw 발생 원인(cause), 이 항은 남은 grip placement 결과(outcome)를 담당한다.
  - **[P39] `one_sided_contact`**: Object-filtered `C_L XOR C_R`. 별도 timestamp 없이 50Hz 매 step
    적분해 최초 contact lag와 운반 중 unilateral loss를 모두 비용화한다. 초기 weight 0.0이라 lifting
    EMA>10 전에는 P38 reward와 정확히 같고, 이후 -5.0으로 latch. P37의 거리 기반
    `plate_approach_symmetry`는 제거하지 않는다.
  - gate threshold **1.0 N** (판 면접촉이라 스침 오탐 방지) — ⚠ 1N은 달성 너무 쉬워 "접촉 여부"만 판정,
    "제대로 잡았나"는 판정 못 함 (사용자 지적, 품질 gate 확장 후보)
- **G를 lifting/tracking 전부에 곱는 이유**: cube §3 실증 exploit 방어 — 높이 gate만으로는 "팔뚝에
  얹기/밀어올리기"가 이득. **이 G가 정렬 gate 확장의 자리** (평행 조건을 여기 넣으면 비스듬 파지 = 전 보상 0)

### 2-5. Termination / Curriculum / DR

| 항목 | 내용 |
|---|---|
| time_out | 5 s. success reward는 P7 이후 존재하고 P29/P38에서는 sustained 구조; **success termination은 없음** |
| object_dropping | **제거 (None)** — 바닥 위 박스는 낙하 개념이 없음 |
| **object_out_of_bounds** | **P11 신설 (NaN 가드)**: 박스 중심이 x ±0.85 / y ±0.84 / z [−0.05, 1.66] 벗어나면 종료. 경계 = `compute_dual_fr3_workspace.py` 몬테카를로 FK 실측(양팔 pinch 영역 + 0.1). 박스 속도 상한(10 m/s, 30 rad/s)과 2층 방어 — P11 1차 크래시(사출→obs 오염→value inf→NaN) 대응 |
| curriculum | `modify_reward_weight_on_lifting_learned` 재사용 (배치 크기 가중 EMA, α=min(0.5, 3·batch/N), term별 분리). lifting>10/s 발동: action_rate/joint_vel −1e-4→−0.1, contact −1e-4→−1.0, box_upright −2→−3, excessive_squeeze −1e-4→−0.05, reaching_object 6→9, **P39 one_sided_contact 0→−5** |
| DR: 박스 크기 | **[08033 버그픽스로 해결]** ~~mode="usd"가 무효~~ → `prestartup`. 역사적으로 v0~P24의 크기 DR은 미작동. V1은 실물 박스 ±15mm 설정을 가질 수 있지만 **현재 P38/V2는 scale=(1,1,1)로 명시적으로 OFF** |
| DR: 박스 질량 | v0는 density×0.6~1.4. **V1/V2는 nominal 4.0kg에 absolute 3.5~4.5kg** (`randomize_rigid_body_mass`, startup, 관성 재계산) |
| DR: 박스 스폰 | x −0.05~+0.10, y ±0.10, current cfg yaw ±0.1 rad. 과거 checkpoint는 저장 당시 env cfg를 기준으로 평가 |
| DR: 박스 마찰 | **static 0.4~1.2 / dynamic 0.3~1.0** (v3에서 cube Run C와 같은 폭으로 확대. startup, 64 buckets, make_consistent). 판은 USD에 재질 명시(0.6/0.55) → 유효 static(average) ≈ 0.5~0.9 |
| DR: joint 물성 | **액추에이터 PD 게인 ±15% scale** (v3, `randomize_actuator_gains`, startup, 양팔 14 joint) |
| DR: goal | **P13: 순수 수직 리프트** — goal x/y = 박스 스폰 위치 추종(`VerticalLiftPoseCommand`), z만 0.15~0.35 DR (world 0.555~0.755). 실물 "제자리 리프트" 정합. (P12까지는 x 0.45~0.60/y ±0.10 독립 샘플) |
| 시각화 | DR 존 마커: 파란 판=박스 스폰 범위(바닥), 초록 판 2장=goal 범위 상/하한 (시각 전용, cube task와 동일 방식. `DR_VIS=0`으로 끄기 가능) |
| PhysX | `gpu_total_aggregate_pairs_capacity` 64×1024 (P6 — cube의 4096 env 동시 접촉 용량 초과 이슈 예방 이식) |
| **없는 DR** | obs latency (검토 보류) |

---

### 2-6. 태스크 변형 (2026-07-29)

| gym id | 박스 | obs | 용도 |
|---|---|---|---|
| `...Plate-v0` (+Play/Play-Noisy) | 정육면체 30~50cm, 0.5~5.3kg (블라인드 범용) | 58 | 범용 정책 연구 |
| `...Plate-v1` (+Play-v1/Play-Noisy-v1) | **실물 박스** 555×475(잡는방향)×275mm ±15mm(크기 DR [08033]부터 실작동 — 실측 검증, 그 이전 run은 고정 크기), 질량 4.0±0.5kg(✅)·마찰 0.4~1.2(✅) | **56** (plate_forces 제외 → τ_ext 불필요) | **실물 1차 검증용** |
| `...Plate-v2` (+Play-v2/Play-Noisy-v2) | **실물 박스 크기 고정**, 3.5~4.5kg·마찰 DR + success-기반 외란·q′·대칭·sustained success/P38 yaw factor | **56** (force-free actor) | **현재 P38 및 sim2real 기준** |
| `Isaac-Lift-Box-DualFR3-Plate-P39-v0` (+`Isaac-Lift-Box-DualFR3-Plate-P39-Play-v0` / `Isaac-Lift-Box-DualFR3-Plate-P39-Play-Noisy-v0`) | Plate-v2/P38 물리·reward 전체 + actual one-sided Object-contact XOR `0→−5`만 추가 | **56** (동일) | **현재 P39 controlled A/B; 별도 log, scratch** |
| `...Plate-ZeroShot-v1` | v1 박스 + v0 obs 58 | 58 | v0 정책의 v1 박스 zero-shot 평가 |

- v1 로그는 `logs/rsl_rl/dual_fr3_plate_lift_v1/` 분리. "안다"의 구현 = 좁은 DR (단일 종이라
  크기/질량 obs는 상수 = 정보량 0). 이후 카드: v1-B = 넓은 범위 + 크기/질량 obs conditioning
  (humanoid 2310.03191 방식 — 여러 박스 단일 정책 + 외삽 22.9kg 실증)

## 3. 설계 논거

### 3-0. 정렬(alignment)의 두 층위 + 정렬 항의 실패사 (P16 갱신, 2026-07-30)

우리 task의 "판이 박스와 잘 맞물림"은 서로 다른 두 실패를 막아야 성립:

| 실패 모드 | 막는 항 | 재는 것 | 시점 |
|---|---|---|---|
| 판이 면에 비스듬 (모서리로 긁음) | **plate_alignment** (자세) | 판 법선 vs 면 법선 반평행도 | 접촉 전~후 |
| 두 판이 대각/어긋나게 잡아 박스 회전 | **misalignment** (위치) | 이상 쌍 벡터 대비 편차 | 접촉 후 (G-gate) |

- **misalignment는 위치 그대로 유지**: 그 "위치 배치"는 정렬용이 아니라 박스 회전 방지(P2 목적).
  법선(자세)으로 대체하면 고유 역할을 잃음. 자세 정렬은 plate_alignment 담당.

**⭐ plate_alignment(자세 정렬) 항의 실패사 — reward로는 구조적으로 못 잡음 (P11~P16, 6번):**

| 시도 | 게이트 | weight | 결과 | 원인 |
|---|---|---|---|---|
| P11 | 접촉 G | 2 | 붕괴 | 접촉 후 grip 굳어 무시 |
| P12 | G × 수평 U | 2 | 어중간 | 곱셈 vanishing gradient |
| P15 A안 | 거리(reaching_fine에 곱) | 3 | 붕괴 | 침투 파지 중 거리 kernel 0 → 정렬도 사망 |
| P15-v2 | 거리 0.15 (별도 항) | 2 | 붕괴 | 정렬 안 해도 success 75% → w2 무시 (경제학) |
| P16 | **없음** | **6** | **P3 재발** | 허공에서 판만 평행 유지해 6점 따먹고 접근 포기 |

- **딜레마 확정**: 게이트 있으면 (접촉 후라) 무시, 게이트 없으면 P3 허공 따먹기. reward weight/게이트
  조합으로는 못 벗어남. 근본 원인: **정렬은 sim success에 기여 안 함**(비스듬해도 마찰로 들림) →
  reward로 주는 한 무시되거나 exploit됨. 실물에선 비스듬 접촉=미끄러짐이라 필수인데 sim-실물 괴리.
- **다음 = 정렬을 보상 아닌 gate/제약으로**: squeeze gate G에 "판이 면에 평행할 때만 접촉 인정"을
  추가 → 비스듬 파지는 lifting/tracking/success **전부 0** (따먹을 점수 아니라 통과 조건).
  P3(접촉 필요) 불가 + 무시(정렬 안 하면 lifting 0) 불가. 주의: 너무 빡빡하면 P7류 발견 차단 →
  커리큘럼 점진 도입 또는 느슨한 임계 시작.

> **⭐ 그 이후 경과 (P17~P22, 위 표는 P16까지의 역사 기록):**
> - **P17**: `plate_alignment`(별도 항)를 없애고 `reaching_object`에 **곱셈 통합** →
>   `reaching_aligned = w·min_i[(1−tanh(d_i/0.3))·align_i]`. "가깝고 AND 평행"만 보상해 허공 따먹기 차단.
>   즉 위 표의 `plate_alignment`/`reaching_fine`은 **현재 코드엔 없음**(None), align은 reaching에 흡수됨.
> - **P18**: 접촉 **품질 q**(법선비율)를 squeezing/lifting/tracking에 곱셈 도입 (모서리 걸침 억제).
> - **P19~P22**: reaching 커리큘럼(6→12 과함 → 9). 긴 학습이 파지 품질 침식(reach kernel 0.33→0.03).
>   → **여전히 "품질을 곱셈으로 섞음"이 트레이드오프의 뿌리.** 위 "gate/제약으로" 방향이 아직 미완 —
>   q를 곱셈에서 빼고 **G gate에 평행/중심 조건 통합**이 다음 후보 (cube 비교가 뒷받침, §6).

### 3-1. Squeeze는 왜 위치 명령으로 되는가 (rule-based 데모의 교훈)

- PD 게인 하에서 **접촉에 막힌 위치 명령의 잔여 오차 = 법선력**. 데모에서 침투 2cm 명령 시
  잔여 오차가 정확히 침투량만큼 남으며(err 0.021) 박스가 유지됨을 확인
- **결정적 함정 (데모에서 실증)**: 이동 중 침투 명령이 풀리면 즉시 미끄러짐.
  데모의 보간이 시작점을 "현재 TCP(면에 막힌 위치)"로 잡는 바람에 lift 시작 순간 조임 명령이
  0으로 리셋 → 판이 박스 면을 타고 허공으로 (1차 시도 실패 원인). `start_b`로 조임 명령을
  고정해 해결. **RL에서도 같은 원리**: policy는 상승 중에도 조임 방향 delta를 계속 출력해야
  하며, 이를 실패 시 즉각적인 보상 손실(G=0 → 초당 ~35 손실)로 학습하게 됨
- delta action은 이 "계속 밀기"와 궁합이 좋음: 접촉에 막힌 상태에서 조임 방향 성분을
  유지하면 자연스럽게 침투 명령이 유지됨

### 3-2. plate_forces 관측의 역사와 V1/V2 force-free 결정

- **v0 역사**: squeeze가 task 본질이므로 양 plate contact-force magnitude 2D를 actor에 넣었다. 실물 대응
  후보는 FR3 관절 토크 기반 `tau_ext_hat_filtered`/`O_F_ext_hat_K`였고, payload·bias·drift calibration이
  배포 전제였다.
- **현재 결정**: 실물 고정 박스를 대상으로 한 V1이 `plate_forces`를 제거했고 V2/P38이 이를 상속한다.
  따라서 현재 actor는 56D이며 `tau_ext`나 sim contact truth에 의존하지 않는다. Delta action, joint state,
  object motion만으로 contact와 slip에 반응한다.
- Sim contact force는 여전히 reward gate, q/q′ 품질, excessive-squeeze 계산에 쓰이는 **privileged training
  signal**이다. 향후 asymmetric critic에도 줄 수 있지만 actor에는 되돌리지 않는다.
- `tau_ext`는 actor 입력과 별개로 실물 collision/force watchdog, contact calibration, 진단 로그에 사용한다.
  Sim의 100N excessive-squeeze threshold를 real `tau_ext` threshold와 직접 등치하지 않는다.
- 상세한 현재 배포 계약과 calibration 계획은 `DUAL_FR3_LIFT_PLATE_SIM2REAL.md` §1·§4를 기준으로 한다.

### 3-3. lifting 판정 높이 0.35의 트레이드오프 (→ v3에서 env별 판정으로 해결)

> **2026-07-27 해결**: `_box_half_sizes`(xformOp:scale 캐싱)로 env별 안착 중심을 알 수 있게
> 되어, 판정을 `z > half_env + Δ`로 교체 (`object_is_lifted_perenv_and_squeezed` 등).
> **Δ 이력**: P5에서 0.20(tipping 차단) → **발견 실패로 P6에서 0.10 원복** (tipping은 발견 디딤돌로
> 허용, 기울임 정제는 box_upright/misalignment/min-alignment 페널티가 담당). 아래는 v1 시절 논의 기록.

- 고정값 0.35는 **최대 박스(안착 중심 0.25) 기준 +10cm**. 큐브가 모서리로 구르는 순간의
  중심 상승(최대 0.25·√2≈0.35)과 겹치지만 G(양판 접촉) 동시 조건이 오탐을 차단
- 대신 **최소 박스(중심 0.15)는 +20cm 들어야** 판정 통과 — 작은 박스일수록 과제가 어려움.
  per-env 판정 높이(스폰 중심 + Δ)가 정석이나, scale DR 값을 reward 함수에서 참조하는
  배관이 필요해 v1은 고정값. 학습이 작은 박스에서만 정체되면 여기부터 의심할 것

---

### 3-4. Reference 문헌과의 보상 구조 대조 (2026-07-28)

가장 유사한 태스크의 보상 설계와 대조해 v6 변경의 근거로 삼음:
**"Sim-to-Real Learning for Humanoid Box Loco-Manipulation" (arXiv:2310.03191)** — 휴머노이드 양팔 박스 들기.

| 그쪽 설계 | 우리 대응 | 상태 |
|---|---|---|
| 양손 **각각** 박스 측면 목표점 거리 (합산, w0.15) | reaching (P17부터 min, align과 곱셈 통합) | 반영 |
| 손 orientation 정렬 페널티 (roll, w0.05) | align_i (P17부터 reaching_aligned에 흡수) | 별도 plate_alignment는 제거됨 |
| 접촉 타이밍 gate (조기 접촉 방지 indicator) | 근접 게이팅 (v5) | 반영 |
| **박스 roll+pitch 수평 페널티** (w0.05) | box_upright | **v6 신설** |
| 손-박스 접촉력 크기 페널티 (부드러운 접촉, w0.05) | 없음 | 보류 카드 (sim 센서만으로 가능, F/T 불필요) |

**추가 대조 (2026-07-28, 본문 정독):**

- **arXiv:2409.15465 (In-the-Wild Ungraspable Picking)** — RL이 아닌 계획 기반 (EE 측면으로 끼우는
  bimanual bracing + 손목 6축 F/T로 hybrid 위치/힘 제어). 배울 점:
  ① 파지 품질 = 최악 외란 wrench에 대한 force closure QP (마찰콘 조건) — 우리 보류 카드
  "gate 접촉력 법선비율"의 문헌 근거 ② **"상호작용 중엔 vision 불신, 고유수용+힘만 사용"**
  — 우리 가림 대응(EKF+기구학 prior, 손감각 운반)과 동일 철학, 독립 검증
- **arXiv:2307.06423 (Bi-Touch)** — 양팔 촉각 RL sim2real (TacTip 촉각 이미지 obs, GAN으로
  real→sim 변환, 물체 pose obs 없음 — 평면 4-DoF 과제라 가능). 배울 점:
  ① reward에 "TCP 표면 수직" 정렬 항 = 우리 plate_alignment와 수렴 ② **실물 실패 사례
  "unwanted squeezing"**: sim에선 무해하던 과조임이 실물에서만 발현 → 과접촉 깊이 페널티
  강화 + 강성 튜닝 + "10-step 자세 유지 후 성공 인정"으로 해결. **→ 우리 과조임(접촉력
  크기) 페널티를 "보류"에서 "실물 배포 전 필수"로 격상하는 직접 근거** + success 정의(§5-5)에
  N-step 유지 조건 채택 근거 ③ obs를 저차원(스칼라 힘+pose)으로 유지한 우리 선택은
  촉각 이미지 GAN 변환 같은 파이프라인을 불필요하게 만듦 (sim2real 단순화)

### 3-5. 양팔 비대칭과 min-form (P4 교훈)

- dual_fr3는 거울 대칭이 아님 (장착 yaw ∓30° 보상해도 FR3 관절 리밋은 비대칭) → 같은 정렬
  자세라도 팔마다 비용이 다름. 전통 제어라면 팔별 설계가 필요한 지점 — RL이 팔별로 다른
  관절 전략을 찾게 하는 것이 이 프로젝트의 취지
- 단, 보상이 mean이면 RL은 "비용 큰 팔은 포기"를 선택함 (P4에서 실증: 왼판 정렬 완벽,
  오른판 기울임, alignment tail 0.078). **min-form은 "다른 방식으로라도 둘 다 달성"을 강제**
  — 비대칭 해법은 허용하되 목표 미달성은 불허
- **재발 이력 — 이 비대칭은 압력이 세질 때마다 오른팔부터 무너지는 형태로 반복됨**:
  P8(60N 벽) 쐐기 파지의 희생양도 오른판, P10(v1 납작 박스)에서도 "접근 땐 평행, lift 후
  오른판 비틀림 + 오른팔 먼저→왼팔 순서 습관" 관찰 (2026-07-29 play, alignment 0.034).
  메커니즘: ① lift 후 작업 영역에서 오른팔 정렬 자세가 리밋 근처의 비싼 자세
  ② **prox 게이팅의 gradient 소멸 함정** — 오른판이 면 중심을 벗어나면 prox→0이라
  "고치라는 신호"도 0 (min이라 왼판 보상도 0으로 함께 소멸). 왼판이 평행을 유지하는 건
  alignment가 아니라 misalignment 페널티 + reaching_fine이 붙잡아서임.
  **→ P11에서 처방 확정 (2026-07-29): alignment/misalignment의 게이트를 근접(P)→접촉(G)으로
  교체.** 근접 게이트는 판이 중심을 벗어나면 방향 gradient까지 꺼지는 함정 (기록 검증 결과
  v5 당시 근접/접촉을 비교 없이 선택한 미문서화 결정이었음). 접촉 게이트는 P3 함정을 동일하게
  막으면서 "잡는 동안"은 위치 무관하게 방향 압력 유지. 예비 카드: reaching_fine mean→min
  (P11로도 위치 이탈이 남으면 — 중심 이탈 자체를 차단하는 보완책)

## 4. 남은 구조적 약점 (심각도순)

1. ~~Success 개념 부재~~ → **P7 이후 reward 도입, P29/P38 sustained success로 발전**. 남은 것은
   0.5s hold+yaw/tilt/drop을 한 번에 재는 고정-seed combined evaluation protocol
2. ~~작은 박스 불리한 고정 lifting 높이~~ → **2026-07-27 반영 (v3)** — env별 판정(안착 중심+0.10)
3. ~~박스 yaw DR 없음~~ → **2026-07-27 반영**, P30 이후 current cfg는 yaw ±0.1 rad
4. ~~obs noise 부재~~ → **2026-07-27 반영 (v3)**. 남은 것: pose/action latency·sample-hold·dropout,
   persistent calibration bias, quaternion noise 모델. `tau_ext`는 V1/V2 actor gap이 아니라 safety/calibration 신호
5. ~~마찰 DR 부재~~ → **2026-07-27 반영** (박스 static 0.5~1.0 DR + 판 재질 0.6 명시)
6. ~~충돌 페널티 없음~~ → **2026-07-27 반영 (v3)**: `arms_contact` net-force 센서(양팔 link1~6,
   판=link7 제외) + `undesired_contacts` 페널티 + 성능 curriculum (cube Run B 패턴).
   남은 구멍: **판(link7)이 바닥을 치는 것은 미감지** — link7 센서는 박스 filter 전용이라
   분리 불가(PhysX body당 filter 1개). 문제 시 판-바닥 filtered 센서 추가 검토
7. **판 질량 미반영**: plate USD를 붙인 뒤에도 link7 mass가 증가하지 않음. 실물 plate/payload 질량·CoM을
   측정해 inertia 표현과 gravity-compensated controller를 함께 정합해야 함
8. ~~양 판 접촉 높이 자유~~ → **V2 `plate_height_symmetry`로 반영**. 남은 것은 실물 좌우 tracking/TCP
   skew에서도 동시·동고 접촉이 유지되는지 calibration하는 것

---

### 4-1. P10 시점 현황 갱신 (2026-07-29)

1. **오른팔 정렬 저하 (lift 후)** — §3-5 재발 이력 참조. alignment 0.034 고착.
   **→ P11: 게이트 P→G 교체 적용 (2026-07-29). resume이 아닌 보상 수정을 택한 근거:
   비대칭은 현 보상 지형의 균형점이라 iter 추가로는 안 깨짐 (P9→P10 재현이 방증)**
2. **excessive_squeeze −1.30 고착 (100N 임계에도)** — plate_forces obs 제거(v1)의 첫 청구서
   가능성: 힘 피드백 없이 "안전빵 과조임" 습관. 카드: resume 관찰 → 잔존 시 obs 복원 재검토
   (τ_ext-free 배포 이점과 트레이드오프)
3. **noise_std 1.27 미수렴 + success 상승 중** — resume(+3000)이 1순위
4. upright hold 0.465 (< 목표 0.7), arms_contact −0.19 (낮은 박스 접근 팔뚝 접촉)
5. (기존) 판 질량 실물 반영, obs latency

## 5. 로드맵

- [x] **5-1. 환경 구현 + 스모크 테스트** — 2026-07-27 완료 (obs 54 / action 14 / DR 분포 확인)
- [x] **5-2. rule-based 실현 가능성 검증** — 데모로 squeeze-lift 물리 성립 확인 (사용자 GUI 검증)
- [x] **5-3. 마찰 DR** — 2026-07-27 반영 (박스 static 0.5~1.0 / dynamic 0.4~0.9, 판 USD 재질 0.6/0.55 명시)
- [x] **5-4. 박스 yaw DR** — 2026-07-27 반영 (±0.3 rad)
- [x] **5-7. DR 존 시각화** — 2026-07-27 추가 (스폰=파랑, goal 상/하한=초록, cube task와 동일 방식)
- [x] **5-0~5-8. Run P1~P4 완료** — 상세 이력·결과는 `docs/log_per_task.txt`가 원본.
   요약: P1(모서리 걸침) → P2(성공 129.6, 모서리 몰림) → P3(실패: 정지 국소해) → P4(성공 123.6,
   역대 최고 속도. 잔존: 오른판 정렬 포기 + 박스 기울어짐)
- [x] **5-9. obs noise / env별 lifting 판정 / orientation obs** — P2(v3)와 P5에서 반영 완료
- [x] **5-10. Run P5a/b — 실패**: "잡고 가만히" 국소해 (+20cm 무보상 절벽 + tipping 디딤돌
   차단 + shaping 꿀단지). 부산물: 접근·정렬·조임 완성도는 역대 최고 확인, misalignment
   world-축 버그 발견. 상세는 `log_per_task.txt`
- [~] **5-11. Run P6 (준비 완료, 스모크 통과, obs 58)**: LIFT_DELTA 0.10 원복 + misalignment
   yaw-aware 버그픽스 + PhysX aggregate 64k. **중력 A/B 재시도** (a=중력 ON, b=disable_gravity
   override — b 성공 시 이후 b로 고정, 실물 중력보상 정합). 관찰: lifting 재발견(P4 페이스),
   alignment(min) 상승, box_upright ~0, play에서 "수평 + 회전 박스 파지" 개선

- [x] **5-5. Success 정의** — P7 반영: goal 10cm + 기울기 15° + 양판 접촉 0.5s 유지 → +30 (1회).
   `Episode_Reward/success / 6 = 성공률`. 평가 스크립트와 기준 통일
- [x] **5-11. P6~P9 시리즈** — 중력보상 채택(P6 A/B), success/과조임/수평 항 정착(P7 실패→P8
   커리큘럼화→P9 100N 픽스), 최종 성공률 77% + 강건성 drop 0. 상세 log_per_task
- [x] **5-12. v1(실물 박스) 신설 + P10** — 성공률 82%, P9 zero-shot 접촉 0으로 학습 필수성 확정
- [~] **5-13. P11 (진행 중: `2026-07-29_15-05-19`)** — alignment/misalignment 게이트 P→G 단일 변경, from-scratch.
   판정: alignment 0.034→상승 + play에서 lift 후 오른판 평행 유지 + success ≥ P10(82%).
   (resume 대신 보상 수정 선택 — 비대칭은 보상 균형점이라 resume으로 안 깨짐. resume은 P11 수렴 부족 시 후속 카드)
   이력: 1차(`13-30-34`)는 it937에서 사출 폭발→obs 오염→value inf→정책 NaN으로 크래시 (로그 보존).
   → NaN 가드 2층(박스 속도 상한 + out-of-bounds 종료, §2-5) 추가 후 재시작. 가드 검증 포인트:
   out_of_bounds 종료 비율이 학습 후 ~0 수렴 + value loss 정상 유지
- [x] **5-13b. P11~P16 정렬 씨름** — 게이트 P→G→×upright 시소, 거리게이트, 발견/정렬 딜레마.
   결론: **reward 항으로는 정렬 못 잡음 확정** (§3-0). 상세 log_per_task.
- [x] **5-13c. P17~P18 통합 + 품질항** — P17 reaching_aligned로 정렬 곱셈 통합(허공 따먹기 차단),
   P18 접촉품질 q를 lifting/tracking에 곱셈 도입 (모서리 걸침 억제, success 12%로 하락하나 품질↑).
- [x] **5-13d. P19~P22 reaching 커리큘럼** — 6→12(과잉지배, 붕괴) → **6→9(P20 success 64%·reach kernel
   0.33, 파지 최고)**. P21 resume 10k(87%)/P22 scratch 10k(83%)는 success↑ 대신 **파지 품질 침식**
   (reach kernel 0.03). 같은 리워드/시드, resume만 차이 → **리워드가 sloppy 파지 허용**이 드러남.
- [x] **5-13e. v2(외란) 신설 + P23/P24** — "품질을 물리로 필연화": 박스 랜덤 wrench + (P24부터)
   success-기반 점진 ramp + plate_height_symmetry. **P24 = success 74%(외란 full 속)·reach kernel 0.44
   유지 — 품질↔성공 트레이드오프 첫 해소.** 교훈: 외란은 안정만 강제(쐐기/기울인 세트도 안정) →
   flush는 별도 신호 필요. 부수: 크기 DR 미작동 실측 확정(§2-5), 질량·마찰 DR은 정상.
- [x] **5-13f. q' = q×align soft ramp ([08032])** — P26 격리 검증(q' 첫 작동, flush 방향↑, drop 0),
   **P28 scratch 수렴 success ~94%**(P27 resume은 막판 하락 → q'류 변경은 scratch가 나음). q'가 **방향(평행)
   해결**. 부산물: P25 confound(q'+크기DR 동시)로 단일변수 원칙 재확인, 크기DR은 v2서 off([08033b]).
- [~] **5-13g. 중심성 + success 지속화** — P27/P28 play 발견: 평행하나 **모서리 접촉**(위치=중심성 미해결,
   align은 평행이면 모서리도 만점) + success **1회성 게이밍**(끝 0.5s만 정렬).
   - **[x] ② success per-step 지속 ([08041], P29 학습중)**: 1회 latch → 0.5s 진입 후 매 스텝(weight 1500→20),
     termination 없어 "일찍 정렬·오래 유지"가 최대 이득 = 게이밍 소멸 인센티브. v2 전용(v0/v1 1회성 유지).
   - **[ ] ① 중심성 신호 (미적용)**: q에 위치 blend(q''=q·align·center) 또는 reaching std 0.3→0.15.
     ②의 upright 개선이 모서리 파지를 간접 완화할 수 있어 P29 결과 관찰 후 결정.
   - 잔여 카드: fine_grained 정리·기하 페널티(sym/mis/upright) 통합.
- [x] **5-13h. P37 접촉 타이밍 대칭 평가** — box yaw 31.0°→24.1°, yaw-aligned 0.103, drop 0.
  transient yaw는 감소했으나 grip-axis≈box yaw≈24°로 kinematic placement 잔차 확인.
- [x] **5-13i. P38 cause+outcome 완료** — run `2026-08-08_15-18-11`, 선택 `model_9850.pt`.
  Yaw ramp it4276→7277 완주 후 붕괴 없음. Full disturbance+Noisy-v2, 128env×1ep×3seed에서
  precise 0.737, upright 0.787, yaw-aligned 0.786, box yaw 2.7°, grip offset 6.4°,
  0.5s sustained base/yaw10/yaw15 SR=0.997/0.997/0.997, drop/OOB=0. P37의 23.6° yaw와
  yaw10 SR=0을 해결하면서 position/upright도 개선했다. 단 최초 접촉 <=40ms는 평균 0.477이고 force p95
  112N·transient grasp-loss 3~4/128ep라 cause/force 안전은 미완; 다음은 yaw 강화가 아니라 timing/force 계측.
- [~] **5-13j. P39 actual-contact XOR 준비 완료** — P38/V2는 동결하고 별도
  `DualFR3PlateBoxLiftEnvCfg_P39`/Gym task/log root를 등록. Object-filtered 좌우 접촉 `>1N`의 XOR를
  매 step 적분하며 weight는 `0.0`, lifting EMA>10 뒤 `-5.0`. 기존 `plate_approach_symmetry`는 유지한다.
  Scratch seed42/10k. 목표는 P38 yaw·position 성능을 유지하며 full-episode one-sided duration 50% 감소.
- [~] **5-14. sim2real/구조 실험 순서** — P39 actual contact → P40 bounded/snapshot action →
  P41 q7 mirror 검증 및 mirror loss/augmentation → asymmetric critic → 실측 temporal DR/history →
  real safety ladder. 각 변경은 직전 합격 기준점에서 한 축씩 분기한다.
- [ ] **5-6. sim2real 잔여**: plate/payload 질량·CoM, 좌우 base/TCP 정합, pose/action latency·interpolation,
  object CoM/contact compliance, independent `tau_ext` safety supervisor. Actor는 현재 force-free 56D 유지

---

## 6. Cube task(오른팔 그리퍼)와의 구조 대비

| 축 | cube lift (`dual_fr3`) | box lift (`dual_fr3_plate`) | 비고 |
|---|---|---|---|
| 파지 원리 | 그리퍼 기구적 파지 | **양팔 squeeze 마찰 파지** | 힘 제어가 본질 |
| action | 오른팔 7 **delta(0.1)** + 이진 그리퍼 | **양팔 14 delta(0.1)** | cube도 Run C 3차에 delta 전환(2026-07-28, 통일) |
| obs 접촉력 | 없음 | **v0만 2D; 현재 V1/V2 actor는 없음** | sim contact truth는 reward/critic 전용, §3-2 |
| 파지 자세 민감도 | **낮음** — 그리퍼가 물체를 가둠(cage) | **높음** — 중심·평행 어긋나면 미끄러짐 | plate가 어려운 근본 이유 |
| 정렬/품질 보상 | **없음** (그리퍼가 파지 보장) | reaching의 align + q(법선비율) 곱셈 | plate만 필요, 품질↔성공 트레이드오프 유발 |
| gate | 양 finger pad 0.5 N (binary "잡았나") | 양 판 1.0 N + q 곱셈 | cube는 품질을 gate/task와 분리, plate는 섞임 |
| 사다리 보상 | reaching → lifting | reaching → **squeezing** → lifting | 중간 단계 추가 |
| 물체 DR | v3 크기 45~55×45~55×95~105mm + density/mass | **현재 V2는 크기 고정 555×475×275mm, 질량 3.5~4.5kg + 마찰·외란** | |
| 물체 위치 | 테이블 위 (z=0.6) | **바닥 (z=0)** — 하향 도달 필요 | delta 채택 사유 중 하나 |
| object_dropping | 상판 −10cm | 제거 (무의미) | |
| curriculum | 성능 기반 (공유) | 동일 함수 재사용 | |
