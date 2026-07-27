# Dual FR3 Cube Lift — 환경 · Reward 정밀 분석

> 대상: `Isaac-Lift-Cube-DualFR3-v0` (`lift/config/dual_fr3/joint_pos_env_cfg.py` + 베이스 `lift/lift_env_cfg.py`)
> 알고리즘: PPO (rsl_rl), 4096 envs
> 작성 기준: **2026-07-27 14:42** — grasp gating(ContactSensor) 구현·검증 완료 시점.
> **갱신 2026-07-27 16시경 (Run B 준비 완료)**: MJCF 테이블(상판 0.75) + 중력보상(disable_gravity)
> + 충돌 페널티 + 측정 joint 토크 obs(43 dims) 반영. §2는 Run B 기준, Run A까지의 상태는 git 이력 참조.
> 이력: run 4개 (`logs/rsl_rl/dual_fr3_lift/`) TensorBoard + `params/env.yaml` 대조 분석 포함.
> Run A(v5, grasp gating만): `2026-07-27_14-59-58`, 3000 iter 완료 — 분석은 §5-0 판정 기준 참조

---

## 1. Task Description

- **작업**: MJCF 테이블(얇은 상판 z=0.75 + 양옆 다리, 선배 MuJoCo 세팅 `dual_fr3_table.md`, 단 x=0.6 배치) 위 DexCube(4.5~5.5cm)를 오른팔로 잡아 들어올려, 매 에피소드 샘플링되는 목표 위치(공중, world z 0.855~1.005)로 이동
- **로봇**: Husky 베이스 + FR3 양팔. **오른팔 7 joint + 오른손 그리퍼만 제어**, 왼팔은 PD가 ready pose 유지 (action 없음). **로봇 링크 disable_gravity=True** — 실물 FR3의 중력보상 모델링 (payload 중력은 유지, stiffness 80과 함께 실물 임피던스 정합)
- **EE 정의**: `right_fr3_link7` + TCP 오프셋 (z+0.2104, yaw −45°) — FrameTransformer로 계산, rule-based 데모에서 검증된 값
- **에피소드**: 5.0 s × 50 Hz 제어 (sim 100 Hz, decimation 2) = **250 step**. 목표는 에피소드당 1회 샘플링 (resampling 5 s = episode 길이)

---

## 2. MDP 구조 (2026-07-27 14:42 기준)

### 2-1. Observation (policy, 총 43 dims, history 없음)

| Term | 내용 | Dims | 비고 |
|------|------|------|------|
| `joint_pos` | 오른팔 7 + 오른손가락 2, default 대비 상대값 | 9 | 왼팔 9개 제외 (차원 축소) |
| `joint_vel` | 위와 동일 joint | 9 | |
| `object_position` | cube 위치, **robot root frame** | 3 | |
| `target_object_position` | goal pose command (pos 3 + quat 4) | 7 | **quat 4 dims는 항상 identity → dead dims** (orientation 명령 미사용) |
| `actions` | 직전 action | 8 | arm 7 + gripper 1 |
| `joint_torques` | 오른팔 측정 joint 토크 (`get_dof_projected_joint_forces`) | 7 | **FR3 내장 토크센서(tau_ext) 대응 → 실배포 가능**. disable_gravity 덕에 링크 자중이 빠져 실물 tau_ext와 성격 일치 |

**obs noise (Run C부터 학습에 포함)**: joint_pos σ0.005 rad, joint_vel σ0.05, object_position σ0.01 m
(perception 모사), joint_torques σ0.5 Nm — 전부 additive Gaussian, `enable_corruption=True`로 학습 시 활성.
PLAY cfg는 corruption off(깨끗한 baseline 평가), `Play-Noisy-v0`가 재활성(강건성 평가, §5-6).
finger pad 접촉력은 obs에 넣지 않는다 — 실로봇에 F/T·촉각 센서가 없기 때문 (contact은 reward 전용, §3).
관절 토크는 실물에 있으므로 예외.

### 2-2. Action (8 dims)

| | 구현 | Scale |
|---|---|---|
| arm (7) | `JointPositionActionCfg`, default offset 기준 상대 위치 명령 | 0.5 |
| gripper (1) | `BinaryJointPositionActionCfg` — open 0.04 / close 0.0 **이진** | — |

### 2-3. Sensors (grasp 판정용, 2026-07-27 추가)

```python
# joint_pos_env_cfg.py — PhysX 제약: filter는 sensor body당 1:1이라 finger마다 분리
leftfinger_object_contact  = ContactSensorCfg(prim_path=".../right_fr3_leftfinger",
                                              filter_prim_paths_expr=[".../Object"])
rightfinger_object_contact = ContactSensorCfg(prim_path=".../right_fr3_rightfinger",
                                              filter_prim_paths_expr=[".../Object"])
```

- `filter_prim_paths_expr`로 **Object와의 접촉력만** 측정 (테이블·자기충돌 제외)
- 전제: `DUAL_FR3_CFG.spawn.activate_contact_sensors=True` (`dual_fr3.py`, 2026-07-27 변경)
- **주의**: ContactSensor 하나에 finger 2개를 regex로 묶으면 PhysX filter가
  `expected 2N, found N` 에러를 냄 → 반드시 finger당 sensor 분리
- 추가 (Run B): `right_arm_contact` = `right_fr3_link[1-7]` net force 센서 (filter 없음 →
  body 수 제약 없음). 충돌 페널티용 — finger는 제외 (물체를 만져야 하는 부위)

### 2-4. Reward (per-second 정규화 값 기준, TensorBoard `Episode_Reward/*`)

```
G      = 𝟙[min(‖F_left‖, ‖F_right‖) > 0.5 N]        # grasp gate: 양 finger pad가 동시에 Object 접촉
L      = 𝟙[z_obj > 0.64]                             # 높이 gate: 상판 + 4cm

r_total = 1.0  · (1 − tanh(d_ee-obj / 0.1))              # reaching_object (비gated, 접근 gradient)
        + 15.0 · L · G                                    # lifting_object        (grasp-gated)
        + 16.0 · L · G · (1 − tanh(d_obj-goal / 0.3))     # object_goal_tracking  (grasp-gated)
        +  5.0 · L · G · (1 − tanh(d_obj-goal / 0.05))    # fine_grained          (grasp-gated)
        − 0.1  · ‖a_t − a_{t−1}‖²   (curriculum 발동 후; 초기 −1e-4)   # action_rate
        − 0.1  · ‖q̇‖²              (curriculum 발동 후; 초기 −1e-4)   # joint_vel
        − 1.0  · Σ 𝟙[‖F_link_i‖ > 1 N]  (curriculum 발동 후; 초기 −1e-4)  # undesired_contact_penalty (Run B)
```

- 구현: `lift/mdp/rewards.py`의 `object_grasped` / `object_is_lifted_and_grasped` / `object_goal_distance_grasped`
- **설계 의도**: 들기·운반 보상 전부에 G를 곱해 "실제 파지 중"일 때만 지급.
  높이 gate(L)만으로는 손등에 얹어 나르는 exploit이 이득이었음 (§3, 2026-07-25 run에서 실증)
- G를 lifting에도 곱는 이유: tracking만 gate하면 얹어 나르기가 여전히 15/s를 벌어 exploit이 절반만 죽는다
- fine_grained weight는 5 (2026-07-25 run에서 10으로 올린 것이 exploit 발현에 기여 → gating 검증 후 재상향 검토)
- reaching은 gate하지 않음: 파지 전 접근 단계의 유일한 gradient

### 2-5. Termination / Curriculum / DR

| 항목 | 내용 |
|---|---|
| time_out | 5 s (성공해도 계속 — **success termination / sparse bonus 없음**) |
| object_dropping | cube z < 0.65 (상판 −10cm) |
| curriculum | `modify_reward_weight_on_lifting_learned` — lifting 보상(만렙~15/s)이 **10/s 초과 시** action_rate·joint_vel·contact 페널티 강화 (latch, 성능 기반) |
| DR: cube 크기 | 한 변 4.5~5.5cm, `mode="usd"` per-env 고정 (`replicate_physics=False`) |
| DR: cube 스폰 | x ±0.08, y ±0.12 (상판 위) |
| DR: goal | base 기준 x 0.35~0.55, y −0.45~**−0.15** (Run B~C에서 −0.05 시도 후 원복 — 중앙선 goal은 실익 없이 왼팔 충돌 위험만 증가, 2026-07-27), z 0.45~0.60 |
| DR: cube 물성 (Run C) | 마찰 static 0.4~1.2 / dynamic 0.3~1.0 (make_consistent), 질량 ×0.7~1.3 — startup, per-env |
| DR: obs noise (Run C) | §2-1 참조 (Gaussian, 학습 시 활성) |
| **없는 DR** | joint stiffness/damping, obs latency, 그리퍼 동역학 |

---

## 3. Grasp gating의 배경 — "높이 gate만으로는 안 된다"

### 3-1. 문제 (2026-07-25 run에서 실증)

gating 이전의 `object_goal_distance`는 gate가 `𝟙[z_obj > 0.64]` 하나뿐이었다. cube가 *어떻게*
공중에 있는지 — TCP로 파지했는지, 손등·팔뚝에 얹었는지 — 를 구분하는 항이 없어서,
**cube를 손등에 얹고 균형 잡아 나르는 정책**(서커스 거동)이 정상 파지와 거의 같은 보상을 받았다.

exploit의 기회비용은 `reaching_object` 1/s뿐 (전체 ~37/s의 3%). fine_grained를 5→10으로 올린
2026-07-25 run에서 "그리퍼 자세 제약 없이 cube 중심을 goal에 붙이는 정밀도 이득 > reaching 손해"가
확정되며 exploit이 지배 전략이 됐다.

### 3-2. 로그 증거 (4 run 비교, tail-50 평균)

| Run | iter | reaching (max 1) | lifting | pos err [m] | 거동 (play 관찰) |
|---|---|---|---|---|---|
| 07-24_13-24-51 | 1.5k | 0.24 | **0.00** | 0.34 | 학습 실패 (안 움직임) |
| 07-24_14-46-05 | 1.9k | **0.83** | 13.96 | 0.21 | **정상 파지** ✅ |
| 07-24_16-07-02 | 3k | **0.86** | 14.04 | 0.22 | 파지하나 왼팔 충돌 |
| 07-25_18-40-22 | 5k | **0.07** | 13.62 | **0.12** | 서커스 (얹어 나르기) |

`reaching_object` tail이 사실상 **"grasp 유지율" 지표**: 0.83~0.86(파지) vs 0.07(exploit).
**gated 재학습의 성공 판정도 이 지표 — tail ≥ 0.8 복귀 여부.**

### 3-3. Run별 config 차이 (env.yaml diff 확정본)

| | v1 (13-24-51) | v2 (14-46-05) | v3 (16-07-02) | v4 (18-40-22) |
|---|---|---|---|---|
| lift 판정 높이 | 상판+8cm | **+4cm** | +4cm | +4cm |
| 페널티 curriculum | 10k step | **30k step** | 30k step | **성능 기반 (lifting>10)** |
| cube 크기 DR | — | — | **4.8~5.2cm** | **4.5~5.5cm** |
| goal y 범위 | ~−0.05 | ~−0.05 | ~−0.05 | **~−0.15** (왼팔 충돌 방지) |
| fine_grained w | 5 | 5 | 5 | **10** |
| 결과 | 붕괴 | 정상 ✅ | 충돌 | 서커스 |

교훈:
- v1→v2: **페널티가 들기 발견보다 먼저 세지면 "안 움직이는" local optimum으로 붕괴**
  (성능 기반 curriculum은 이것의 올바른 일반화)
- v3: goal이 몸 중앙선(−0.05)까지 가면 오른팔이 왼팔 영역 침범 — **충돌은 goal 샘플링 문제**
- v4: fine_grained 2배 + 장기 학습(5k)이 grasp 없는 reward 구멍을 노출

### 3-4. Gate 설계 논의 — 왜 contact이고, 왜 obs가 아닌가

- **Reward는 배포와 무관**: reward는 학습 중에만 존재. 실로봇에 F/T·촉각 센서가 없어도
  sim의 contact 정보를 reward에 쓰는 데는 아무 대가가 없다.
- **Contact gate > EE 거리 gate**: 손등에 얹으면 finger *pad*가 아니라 hand 몸체에 닿으므로
  "양 pad 동시 접촉" 조건은 진짜 파지만 통과시킨다. EE 거리 gate는 얹은 위치가 TCP에서
  가까우면 뚫릴 수 있다.
- **Actor obs에 contact은 넣지 않는다**: 실로봇 배포 시 없는 신호. 대신 finger joint_pos가
  암묵적 파지 센서 역할 (close 명령 후 손가락이 cube 폭 ~2.2–2.8cm에서 멈추면 파지 성공,
  0까지 닫히면 실패 — FR3 실물 그리퍼도 width 피드백 제공).
- critic-only privileged obs(asymmetric AC)는 선택사항으로 보류 — 이 태스크는 partial
  observability가 크지 않음.

### 3-5. Gate 검증 결과 (2026-07-27)

v2 정상 파지 정책(model_1850)을 gated 환경에서 replay (16 envs × 250 steps):

```
step  25: lifted=0.94  gate=0.94
step  75~: lifted=1.00  gate=1.00   (이후 끝까지 동일)
```

- 모든 측정 지점에서 **gate == lifted** → 진짜 파지에서 false negative ≈ 0, threshold 0.5N 적절
- 검증 중 발견한 운영 팁: headless 스크립트가 `simulation_app.close()`에서 무한 대기하는
  현상 있음 (작업은 수 초 만에 완료된 상태) — 멈춘 듯 보이면 `kill -USR1`로 스택 확인

---

## 4. 남은 구조적 약점 (심각도순, 2026-07-27 Run C 기준 갱신)

> 해결됨 (이력): ~~충돌 무페널티~~ → Run B에서 right_arm_contact 페널티 + 성능 curriculum으로 해결
> (goal y −0.05 복원 후에도 contact ~0 확인). ~~obs noise / 물성 DR 부재~~ → Run C 학습 cfg에 포함
> (§2-1, §2-5; 근거는 §5-6 상호작용 진단).

1. **Success 개념 부재**: sparse bonus도, success termination도, 성공률 지표도 없다.
   "goal 근처에서 얼쩡거리기"와 "정확히 유지"의 구분이 tanh gradient에만 의존.
   (eval 스크립트의 precise-hold가 임시 대용, §5-6)
2. **그리퍼 인터페이스 갭**: sim은 50Hz 이진 토글이 가능하지만 실물 Franka 그리퍼는
   명령당 수백 ms — 정책이 토글을 남용하면 실물 전이 불가. play에서 토글 빈도 확인 필요,
   심하면 toggle 페널티 또는 명령 latch 도입.
3. **Cube 자세 — 두 층위로 구분 (2026-07-27 재논의)**:
   - *goal orientation* (목표 지점에서의 cube 자세): 대칭 cube 위치 이동 태스크라 불필요 — 의도된 미사용.
     obs의 goal quat 4 dims는 죽은 입력, `Metrics/orientation_error`는 무시할 지표.
   - *파지 시점 자세 정합* (그리퍼 yaw ↔ cube 면 정렬): **실환경 필수인데 현재 미커버.**
     스폰 yaw DR이 없어(x,y만 랜덤) 항상 축 정렬 스폰 → 정책이 고정 접근 방향만 학습
     → 실물에서 임의 yaw cube에 모서리 파지로 실패할 것. obs에도 cube 자세 정보 없음.
     **Run D 후보**: 스폰 yaw ±45° DR(정사각 대칭상 전 구성 커버) + obs에 cube yaw 추가
     (실물 6D pose 인식으로 획득 가능한 신호). obs 차원이 바뀌므로 별도 run으로 분리.
4. **obs latency 미모델링**: 실물 perception은 노이즈뿐 아니라 지연(수십 ms~프레임 단위)도
   있음. Run C 검증 후 1~2 step 지연 랜덤화 검토.
5. **이진 그리퍼**: 파지력 조절 불가. force 제어가 필요해지면 한계.

---

## 5. 로드맵

- [x] **5-1. Grasp-gated reward** — 2026-07-27 구현·검증 완료 (§2-3, §2-4, §3-5)
- [x] **5-2. fine_grained 5 원복** — 2026-07-27 완료
- [x] **5-0. Run A (gating만 재학습)**: `2026-07-27_14-59-58`, 3000 iter — **판정 성공**: reaching tail **0.853** (기준 0.8, v4 0.07), lifting 14.04, tracking 14.29 / fine 3.68 (역대 최고), drop ~0%. 서커스 exploit 소멸
- [x] **5-4. 충돌 페널티** — Run B에 포함 (right_arm_contact + undesired_contacts, 성능 curriculum). goal y −0.05 복원
- [x] **Run B 학습 실행 중**: `2026-07-27_15-59-59`, 3000 iter (2026-07-27 16시경 시작). 체크: contact_penalty가 curriculum 발동 후에도 ~0 유지 / reaching ≥ 0.8 / tracking 절대값은 goal 상향 탓에 Run A보다 낮게 시작 가능
- [ ] **5-3. Success 정의**: `𝟙[d < 0.05, N step 유지]` sparse bonus(+50) 및/또는 종료 + eval 성공률
- [x] **5-5. Run C 1차 (`2026-07-27_18-38-13`, 5000 iter) — 실패, 원인 규명 완료**:
  구성 = obs noise + cube 물성 DR + `object_hold_still`(goal 10cm 내 속도² w−0.5).
  실패 사슬: ① obs noise로 advantage 흐려짐 → entropy 우세 → noise_std 1.0→5.5 폭주
  ② iter ~1520 curriculum 조기 latch — **drop 종료 env는 "들다 놓친" 것이라 lifting 합이
  구조적으로 높은 편향 표본**인데 소수 drop 배치 평균으로 판정해버림 → 휘젓는 정책에
  action_rate −59/s 직격, 보상 −288 붕괴 ③ 이후 페널티를 쓴 채 재학습되어 나쁜 국소해 수렴
  (최종: reward 80 vs B 143, pos err 0.32, **contact −1.64/s = 왼팔 상시 충돌**, fine 0.16)
- [x] **curriculum EMA 수정 + goal y 원복** (2026-07-27, 스모크 통과): 배치 크기 가중 EMA
  (α = min(0.5, 3·batch/num_envs), term별 분리 저장) — drop 배치는 α~0.001로 무력화,
  timeout 배치도 2회 연속 초과 필요. goal y 상한 −0.05→**−0.15 원복** (중앙선 goal은
  실익 없이 충돌 위험만 키움). **Run C 2차 from-scratch 재학습 대기**

### 5-6. 강건성 진단 (2026-07-27, Run B 정책 model_4999, 32env×4ep/조건)

`Isaac-Lift-Cube-DualFR3-Play-Noisy-v0` + `scripts/tools/eval_dual_fr3_lift_robustness.py`
(축별 스위치 `--no_obs_noise` / `--no_physics_dr`)로 재학습 없이 취약점 측정:

| 조건 | hold | precise(<5cm) | drop/128ep |
|---|---|---|---|
| baseline (학습 분포) | 0.957 | 0.688 | 0 |
| obs noise만 | 0.920 | 0.592 | 0 |
| 물성 DR만 | 0.951 | 0.690 | 0 |
| **둘 다** (2회 반복) | 0.93 | 0.62~0.64 | **468~750** |

**결론**: 단독 축은 무해하나 **"perception 떨림 × 낮은 마찰" 상호작용**이 이동 중 슬립 드랍을
유발 — 실환경에서 항상 공존하는 조합이므로 zero-shot 불가의 주 원인. obs noise는 정밀도(−14%)를,
조합은 파지 유지(드랍)를 깎는다. → Run C는 두 축을 반드시 **함께** 학습에 포함 (구현 완료).
학습 후 같은 스크립트·같은 지표로 재평가하여 drop이 0 근처로 복귀하는지 확인할 것.

---

## 6. remove_hook (ICROS) 환경과의 구조 대비

| 축 | remove_hook (fixed_multi) | dual_fr3 lift (현재) | 시사점 |
|---|---|---|---|
| 순서 강제 | multiplicative r_ori × r_pos | **L · G · r_track (동일 철학 적용됨)** | 완료 |
| coarse+fine | 0.7·tanh(d/0.4) + 0.3·tanh(d/0.05) | 16·tanh(d/0.3) + 5·tanh(d/0.05) | 구조 동일 |
| sparse bonus | +100 (4cm 도달) | 없음 | §5-3 |
| 접촉 안전 | FT tare 페널티 + threshold DR | **충돌 페널티 (Run B, undesired_contacts + 성능 curriculum)** | 완료 |
| obs noise | pos/vel/FT 전부 | **pos/vel/obj_pos/토크 Gaussian (Run C)** | 완료 |
| history | ×5 stack (175 dims) | 없음 (36 dims) | 정적 태스크라 당장 불필요 |
| 성공 판정 | 4cm threshold, eval 3 seeds×100ep | 없음 | §5-3 |
