# dual_fr3_lift_v3 — 세로 cube(50×50×100) 배포 후보 환경 스펙 & sim2real 가이드

> **2026-08-08 현재 배포 기준:** [`DUAL_FR3_LIFT_CUBE_SIM2REAL.md`](DUAL_FR3_LIFT_CUBE_SIM2REAL.md).
> 이 문서의 v3 실험 결과는 유지하지만, inherited action loop와 `tts` 초 환산은 새 문서의 정정을 따른다.

> **v2(20Hz)의 후속 — 실물 cube 형상(50×50×100mm)에 맞춘 버전.** v2.md와 대부분 동일하며,
> **차이점(cube 크기 DR)만** 여기 명시하고 나머지(주기·obs·action·배포 절차)는 v2.md/v1.md 참조.
> 설계 근거: `DUAL_FR3_LIFT_ENV_ANALYSIS.md` §5-12·§9. run 기록: `log_per_task.txt`.
> 작성: 2026-08-03 (v3 eval 판정 직후)

## 0. 정책 신상명세

| 항목 | 값 |
|---|---|
| 배포 후보 checkpoint | `logs/rsl_rl/dual_fr3_lift_v3/2026-08-03_16-52-40/model_4999.pt` |
| 학습 | PPO (rsl-rl-lib 3.0.1), 4096 envs, 5000 iter, from-scratch, seed 42, [5090-v2: IsaacLab 2.2.1/Isaac Sim 5.0] |
| 네트워크 | Actor/Critic MLP [256,128,64] ELU, 입력 **36** → 출력 8 (v1/v2와 동일) |
| obs 정규화 | 없음 — raw 값 그대로 |
| 제어 주기 | **20 Hz** (물리 100Hz × decimation 5) — v2 상속 |
| 에피소드 | **8초 = 160 step** — v2 상속 |
| **cube 형상** | **50×50×100 mm 직육면체** (단면 5cm 정사각 × 높이 10cm) — v3의 핵심 차이 |
| 검증 성능 | 성공률 **1.000 / 1.000 / 0.984** (yaw 0/5/10°, strict 2.5cm 동일), **drop 0/384ep**, hold 0.94, precise 0.91, tts 1.25s — Play-Noisy-v3 기준 |

## 1. v2 대비 차이점 (이것만 다름, 나머지는 v2.md/v1.md 참조)

| 항목 | v2 | **v3** |
|---|---|---|
| cube 형상 | 정육면체 ~5cm | **직육면체 50×50×100** (단면 4.5~5.5 × 높이 9.5~10.5cm DR) |
| object spawn | ~~DexCube USD~~ | **`MeshCuboidCfg` primitive** (base 5cm, density 1000, physics_material 명시) |
| 크기 DR mode | ~~`"usd"` (무효)~~ | **`"prestartup"`** (실작동) |
| decimation / episode / obs / action / reward / 물성 DR | — | **v2와 완전 동일** |

**변경 이유**: 실물 cube가 **50×50×100mm로 확정(자로 실측)**. v2(정육면체) 정책을 10cm 세로 cube에
zero-shot 적용 시(Play-Tall-v2 검증) 파지는 됨(hold 0.97, SR 0.99)이나 유지 중 **drop 160/128ep**
(CoM이 높아 기울며 슬립). → 세로 물체를 학습 분포에 넣어 재학습.

## 1-1. ⚠ 크기 DR 버그 수정 (2026-08-03, v0~v3 소급) — 중요

v3 개발 중 **프로젝트 내내 크기 DR이 무효였다는 것을 발견**했다 (play에서 사용자가 "cube가 세로가
아닌 것 같다"로 포착 → sim 실측 → 전 env 4.80cm 균일). 원인·수정:

- **원인**: `randomize_cube_size`가 `mode="usd"`였는데, IsaacLab 2.2.1에서 **env는 "prestartup"
  모드만 호출**한다(`manager_based_env.py:155`). "usd"는 아무도 호출 안 하는 모드 → 이벤트가
  등록만 되고 **에러 없이 침묵**. (함수 docstring이 "usd"라 잘못 안내 — 버전 네이밍 잔재에 낚임.)
- **영향**: v0~v2의 "크기 DR 4.5~5.5cm"는 전부 허상 — 실제 **4.80cm(DexCube 6cm × 0.8) 고정**
  학습이었음. 성공률 100% 등 성과는 "4.8cm 고정 정책"으로서 유효하나 **크기 강건성 주장은 무효**.
- **정상이었던 것**: 물성 DR(마찰·질량)은 `startup` 모드라 정상 작동 (실측: 16env 전부 상이).
- **수정**: ① object를 primitive(`MeshCuboidCfg`)로 전환 (크기 투명·instanceable 이슈 회피)
  ② `mode="prestartup"` ③ scale 재계산. 실측 검증: v3 = 단면 4.5~5.5 × 높이 9.5~10.5cm
  **per-env 랜덤** 확인 (프로젝트 최초 크기 DR 실작동).
- **다른 태스크 참고**: dexsuite 8종이 같은 함수를 `mode="prestartup"`으로 정상 사용 중이었음.

## 2. 성능 (v2 대비, 동급 — drop은 핵심 성과)

| 지표 | v3 (세로) | v2 (정육면체) | 비고 |
|---|---|---|---|
| 성공률 (yaw 0/5/10°) | 1.000 / 1.000 / **0.984** | 1.000 / 1.000 / 1.000 | yaw10°만 미세 하락 |
| **drop** | **0 ×3** | 0 ×3 | **v2 zero-shot의 160을 학습으로 완전 해결** ← v3 존재이유 |
| hold / precise | 0.94 / 0.91 | 0.96 / 0.94 | 세로라 약간 낮음 |
| upright (학습 tail) | **2.38** | 2.75 | ⚠ 낮음 (숙인 파지 — §3-1) |
| reaching / lifting | 0.824 / 13.89 | 0.861 / 14.44 | 세로 파지 난이도 |
| tts | 1.25s | 1.16s | 동급 |

→ **세로 cube를 안정적으로 드는 데 성공 (drop 0).** 정밀도·upright가 v2보다 약간 낮은 건 세로
물체의 본질적 난이도 + 아래 파지 전략(§3-1) 때문. 기능적으로는 배포 가능 수준.

## 2-1. 새 파지 전략 — "숙여서 CoM을 무는" 거동 (2026-08-03 관찰)

play에서 v3는 v2와 다르게 **cube 옆면 중간(CoM, 바닥+5cm)을 그리퍼로 숙여서 무는** 전략을 씀.
- **원인**: object_position이 CoM(중심)이고 tracking/fine 보상이 "cube 중심을 goal에 붙여라"라
  → 정책이 중심 높이(옆면 중간)를 무는 게 유리 → 위에서 수직이 아니라 숙여서 접근.
- **평가**: 이것이 **drop 0을 만든 좋은 적응** (세로 물체를 위에서 어설피 잡으면 기울어 슬립).
  콩 박기(finger가 테이블 타격)는 play에서 거의 안 보임 (사용자 확인).
- **트레이드오프**: upright 2.38로 낮음 → 운반 중 cube가 다소 기울 여지. 실물 전이엔 위쪽 파지 +
  수평 운반이 더 안정적이므로 개선 여지 있음 (→ v3.1, §4).

## 3. 배포 절차 — v2.md와 100% 동일

주기(20Hz), obs 36, action 규약, ready pose, 그리퍼 히스테리시스, object_position 소스 전략
(A안 ArUco / B안 파지 후 FK 전환, grasped 판정 안전조건), frame 정합 검증 — **전부 v2.md §3·§3-1
그대로**. cube 크기만 다르므로 아래 하나만 추가로 유의:

- **cube 오프셋 (B안 FK 전환용)**: cube 중심 = TCP에서 그리퍼가 문 지점 기준 오프셋. 세로 cube라
  그리퍼가 CoM(중심)을 물면 오프셋 ≈ 0, 중심에서 벗어나 물면 그만큼 보정 (CAD로 확정).
- perception(팀원 코드 `6_publish_object_multiple_qr.py`)이 이미 cube **중심**을 base frame으로
  발행 → 우리 object_position과 정합. 단 §3-1의 frame 정합 검증(스폰존 중앙 대조)은 필수.

## 4. TODO / 다음 카드

### v3 고유
- [x] cube 크기 DR 버그 수정 + primitive 전환 + 실측 검증 (2026-08-03)
- [x] v3 학습 + eval 통과 (drop 0 확정)
- [ ] **v3.1 예약 (수평 운반 개선, 선택 — 실물 1차 후 판단)**: upright 2.38 낮음 + 숙인 파지 →
      운반 중 기울 여지. 방법: v3 동일 + **upright weight 3 → 5** (온건; 6은 과함 — 숙임은 drop 0을
      만든 적응이라 너무 세게 누르면 놓칠 위험). 판정: 위쪽 파지 전환 + drop 0 유지 + upright 2.7+.
      **⚠ 지금 바로 하지 말 것** — 실물에서 운반이 실제로 불안정한지 보고 결정 (sim upright 2.38이
      실물 문제일지 미지수, 데이터 없이 튜닝 = 추측). v3로 실물 1차 → 불안정하면 v3.1.
- [ ] 콩 박기 대응 (실물서 관찰 시에만): 현재 `finger_table_impact`는 무효 (finger 센서가 Table을
      `force_matrix_w`로 읽는데 **static collider는 GPU contact filter 미지원** → 항상 0, no-op).
      해결: **locomotion식 net-force 방식** — finger에 filter 없는 net-force 센서 + `undesired_contacts`
      (net_forces_w, threshold) + 들어올림 게이팅(cube 무는 정상 접촉 제외). static Table 문제 원천
      회피 + 팔 충돌 페널티(이미 작동 중)와 동일 방식. Table을 kinematic으로 바꾸는 것(성능비용)보다
      나음. **단 play에서 콩 박기 거의 안 보여 우선순위 낮음.**

### v2에서 이어받은 카드 (여전히 유효)
- [ ] 실물 perception 실측 주기 확인 (팀원) → 20Hz 정합 최종 판단. 팀원 값: 30Hz, base frame, 중심 발행
- [ ] **팀원 확인 요청 (조용한 불일치 방지 — 크기 DR 교훈)**:
      ① 정지 상태 ArUco 오차 (1~2cm면 재학습 불필요, 5cm면 calibration bias → 재수행)
      ② `--base-frame`이 우리 URDF root(팔 장착 상판, world z=0.405)와 **같은 링크인지** (§3-1 대조)
      ③ 마커 layout(ID↔면)이 실물과 일치하는지 (`6_publish_..._qr.py`의 DEFAULT_MARKER_LAYOUT)
- [ ] latency DR 추가 (perception 지연 실측 후 그 값으로 — 1차 실험 전 필수 아님)
- [ ] (선택) dec 10(10Hz) 실험 — 표준 하한
- [ ] 외란 주입 (실물 슬립 관찰 시): (b) 마찰 하한 → (a) external force 순 (§외란 관찰카드)
- [ ] vision/tactile 직접 입력: 현재 불필요 (좌표+DR로 충분), 좌표 부족 시 VT-Refine 참고
- [ ] eval tts 표시 버그: 스크립트 50Hz 하드코딩 → 20Hz 정책 tts 오표시 (성공률·drop은 정확)
- [ ] 실물 cube 무게 측정 ★: sim density 1000 → 5×5×10 ≈ **0.25kg** 가정, mass DR ×0.7~1.3 커버.
      실물이 이 범위(0.175~0.325kg) 밖이면 density 수정 후 재학습

### 배포 checkpoint 선택
- **세로 cube(50×50×100) 실물 = v3**. 정육면체 실물이면 v2. perception 주기(30Hz)엔 v2/v3(20Hz) 정합.
