# Sim2Real Transfer 전략 — dual_fr3 (plate box-lift / cube lift)

> 작성: 2026-07-28. 대상: `~/dual_fr3_lab`의 두 태스크 (plate가 주 대상, cube는 공용 항목만)
> 전제: 실물 = 양팔 FR3 + husky(고정 베이스), **FT 센서 없음**, **외부 고정 카메라** (결정 사항),
> 박스 인지는 1단계 ArUco → 2단계 vision policy
> 관련 문서: `DUAL_FR3_LIFT_PLATE_ENV_ANALYSIS.md` (§3-2 힘 추정, §4 남은 약점), `log_per_task.txt`

---

## 0. 갭의 분류와 현재 상태

sim2real 갭을 4축으로 나누고, 축별로 "이미 학습에 반영된 것 / 남은 것"을 관리한다.

| 축 | 이미 반영 (P5 기준) | 남은 것 |
|---|---|---|
| **Perception** | obs noise (pos σ1cm, ori σ2°) | ArUco 파이프라인, 가림, latency, vision policy |
| **Dynamics** | 물성 DR (마찰 0.4~1.2, 질량 0.6~1.4×, 크기 30~50cm), 액추에이터 게인 ±15% | 판 질량 미반영, obs/action latency, 실물 마찰 실측 |
| **Actuation** | delta action (스텝당 0.1rad = rate limit 내장), 중력보상 모델링 (P5b A/B 중) | FR3 제어 인터페이스 매핑, payload 등록 |
| **힘 감각** | plate_forces obs + noise σ2N | τ_ext 기반 추정 파이프라인, bias/drift 모델 |

원칙 (cube 강건성 진단에서 실증): **noise와 물성 DR은 단독으론 무해해도 "조합"이 드랍을 유발**
→ 축별 방안은 반드시 조합 상태로 sim 평가를 통과시킨 뒤 실물로 간다.

---

## 1. Perception — 박스 pose

### 1-1. 1단계: ArUco (state 정책 그대로 배포)

- 외부 고정 카메라 → ArUco 6DoF → obs의 `object_position`(3) + `object_orientation`(quat 4, w≥0)
- **다중 마커**: 판이 측면을 덮으므로 윗면 포함 여러 면에 부착, 보이는 마커로 pose 융합
- **필터링**: EKF/저역필터로 마커 소실 구간(가림) 유지 — 잡은 뒤에는 pose 변화가 EE 기구학과
  일치하므로 kinematics prior로 보간 가능
- **캘리브레이션**: 카메라-로봇 base 외부 캘리브레이션 오차가 곧 obs bias → sim의 noise DR은
  zero-mean이라, 캘리브레이션 잔차(수 mm~1cm)가 크면 bias 항 DR 추가 검토
- **각도 오차가 위치보다 큼** (2~5°) → 이미 orientation noise σ2° 반영, 실측 후 필요시 상향

### 1-2. 2단계: Vision policy (마커 제거) — Teacher-Student Distillation

- **Teacher** = P5 계열 state 정책 (정확한 box pose를 보는 특권 정책)
- **Student** = 외부 고정 카메라 이미지 + 고유수용 obs(관절 28, action 14, 접촉력 2) + goal command
  → teacher action을 DAgger 방식으로 모방 (student가 행동, teacher가 매 스텝 라벨)
- **Depth 우선**: 텍스처/조명 sim2real 갭이 RGB보다 작음. RealSense 등 RGB-D 전제.
  RGB로 확장 시 텍스처/조명/재질 DR 필수
- **메모리 필수**: 잡는 순간 판+팔이 박스를 가림 → frame stack 또는 LSTM으로 "가려지기 직전
  pose" 유지. 접촉 후엔 접촉력+기구학 obs로 "손 감각 운반"이 가능한 obs 구성이 이미 갖춰짐
- **카메라 DR**: 외부 고정이지만 재설치 오차 대비 pose ±수 cm/±수 deg DR
- **VRAM**: tiled depth camera × 512~1024 envs — 32GB의 주 용도
- 백업 옵션: perception 분리 (이미지→pose 추정 신경망 + state 정책 유지) — distillation이
  가림에서 고전하면 이 하이브리드로 후퇴
- 참고: DeXtreme(arXiv 2210.13702), Learning by Cheating, 휴머노이드 box loco-mani(2310.03191)

---

## 2. 힘 감각 — plate_forces (FT 센서 없음)

- **소스**: libfranka `tau_ext_hat_filtered` → `O_F_ext_hat_K` (EE 외력 추정, 가상 F/T). norm을
  obs `plate_forces`에 매핑
- **Payload 등록 필수**: 판+볼트(~1.4kg) 질량·무게중심을 FR3 payload로 등록해야 판 무게가
  외력으로 잡히지 않음 (FT tare에 해당)
- **오차 특성**: 모델 오차·관절 마찰로 수 N급 bias/drift — sim의 σ2N zero-mean noise에 더해
  **bias DR**(에피소드별 상수 오프셋 ±2~3N) 추가 검토
- **joint7 축-정렬 관측성 문제 (2026-07-28 논의)**: 판 법선력(조임력)의 방향 = TCP z = 정확히
  joint7 회전축 → joint7 토크는 조임력에 기여 정보가 거의 없음. 조임력 추정은 사실상
  joint 1~6이 담당 (어깨/팔꿈치는 레버암 0.4~0.6m라 20~40N 조임 시 10~20Nm — SNR 양호).
  norm만 쓰고 판 법선축 모멘트를 안 쓰는 우리 구성은 유리한 조건.
  남는 리스크: ① 자세 의존 관측성 (손목 폄/특이자세 근처에서 특정 성분 추정 저하 —
  task 자세 범위에서 J^T 조건수 확인) ② 관절 마찰/히스테리시스 ~0.1-0.3Nm/관절 → EE 힘
  오차 수 N (sim의 σ2N 가정과 대략 일치하나 실측 필요)
- **실측 캘리브레이션 계획**: 판을 고정된 힘 게이지(저울)에 task 대표 자세 수 개로 눌러
  τ_ext 추정치 vs 실측 대조 → 오차 크기/자세 의존성을 얻어 sim noise/bias DR 파라미터 보정
- 보험: 추정 품질이 낮으면 obs에서 제거하고 재학습해도 task 성립 (delta action + joint_pos가
  "명령 대비 미동 = 접촉"이라는 암묵 신호 제공 — 분석 md §3-2)

---

## 3. Dynamics / Actuation

- **판 질량 반영** (§4-7): sim link7에 판 1.4kg가 없음. 두 갈래 중 택1로 정합:
  ① sim USD에 판 질량 추가 + 실물 payload 미등록 (둘 다 "무거운 손목")
  ② sim 무반영 + 실물 payload 등록 (둘 다 "보상된 손목") ← **P5b(disable_gravity)와 정합, 권장**
- **중력보상**: P5a(중력 ON) vs P5b(disable_gravity) A/B 진행 중 — 실물 FR3는 항상 g(q) 보상
  하므로 원리상 P5b가 정합. A/B 결과로 확정
- **제어 인터페이스**: policy 50Hz delta joint position → FR3 측 1kHz 보간 스트리밍.
  delta scale 0.1rad가 자체 rate limit 역할 (스텝당 최대 5rad/s 상당). 실물 조인트
  속도/가속 리밋과 대조해 필요시 scale 하향
- **Latency**: obs(카메라 30Hz + 추정 지연 ~수십 ms)와 action 전송 지연 — sim에 obs latency
  DR(1~2 스텝 지연 랜덤)을 넣는 것이 다음 강건화 카드 (cube와 공용)
- **마찰 실측 보정**: 실물 판(알루미늄)-박스(골판지 등) 마찰을 간단 경사 실험으로 실측해서
  sim DR 범위(0.4~1.2)의 중심이 실물을 포함하는지 확인. 벗어나면 판 USD 재질값 조정

---

## 4. 안전 (실물 전용)

- **조임력 상한 → P7에서 반영 완료 (2026-07-28)**: `excessive_squeeze` 페널티 (판당 60N 초과분,
  w−0.05) 학습에 포함. 실물에서는 τ_ext 상한 reflex 병행. 근거: Bi-Touch(arXiv:2307.06423)의
  실물 "unwanted squeezing" 사례 — 전형적 sim 무해/실물 발현 갭
- **점진 투입**: ① 접근만 (박스 없이 자세 재현) → ② 정지 squeeze (들지 않음) → ③ 낮은 lift
  (판정선 수준 +20cm) → ④ full goal tracking. 각 단계에서 τ_ext·거동 확인 후 진행
- 가벼운 박스(<1kg)부터, workspace 소프트 리밋 + E-stop 상시
- 양팔 충돌: sim에서 contact penalty로 억제 학습됨 + 실물 FR3 자체 reflex가 최후 방어선

---

## 5. 평가 프로토콜

1. **Sim 강건성 평가** (배포 전 관문): noise+DR "조합" 조건에서 drop률·성공률 측정
   (cube의 `eval_dual_fr3_lift_robustness.py` 패턴을 plate로 이식 — hold/precise/drop 지표)
2. **실물 단계 평가**: §4의 점진 투입 각 단계에서 성공률 (예: 10회 중 성공 수), 실패 모드
   기록 (미끄러짐/기울어짐/충돌/미도달 분류 — sim의 실패 모드 분류와 대응시켜 갭 원인 추적)
3. **갭 원인 분리 원칙**: ArUco+state 정책을 먼저 배포해 "정책 갭"을 확정한 뒤 vision으로 —
   vision student가 실패하면 원인은 인지 쪽이라고 바로 확정 가능

---

## 6. 로드맵 (의존 순서)

```
0. P5 state 정책 완성 (진행 중: P5a/P5b A/B)
   └─ 판정 후: 접촉력 크기 페널티 + obs latency DR 추가 run (실물 안전 준비)
1. 실물 ArUco + state 정책 zero-shot
   ├─ 사전: payload 등록, 카메라-base 캘리브레이션, τ_ext→plate_forces 파이프라인, 점진 투입(§4)
   └─ 산출: 정책 갭 확정, 실측 noise/마찰/latency 값 → sim DR 파라미터 보정
2. sim: tiled depth camera + DAgger distillation (student 학습, VRAM 본격 사용)
   └─ 가림 대응: frame stack/LSTM, 카메라 pose DR
3. 실물 vision zero-shot → 갭 크면 perception 분리 하이브리드로 후퇴
```

각 단계의 실측값(오차·성공률)은 `log_per_task.txt`에 run 기록과 같은 형식으로 남길 것.
