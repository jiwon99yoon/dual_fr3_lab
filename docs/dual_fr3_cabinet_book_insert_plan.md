# 듀얼 FR3 캐비닛 열고 책 넣기: 작업 계획 (2026-11-05)

토큰이 끊겨도 이어서 할 수 있게 목표, 순서, 지금 상태, 논의할 것을 적어 둔다. 단계가 끝날 때마다 이 문서를 갱신한다.

## 1. 목표

- 시작: **왼손은 캐비닛 손잡이를 잡은 상태**, **오른손에는 책(네모난 박스)이 고정된 상태**.
- 동작: 왼손으로 손잡이를 당겨 캐비닛 문을 열고, 오른손의 책을 캐비닛 안에 넣는다.
- 성공: 책이 캐비닛 안에 들어가면 성공. 놓을 필요는 없다.
- 캐비닛 위치는 고정한다. 테이블·캐비닛 위치는 "왼팔이 손잡이를 잡은 채 문을 열 수 있는 workspace 안"이면 되고, 정확한 값은 rule-based 단계에서 움직여 보며 정한다.
- 그리퍼: UMI 팁을 단 franka hand(아래 2절).

## 2. 순서

| 단계 | 할 일 | 끝나는 조건 |
|---|---|---|
| 0. asset | UMI 팁 dual FR3 + husky USD를 만들고 robot cfg 추가 | Isaac에서 관절 이름·손가락 범위·충돌·TCP 위치 확인 |
| 1. 작업 환경 | 로봇(UMI), 테이블, 고정 캐비닛, 책. 시작 상태(왼손 손잡이 파지, 오른손 책) 만들기, 미리보기 스크립트 | **사용자 확인** |
| 2. rule-based | 왼팔: 경첩 원호를 따라 문 열기(IK). 오른팔: 열린 틈으로 책 넣기. 테이블·캐비닛 위치 조정 | 성공 영상, 성공률, 위치 확정 |
| 3. RL 환경 | manager-based task. 관측·행동·보상·성공·DR. 행동은 plate처럼 20 Hz, 측정 q + ±0.1 rad, PD | 학습, 평가, 영상 |

plate·cube lift 때처럼 rule-based로 작업이 물리적으로 되는지, 위치가 맞는지 먼저 확인하고 RL로 간다.

## 3. 지금 있는 것

### codex가 만든 것 (커밋 안 됨, 진단용)

- task: `source/dual_fr3_lab/dual_fr3_lab/tasks/manager_based/cabinet_insert/` (`cabinet_insert_env_cfg.py`, `mdp/`, `agents/`)
  - 이전 목표: 왼손으로 손잡이에 다가가 잡고 문을 열고, 오른손으로 큐브를 집어 넣기. 이번 목표는 시작 상태가 다르다(이미 잡고 있음, 책 고정).
  - 로봇: 기존 `dual_fr3_no_handeye.usd`(franka hand, UMI 아님), 행동에 그리퍼 binary 행동 포함, 중력 끔.
- 캐비닛: `content/cabinet/cabinet_bimanual_handle.xml`(MJCF를 Isaac에서 직접 import)
  - 문 `door`, 경첩 `drawer_joint`(hinge, z축, 범위 −120°~0°, damping 1, stiffness 0.2), 손잡이 body `handle`.
  - `cabinet_left_handle.xml`도 있다.
- 미리보기: `scripts/tools/preview_cabinet_cube_insert.py`
- rule-based: `scripts/tools/run_dual_fr3_cabinet_cube_insert.py`, 영상 `logs/rule_based/cabinet_left_open_right_insert.mp4`. 성공 동작은 확인되지 않았다.

### 로봇 asset

- 지금: `source/dual_fr3_lab/dual_fr3_lab/assets/dual_fr3_no_handeye.usd`, cfg `robots/dual_fr3.py` `DUAL_FR3_CFG`. 구동 관절 18개(팔 14 + 손가락 4).
- USD 만드는 법: `content/make_usd.md` (xacro → URDF → mesh 절대경로 URDF → Isaac Lab URDF 변환). 빌드 워크스페이스는 `~/fr3_ws/src`(`fr3_husky_description`, `franka_description`, `husky`).
- UMI 팁: JunHeonYoon/franka_description `main` 커밋 `954350f` "add support for UMI tip in Franka hand configuration"
  - franka hand xacro에 `umi_tip` 인자(기본 false). true면 손가락 visual·collision이 `meshes/robot_ee/franka_hand_white/visual/UMI_tip.stl`로 바뀌고 inertial도 바뀐다. MJCF xacro도 같이 바뀌었다.
  - `~/fr3_ws/src/franka_description`은 이 커밋보다 2개 뒤에 있다(fetch만 해 둠, 작업 트리는 그대로).
- JunHeonYoon/fr3_husky `devel` 브랜치(`9d37f90`): dual FR3 + husky 조립에 UMI를 단 버전. `~/fr3_ws/src/fr3_husky_description`은 git이 아닌 Downloads 복사본이라 devel과 비교가 필요하다.

## 4. 단계별 세부 계획

### 0. asset

1. `franka_description`을 `954350f`로, `fr3_husky_description`을 fr3_husky `devel`의 것으로 맞춘다.
2. `dual_fr3.urdf.xacro`를 `hand:=true`, UMI 팁 켜기, `hand_camera:=false`, `mobile:=true`로 전개해 URDF → abs URDF → USD를 만든다(`make_usd.md` 그대로).
3. `assets/dual_fr3_umi.usd`, `robots/dual_fr3_umi.py` `DUAL_FR3_UMI_CFG` 추가. 기존 `DUAL_FR3_CFG`와 관절 이름·게인·contact reporting을 맞춘다.
4. 확인: 손가락 열림 범위, UMI 팁 충돌 모양(convex 분해), TCP(손가락 끝 중앙) 오프셋, 기존 USD 대비 팔 관절 동일.

### 1. 작업 환경

- 새 task 폴더(이름 후보 `cabinet_book_insert`). codex의 `cabinet_insert`는 건드리지 않고 필요한 부분만 가져온다.
- 장면: 로봇(husky 고정), 테이블, 캐비닛(고정, 문 경첩만 움직임), 책(박스).
- 시작 상태: 캐비닛·테이블이 고정이므로 IK로 "왼손이 손잡이를 잡은 자세"와 "오른손이 책을 든 자세"를 한 번 풀어 표로 저장한다(plate의 pregrasp 표와 같은 방식). 에피소드마다 작은 노이즈만 준다.
- 미리보기: 시작 상태를 여러 시점에서 찍는 스크립트(영상·스크린샷).

### 2. rule-based

- 왼팔: 손잡이 위치를 경첩 축 둘레 원호로 보간한 목표를 IK로 따라간다. 손 방향도 문과 같이 돌린다.
- 오른팔: 문이 일정 각도 이상 열리면 책을 캐비닛 앞에 맞춘 뒤 안쪽으로 밀어 넣는다.
- 기록: 문 각도, 손잡이-손 상대 위치(미끄러짐), 책 위치, 충돌, 관절 한계 여유. 성공률과 영상.
- 여기서 테이블·캐비닛 위치와 문 여는 각도를 정한다.

### 3. RL

- plate와 같은 제어 계약: 20 Hz, q_target = q_측정 + clip(0.1a, ±0.1 rad), 팔 PD.
- 행동 14(팔만). 그리퍼는 닫힌 채로 둔다(아래 논의 4).
- 관측 후보: 관절각·속도, 문 각도·속도, 손잡이 pose, 책 pose(또는 손 기준 고정이면 생략), 캐비닛 입구 pose, 직전 행동.
- 보상 후보: 문 각도 진행, 손잡이 파지 유지, 책-입구 정렬, 책 삽입 깊이, 충돌·관절 한계·행동 변화 벌점, 성공 보너스.

## 5. 논의할 것 (시작 전에 정한다)

1. **asset 소스.** `~/fr3_ws/src`를 직접 바꿔도 되는가?
   - `franka_description` pull(2커밋)
   - `fr3_husky_description`을 fr3_husky `devel`로 교체
   - 아니면 별도 워크스페이스에 받아 빌드
2. **캐비닛 모델.** codex MJCF(여닫이 문, 경첩 −120°~0°)를 그대로 쓰는가, 실물 캐비닛 치수로 새로 만드는가? 실물이 있다면 문 크기, 경첩 쪽(왼/오), 손잡이 모양·높이, 내부 크기.
3. **책.** 크기·질량. 오른손 고정 방식:
   - (a) 손에 fixed joint로 붙임(미끄럼 없음, 간단)
   - (b) 손가락으로 실제로 쥠(마찰, 실물에 가까움)
   - v1은 (a)를 권한다.
4. **왼손 손잡이 파지.** 시작부터 잡고 있는 상태를 어떻게 만드나?
   - (a) 손가락을 닫아 실제 접촉으로 잡음(문을 당길 때 미끄러질 수 있음, 실물에 가까움)
   - (b) 손잡이에 붙이는 제약(안 미끄러짐, 대신 손이 문 원호를 정확히 따라가야 함)
   - 손잡이가 막대형이면 (a)도 해볼 만하다.
5. **성공 조건.** 책 8개 꼭짓점이 모두 캐비닛 내부 상자 안이면 성공으로 할지, 책 깊이 일정 비율 이상으로 할지. 문 각도 조건은 없음으로 둔다.
6. **그리퍼 행동.** 두 손 모두 처음부터 끝까지 닫힌 상태로 고정하고 행동은 팔 14개만 쓸지.
7. **codex 코드 처리.** 새 폴더로 따로 만들지, `cabinet_insert`를 고칠지. codex 파일은 아직 커밋 전이다.
8. **실물 목표.** 실로봇에도 UMI 팁이 달려 있는지, 실물 캐비닛이 있는지. 있으면 처음부터 plate와 같은 sim2real 계약으로 만든다.
9. **로봇 base·테이블 높이.** husky 고정, 테이블·캐비닛 높이를 실물 기준으로 할지.

## 5-1. 정한 것 (2026-11-05)

| # | 결정 |
|---|---|
| 1 | `~/fr3_ws/src`를 직접 업데이트한다 |
| 2 | 캐비닛은 codex MJCF(`cabinet_bimanual_handle.xml`)로 시작한다. 실물 캐비닛 모델이 준헌 description(fr3_husky `devel`)에 있으면 치수를 맞춘다 |
| 3 | 책은 캐비닛 안에 들어가는 박스. 크기는 내가 정한다(나중에 CAD로 뽑음) |
| 4 | 왼손은 손잡이를 UMI 그리퍼로 쥔 형태. v1은 손잡이에 왼손을 붙인다(제약). 손가락 관절은 손잡이 두께에 맞춰 고정 |
| 5 | 성공: 책 중심이 캐비닛 내부 중심 근처(일정 깊이 이상). 정확한 기준은 rule-based를 보고 정한다 |
| 6 | 그리퍼는 손잡이·책 두께에 맞는 열림으로 고정. 행동은 팔 관절 14개 |
| 7 | codex `cabinet_insert`는 두고 새 task를 따로 만든다 |
| 8 | 실로봇에 UMI 그리퍼가 있고 실물 캐비닛도 있다(준헌 description). 처음부터 plate와 같은 sim2real 계약으로 만든다 |
| 9 | husky 고정. 캐비닛 높이는 실물 기준. 테이블 높이는 조절 가능하므로 workspace에 맞춘다 |

## 6. 진행 기록

- 2026-11-05: 계획 작성. UMI 팁 위치 확인(franka_description `954350f`, fr3_husky `devel`). 논의 대기.
- 2026-11-05: 0단계 asset 완료. `~/fr3_ws/src`를 fr3_husky `devel` + franka_description `954350f` + JunHeonYoon/husky로 바꾸고 `assets/dual_fr3_umi.usd`, `robots/dual_fr3_umi.py`(`DUAL_FR3_UMI_CFG`, `DUAL_FR3_UMI_SIM2REAL_100HZ_CFG`, `UMI_TCP_OFFSET_POS/ROT`)를 만들었다. 절차와 확인 결과는 `content/make_usd.md`의 "UMI 그리퍼 USD". 실물 캐비닛 모델은 fr3_husky `devel`에 없다(위치 확인 필요).
- 2026-11-05: 1단계 작업 환경 초안. 사용자 확인 대기.
  - task `tasks/manager_based/cabinet_book_insert/` (`Isaac-Cabinet-BookInsert-DualFR3-UMI-v0`, `-Play-v0`). codex `cabinet_insert`는 그대로 둠.
  - 배치(base frame): 테이블 윗면 +0.045(바닥에서 0.45 m), 캐비닛 모델 원점 (0.90, 0.10), 180° 돌려 문이 로봇 쪽. 손잡이 막대 중심 (0.66, −0.035, 0.24), 경첩 (0.71, 0.26).
  - 캐비닛 MJCF: codex `cabinet_bimanual_handle.xml`은 손잡이 막대가 문 가장자리 밖(문 반폭 0.1725, 막대 y 0.205)에 붙어 있어 실물과 달랐다(사용자 지적, 실물 사진: 경첩 반대쪽 문 앞면 가장자리 근처에 세로 막대 손잡이).
    복사본 `assets/cabinet_book/cabinet_book_insert.xml`에서 손잡이를 문 앞면 y 0.135(막대 바깥이 문 가장자리 안쪽 2.25 cm)로 옮김. 경첩은 로봇 왼쪽, 문은 로봇 쪽으로 열린다(60° 열면 손잡이 (0.46, 0.12)). codex 파일은 그대로.
    codex 배치(0.80, 테이블 0.75 m)는 손잡이가 왼팔 base에서 0.95 m라 닿지 않았다. UMI TCP가 link7에서 0.284 m 나와 있어서 정면 접근 목표는 x 0.6~0.85가 맞다.
  - 책 0.15(깊이) × 0.03(두께) × 0.22(높이) m, 0.3 kg. 시작 중심 (0.70, −0.35, 0.24), 문이 쓸고 지나가는 원(경첩에서 0.364 m) 밖.
  - 시작 자세: `scripts/tools/solve_cabinet_book_start.py`(DLS IK, 손가락 방향 2가지 × 초기값 6개, 관절 여유 큰 것) → `assets/cabinet_book/start_pose.npz`. 왼손 flip(j7 + π), 관절 여유 왼 0.50, 오른 0.09 rad.
  - 붙이기: prestartup 이벤트가 env마다 fixed joint 2개를 만든다(왼 TCP ↔ 손잡이, 오른 TCP ↔ 책, articulation 밖). 손가락은 막대·책 두께 + 1 mm로 고정.
  - 확인(`scripts/tools/preview_cabinet_book_insert.py`, 사진 `logs/rule_based/cabinet_book/`): 시작 후 TCP-잡는 점 차이 0.01 cm. 시작 관절각을 절대 목표로 2 s 유지하면 관절 최대 1.1°, 책 1.8 cm 처짐(손잡이 이동 후 다시 잰 값). zero action(상대 계약)이면 책 무게 때문에 2 s에 8 cm 처진다 → 정책이 보상해야 한다.
- 2026-11-05: 2단계 rule-based 첫 성공(`scripts/tools/eval_dual_fr3_cabinet_book_rule_based.py`, 기본값). 사용자 확인 대기.
  - 결과(10 s, 1 env): 문 90.0°, 책 중심이 입구에서 11.8 cm 안(뒷면 4.3 cm 안), 기울기 1.9°, 캐비닛·문·책·다른 팔과 1 N 넘는 접촉 없음, 관절 여유 최소 0.40 rad, 행동 포화 22 step. 영상 `logs/rule_based/cabinet_book/rule_v4.mp4`(+ `_sheet.png`).
  - 제어: env 행동 계약 그대로(20 Hz, q_meas + clip(0.1a, ±0.1)). 팔마다 DLS IK(gain 2, 적분 없음). 여분 자유도는 시작 자세 solver가 따라간 관절 경로(start_pose.npz `left_path_q`, `right_path_q`) 쪽으로 null 0.5로 당긴다.
    null을 시작 자세로만 당기면(0.05~0.2) 왼팔이 문 20~25°에서 다른 해로 넘어가 j6 한계에 걸리고 문이 86°에서 멈췄다. 적분항은 문보다 앞선 왼팔 목표에서 windup이 생겨 뺐다. gain 1이면 책이 1.9 cm 처져 pre 도달 판정(1 cm)을 못 넘었다.
  - 순서: 왼팔이 문을 25°/s로 연다(실측보다 최대 5° 앞선 목표, 손잡이 원호). 문 75°부터 책이 pre 지점까지 옆으로 0.2 m/s, 그다음 +x로 0.12 m/s 삽입.
  - 배치·파지를 바꾼 이유(배치 탐색 `solve_cabinet_book_start.py --cabinet_x .. --cabinet_y .. --pitch ..`, 로봇만 놓고 기구학 검사):
    UMI TCP가 link7에서 0.284 m, 책 중심은 TCP에서 5.5 cm 더 앞이다. 그래서 수평 파지는 캐비닛 앞에서 팔이 접히고 안쪽 깊이에서는 손이 닿지 않는다.
    수평 파지는 x 0.90~1.05, y 0~0.3 어디서도 왼팔(문 10~75°에서 막힘)이나 오른팔(옆 이동 중 막힘) 중 하나가 실패했다.
    양손 45° 위에서 비스듬히 잡기(pitch, 손가락 축 기준 회전이라 막대·책을 잡는 방향은 그대로)에 캐비닛 (0.95, 0.10)이면 문 0~90°와 책 경로 전체를 관절 여유 0.40/0.48 rad로 따라간다.
  - 접촉으로 찾은 것(`--contacts`): 책 레인을 캐비닛 중심선에 두면 오른손(y 방향으로 넓음)이 열린 문에 닿았다(369 N) → 책 y를 중심 −4 cm로 옮김. 책이 캐비닛 앞면에 스쳤다(2 cm 간격) → 5 cm로. 캐비닛 중심(17.8 cm)까지 넣으면 45° 오른손이 윗판 앞 모서리에 닿는다(92 N, 14.5 cm에서) → 삽입 목표 12 cm.
  - 성공 깊이(결정 필요): 캐비닛 중심까지는 45° 파지로 닿지 않는다. 후보: 책이 완전히 들어감(중심 ≥ 입구 + 7.5 cm + 여유).
  - 손잡이·책 파지 방향: 왼손 flip True, 오른손 flip True(solver가 고름, cfg와 반드시 같아야 함 — 다르면 시작부터 손가락이 책에 박혀 시뮬레이션이 튄다).
    → RL env 확인 때 왼손 flip True가 특이 자세를 지나가 손잡이에 500~1200 N이 걸려서 왼손은 flip False로 바꿨다(7-9).
- 2026-11-05: 3단계 RL env 구현, rule-based 전체 행 확인, 학습 시작. 자세한 내용은 7-9.
- 2026-11-06: 실물 책상 최소 70 cm → 75 cm 배치(책상 +0.30, 수납장 x −0.10) Table75 task 추가. solver 121/121, rule-based 121/121(`--ki_cap 0.04`). 7-12.
- 2026-11-07: 실로봇에서 7498이 문을 다 열기 전에 책을 넣으려 함(sim과 같음) → DoorFirst task(v2 속도·보상 + 문 열릴 때까지 책 대기). 7-13.

## 7. RL 환경 설계 (2026-11-05, 사용자 결정 반영 — 남은 논의는 7-8)

기준은 rule_v4다. 시작 자세, 붙이기(fixed joint 2개), 배치(캐비닛 (0.95, 0.10), 양손 45° pitch), 책 레인(y −4 cm), 행동 계약을 그대로 쓴다.

### 7-1. 행동·제어 (rule_v4와 같음)
- 팔 관절 14개. q_target = q_meas + clip(0.1a, ±0.1 rad), 20 Hz, PD 80/4 + torque-rate 제한(plate Sim2Real-100Hz 계약). 손가락은 손잡이·책 폭에 고정.
- 에피소드 12 s(rule_v4는 9.6 s에 끝남).

### 7-2. 관찰
actor(policy)는 실물에서 얻을 수 있는 값만 받는다. 실물에서는 카메라 없이 FK만으로 계산한다.
문 각도는 왼손 TCP와 경첩 위치로, 책 pose는 오른손 TCP ⊗ 파지 오프셋으로 구한다(둘 다 손에 고정돼 있으니 성립).

| 항 | 차원 | 노이즈 | 실물 출처 |
|---|---|---|---|
| 팔 관절 위치(시작 자세 기준) | 14 | 0.005 rad | 엔코더 |
| 팔 관절 속도 | 14 | 0.05 rad/s | 엔코더 |
| 직전 행동 | 14 | - | 노드 |
| 문 각도 | 1 | 0.02 rad | 왼손 FK + 캐비닛 캘리브레이션 |
| 책 중심 위치(캐비닛 입구 기준: x 깊이, y·z 레인 오차) | 3 | 0.005 m | 오른손 FK + 파지 오프셋 |
| 책 자세(base 기준 quat) | 4 | 0.01 | 오른손 FK |

합계 50차원. phase 지시자는 넣지 않는다(plate처럼 실패 후 다시 시도할 여지를 남긴다).

critic은 위 항을 노이즈 없이 받고, privileged 값을 더 받는다: 문 각속도, 접촉력(책-캐비닛/문, 오른손-문/캐비닛, 팔-팔), 경첩 반력, DR 값(책 질량, 경첩 감쇠).

### 7-3. 보상 (rule_v4 순서를 따르는 단계형, 모두 dense)

| 항 | 식(개요) | 메모 |
|---|---|---|
| door_open | clip(φ/90°, 0, 1) | 문 여는 진행 |
| book_lane | f_door · (1 − tanh(‖책 yz − 레인 yz‖ / 5 cm)) | f_door = clip((φ − 60°)/20°, 0, 1)로 문이 열린 만큼만 켠다 |
| book_depth | f_door · f_lane · clip((깊이 + 0.15)/0.27, 0, 1) | 입구 15 cm 앞 → 12 cm 안 |
| success | 성공 조건이면 매 step +1 | 종료하지 않음(빨리 넣을수록 이득) |
| contact | −relu(F − 5 N) 합 | 책-캐비닛/문, 오른손-문/캐비닛, 팔-팔 |
| handle_force | −relu(경첩 반력 − 30 N) | 왼손이 문을 비틀거나 잡아당기는 힘 = 실물 UMI 미끄럼 위험 |
| joint_limit | −relu(0.1 rad − 여유) 합 | rule_v4에서 j6 한계가 문제였음 |
| action_rate, joint_vel | 작은 음수 | plate와 같은 크기 |

- 경첩 반력: 손잡이 고정 joint의 힘은 바로 읽을 수 없다. 대신 문 링크로 들어오는 경첩 joint 힘을 쓴다(문 자체 동역학은 작아서 ≈ 왼손이 문에 거는 힘). 구현할 때 API를 확인한다.
- 성공 조건(결정): 책 중심이 입구에서 ≥ 9 cm 안(뒷면 1.5 cm 안). 레인 y·z 오차 < 3 cm, 기울기 < 10°는 같이 본다.
- 경첩 반력 기준(결정): 30 N 초과분에 페널티.

### 7-4. 리셋(시작 상태 분포)
- 결정: 오른손이 책을 잡는 위치를 DR한다. 책은 테이블 위에 놓인 상태(바닥면이 테이블 윗면)이고, x·y를 각각 20 cm 안팎 범위에서 뽑는다. 목표(삽입 위치)는 고정한다.
  왼손(손잡이)과 문 0°는 고정. 책 위치마다 오른팔 시작 관절을 solver로 미리 풀어 표로 두고, 리셋 때 한 행을 뽑는다(관절에 노이즈를 주면 붙이기가 튀므로 표 방식).
- 결정: 순수 RL(rule-based 궤적 중간 상태 리셋은 쓰지 않음).
- 대비책(학습이 너무 안 될 때): rule-based 궤적 위 임의 시점(로봇 관절, 문 각도, 책 pose)에서 일부 에피소드를 시작한다. 기록기(`--record`)를 만들어 표로 쓰면 된다.
  긴 순차 과제라 뒤 단계(책 넣기)를 처음부터 연습하게 한다.
  rule-based 설정을 몇 가지 바꿔 궤적을 여러 개 뽑는다(문 속도 15/25°/s, 문 85~95°, 레인 ±2 cm).
- 상태 기록은 손에 붙인 관계와 맞아야 한다. 관절에 노이즈를 주면 붙이기가 튀므로 노이즈는 주지 않는다.

### 7-5. DR
- v1(결정): 동역학(책 질량 0.2~0.5 kg, 경첩 감쇠 0.5~3 N·m·s/rad, PD 게인 ±15%) + 관찰 노이즈(7-2) + 시작 책 위치(7-4).
- v2(나중): 캐비닛 위치 ±2 cm·yaw ±3°. 배치별 시작 자세 표가 필요하다(solver가 이미 배치 묶음 계산을 함). 책 파지 위치 ±1 cm.

### 7-6. 학습 설정
- rsl_rl PPO. plate Sim2Real-100Hz와 같은 tanh-squashed actor. 은닉층 [256, 128, 64] → 과제가 길어서 [512, 256, 128] 추천.
- gamma 0.99(12 s = 240 step이라 plate의 0.98보다 길게), lam 0.95, lr 1e-4 adaptive, entropy 0.006, num_steps_per_env 24.
- obs_groups {"actor": ["policy"], "critic": ["critic"]}.
- num_envs: 4096 (7-9 smoke 측정: 1024보다 처리량 2배, GPU 5.8 GB). cfg 기본값은 1024라 명령에서 `--num_envs 4096`을 준다.
- 로그: 성공률, 문 각도, 깊이, 최대 접촉력, 경첩 반력.

### 7-7. 작업 순서
1. rule-based 궤적 기록기(`--record`)와 상태 표(npz)
2. env: 관찰·보상·리셋·DR·critic 그룹, task 등록(`...-v0`, `-Play-v0`), rsl_rl cfg
3. 확인: rule-based를 RL env에서 돌려 보상 항별 값과 성공 판정이 맞는지 본다(학습 전)
4. 학습은 사용자 지시 후

### 7-8. 남은 논의 → 결정 (2026-11-05)
- 1(영역): 내가 정해서 보고. 2(들기 단계): 추가, 삽입 높이는 캐비닛 중심으로 고정. 3: 궤적 리셋 안 씀(대비책으로만 기록). 4: 12 s 유지. 5: 순서대로 진행.

(논의 당시 메모)
1. 책 시작 영역: x·y 각 20 cm 범위(±10 cm)로 해석. 중심 후보는 rule_v4 시작 (0.63, −0.39) → x 0.53~0.73, y −0.49~−0.29.
   y가 −0.29보다 크면 문 궤적·왼손에 가까워진다. 테이블 앞 끝(지금 x 0.55)은 0.45까지 당겨야 한다. 도달 가능 영역은 solver로 확인한 뒤 정한다.
2. 책 높이: 테이블 위면 책 중심 z 0.155로 rule_v4(0.24)보다 8.5 cm 낮다. 캐비닛 바닥판이 테이블보다 1.7 cm 높아서, 들어 올리기 단계(→ 0.24)가 추가된다.
3. 궤적 중간 상태 리셋을 같이 쓸지(추천 30~50%).
4. 에피소드 길이: 들기 단계가 늘어 12 s가 빠듯할 수 있다. 새 시작 분포로 rule-based를 돌려 본 뒤 정한다.
5. RL 전에 rule-based를 시작 분포 전체(표의 모든 행)에서 돌려 성공률을 본다.

### 7-9. 구현·확인 결과 (2026-11-05)

**책 시작 영역(결정)**: x 0.53~0.73, y −0.49~−0.29(rule_v4 시작 (0.63, −0.39) ±10 cm). 책은 테이블 위에 서 있다(중심 z 0.156).
- 테이블 앞 끝을 0.40, 폭을 1.2 m로 늘려 영역 전체가 테이블 위에 오게 했다.
- solver 지도(로봇만, 기구학): x ≥ 0.52는 전부 된다. x ≤ 0.49이면서 y ≥ −0.39이면 실패한다. 그래서 앞 끝을 0.53으로 정했다.
- 시작 자세 표 `assets/cabinet_book/start_table.npz`: 11×11 = 121행(2 cm 간격). 121/121이 들기 → x → 옆 → 삽입 경로 전체를 따라간다.
  - 관절 여유: 오른팔 경로 최소 0.13, 평균 0.39 rad, 왼팔 0.30 rad.
  - 명령 `python scripts/tools/solve_cabinet_book_start.py --headless --grid 11`. `--region_x/--region_y`를 주면 탐색만 하고 저장하지 않는다.
- 리셋은 표에서 한 행을 고르게 뽑는다(`mdp.reset_to_start_table`). 이전 `start_pose.npz`는 더 쓰지 않는다.

**solver 수정**
- 경로 따라가기: 영역 중심에서 2 cm씩 걸어 나가며 시작 자세를 푼다. 먼 행을 바로 풀면 다른 해로 넘어간다.
- null은 기준 관절 경로(책 위치로 가장 가까운 waypoint) 쪽으로 당긴다(k 0.05).
- 충돌 판정: 한 step 뒤 실측 관절이 명령에서 0.003 rad 넘게 밀리면 충돌로 본다.
- 오른팔 후보: 시작 q가 0.2 rad 넘게 다른 후보 4개 중 표 전체를 가장 많이 덮는 것을 고른다(seed 8, 121/121).
- σ_min 상승(새로 추가):
  - TCP Jacobian 최소 특이값을 DH 모델로 계산한다(sim Jacobian과 차이 0.005~0.014).
  - null-space에서 σ_min을 0.15까지 올린다(`--k_sigma 0.5 --sigma_cap 0.15`).
  - 후보 점수 = 관절 여유(최대 0.3) + 2·σ_min(최대 0.15).

**손잡이 힘 문제와 해결**
- 증상: RL env에서 rule-based를 돌렸더니 모든 env에서 문 17~26°일 때 손잡이 힘이 500~1200 N까지 올라가고 문이 1~1.5 s 멈췄다. 그 외 구간은 5~12 N이었다.
  - 측정: `JointWrenchSensorCfg`. 손잡이 링크로 들어오는 joint 힘 = 왼손이 손잡이에 거는 힘.
- 원인은 왼팔 특이 자세다.
  - 문·손잡이·캐비닛·테이블과 접촉은 없었다.
  - 왼손 flip True에서는 모든 후보가 문 20~25°에서 σ_min 0.02(평소 0.2)를 지나갔다.
  - IK를 빼고 solver 관절 경로를 관절 공간에서 그대로 따라가도 똑같이 걸렸다. 그러니 IK 문제가 아니라 물리 쪽 문제다.
- σ_min 상승만으로는 부족했다. 약하게 걸면 0.026, 강하게 걸면 0.057까지 올라가지만 문 85°에서 j6 한계에 걸렸다.
- 해결: 왼손 파지를 손목 축 기준 180° 돌렸다(`LEFT_GRASP_FLIP = False`). 여기에 σ_min 상승을 더하니 문 경로 전체에서 σ_min ≥ 0.15가 됐다.
  - 손잡이를 쥐는 모양은 같고 j7만 반대로 돈다. **실물에서 이 방향이 되는지 확인이 필요하다.**
- 결과: 121행 전체에서 손잡이 힘 최대 13 N, 30 N 초과 0.

**rule-based 전체 행 (RL env, DR 켬)**
- 명령 `python scripts/tools/eval_dual_fr3_cabinet_book_rule_based.py --headless --all_rows`
- 기본값을 바꿨다.
  - `--near` 0.02: 무거운 책(0.5 kg)이 1.6 cm 처져서 7행이 SLIDE에서 멈췄다.
  - `--ki` 0.05: 오른팔 적분항으로 처짐을 보정한다.

  | ki | 성공 | 메모 |
  |---|---|---|
  | 0 | 120/121 | z 처짐 3.3 cm |
  | 0.05 | 121/121 | 기본값 |
  | 0.2 | 59/121 | y가 4 cm 넘게 흔들림 |

- 결과: 121/121 성공.
  - 깊이 11.2~12.8 cm, 기울기 최대 2.2°, HOLD 도달 6.95~7.95 s.
  - 관절 여유 최소 0.12 rad, 손잡이 힘 최대 13 N, 1 N 넘는 접촉 없음.
- 보상 합(env 평균): door_open 10.2, book_approach 3.7, book_lane 3.0, book_depth 4.9, insert_success 20.6. contact, handle_force, joint_limit는 0이다.

**RL env (구현)**
- task: `Isaac-Cabinet-BookInsert-DualFR3-UMI-v0` / `-Play-v0`. replicate_physics False, 종료는 time_out(12 s)만.
- 관찰(`mdp/observations.py`)
  - policy 50차원: 7-2 그대로. 책 위치는 삽입 목표 기준이다.
  - critic 67차원: policy 항(노이즈 없음)에 문 각속도, 접촉력 13개(양팔 link4~7·손가락 12개 + 책, ×0.01), 손잡이 힘(×0.01), 책 질량, 경첩 감쇠를 더한다.
- 보상(`mdp/rewards.py`). 책 항은 문 60→80°에서 켜진다(f_door).

  | 항 | 가중치 | 내용 |
  |---|---|---|
  | door_open | 1.0 | clip(φ/90°) |
  | book_approach | 1.0 | f_door · (1 − tanh(‖책 − pre‖/0.3)) |
  | book_lane | 1.0 | f_door · (1 − tanh(레인 yz 오차/0.03)) |
  | book_depth | 2.0 | f_door · 레인 · 깊이 진행(입구 −15 cm → 목표) |
  | insert_success | 5.0 | 깊이 ≥ 9 cm, 레인 < 3 cm, 기울기 < 10° |
  | contact | −0.01 | 센서별 5 N 초과분 합 |
  | handle_force | −0.02 | 30 N 초과분 |
  | joint_limit | −1.0 | fr3 관절 한계 여유 |
  | action_rate / joint_vel | −0.005 / −0.001 | |

- DR(startup): 책 질량 0.2~0.5 kg, 경첩 감쇠 0.5~3, 팔 PD ×0.85~1.15. 관찰 노이즈는 7-2, 시작 행은 고르게 뽑는다.
- PPO(`agents/rsl_rl_ppo_cfg.py`): 7-6 그대로, max_iterations 5000, save_interval 100.
- 확인 못 한 것: 팔 링크 접촉 센서 12개가 rule-based 내내 0 N이었다. 접촉이 없어서인지 센서가 안 잡는지는 모른다(책 센서는 2.8 N으로 동작). 학습 중 contact 항이 계속 정확히 0이면 의심한다.

**학습**
- smoke(5070, 4 iteration)

  | num_envs | iteration | 처리량 | GPU 최대 | 시작 시간 |
  |---|---|---|---|---|
  | 1024 | 2.0 s | 12k step/s | 3.6 GB | 30 s |
  | 4096 | 4.0 s | 24.5k step/s | 5.8 GB | 100 s |

  4096으로 6시간이면 5070에서 약 5400 iteration이다. 에피소드 하나(240 step)가 10 iteration이다.
- v1 학습(2026-11-05 23:50~): 5070 `--seed 2 --run_name v1_5070`, 4090 기본 seed 42 `--run_name v1_4090`. 둘 다 4096 env, `--max_iterations 15000`.
  - 100 iteration마다 저장되므로 6시간 시점의 checkpoint를 본다.
  ```
  python scripts/rsl_rl/train.py --task Isaac-Cabinet-BookInsert-DualFR3-UMI-v0 --headless --num_envs 4096 --max_iterations 15000 --run_name v1_4090
  ```
- 볼 것(tensorboard `Episode_Reward/*`)
  - door_open이 먼저 오르는지(문 여는 법).
  - book_approach / book_depth가 오르는지(문 다음 단계로 넘어가는지).
  - insert_success. rule-based 기준 합은 약 20이다.
  - handle_force 페널티가 큰지.
- 잘 안 되면 7-4의 대비책(rule-based 궤적 중간 상태 리셋)을 쓴다.


### 7-11. v1 결과와 Upright (2026-11-06)

**v1_4090 (seed 42, 15000 iteration) 결과: 성공 0.**
- `insert_success`는 학습 내내 0이었다. reward는 iteration 4000쯤부터 평탄했다(mean_reward 약 44).
- 영상(`model_14999_views_ep1`)을 보면 2 s 안에 문을 86°까지 연다(끝에는 119°). 동시에 오른 손목을 돌려 **책을 77~113° 기울인 채로** 깊이 14 cm, 레인 오차 0 cm에 넣고 끝까지 버틴다.
- 원인: `book_lane`과 `book_depth`가 책 중심 위치만 보고 자세는 보지 않는다. 성공은 기울기 < 10°를 요구하므로, 정책이 성공 없이도 깊이와 레인 보상을 거의 최대로 받는 국소해에 머물렀다(Episode_Reward book_depth 1.67/2.0, book_lane 0.84/1.0). 45° 오른손은 깊이 14.5 cm에서 캐비닛 윗판 앞 모서리에 닿는데, 책을 눕히면 이 간섭을 피할 수 있다.

**Upright task (`Isaac-Cabinet-BookInsert-DualFR3-UMI-Upright-v0`, `-Upright-Play-v0`)**
- `book_lane`과 `book_depth`에 기울기 게이트(`tilt_gate_deg` (10°, 30°))를 곱했다. 10° 이하는 1, 30° 이상은 0이고 그 사이는 선형이다. 기울기는 시작 자세(base 축 정렬) 대비 회전각이다.
- dense 항 `book_upright` = 1 − tanh(기울기/0.3 rad)를 가중치 0.5로 추가했다.
  - 함수 이름은 `mdp.book_upright_tanh`다(38b0cb7의 `book_upright`(exp, std_deg)와 다름). 보상 항 이름은 `book_upright`.
- 38b0cb7(codex v2)이 기본 cfg에 넣은 `book_orientation`(가중치 2)과 link1~3 접촉 벌점(가중치 −0.03)은 Upright cfg에서 다시 뺀다. upright_4090 학습 때의 보상과 같게 하려는 것이다(관측은 같음).
- v1 task는 그대로 둔다(`tilt_gate_deg` 기본값 None).
- rule-based 전체 행(Upright task): 121/121 성공. book_lane 2.97, book_depth 4.94로 v1과 같고(기울기 최대 2°라 게이트 1), book_upright 5.44.
- 학습: `--num_envs 4096 --max_iterations 15000 --run_name upright_4090` (4090, seed 42, scratch), 로그 `2026-11-06_18-38-10_upright_4090`. iteration당 3.0 s.

### 7-12. 75 cm 책상 (Table75, 2026-11-06)

실험실 책상은 최소 70 cm라 45 cm 배치(책상 윗면 base +0.045)를 실물에서 쓸 수 없다. 실물 팔 장착면은 바닥 위 약 0.38 m다(sim 0.405, 타이어 눌림).

**배치**
- 책상 윗면 base +0.345 (+0.30). sim 바닥에서 0.75 m, 실물 바닥에서 0.725 m.
- 수납장 (0.85, 0.10) (x −0.10). 손 접근(45°)과 책 시작 범위(x 0.53~0.73, y −0.49~−0.29)는 그대로다.
- 넣기 직전 점과 목표는 수납장과 함께 움직인다: 목표 (0.792, 0.06, 0.54), 입구 x 0.672.

**왜 10 cm 당기나 (solver, 수납장·책 없이 팔만)**
- 수납장 그대로 + 45°: 오른팔이 넣기 끝에서 어깨→link7 0.79 m(최대의 약 98%)로 거의 펴진다. sigma_min 0.058.
- 수납장 그대로 + 20°: 121/121이나 여유 왼 0.28 / 오른 최소 0.11 rad. 파지도 바뀐다.
- 10 cm 당김 + 45°: 121/121, 여유 왼 0.57 / 오른 최소 0.32, 평균 0.50 rad. 45 cm 표(0.30 / 0.13 / 0.39)보다 크다.
- 시작 표 `assets/cabinet_book/start_table_75cm.npz` (env cfg 주석에 solver 명령).

**구현**
- `apply_table75(cfg)`: 책상(다리 포함), 수납장, 책 초기 위치, 리셋 표, 관찰 `book_pos` 목표, 보상(book_approach / lane / depth / insert_success)의 목표와 입구를 옮긴다. 다른 cfg(Slow 등)에도 쓸 수 있다.
- task: `Isaac-Cabinet-BookInsert-DualFR3-UMI-Table75{,-OpenDoor}{,-Play}-v0`. 보상은 v2(기본 cfg) 그대로.
- runner: 두 단계 모두 experiment `dual_fr3_cabinet_book_insert_75cm`. 그래서 2단계에서 `--load_run`으로 1단계 run을 바로 찾는다. OpenDoor는 2500 iteration.
- rule-based 스크립트와 `record_views.py` HUD는 목표, 입구, 성공 조건, 시작 표를 상수 대신 task cfg에서 읽는다.
- 45 cm cfg는 그대로다. 한 프로세스에서 두 cfg를 함께 만들어도 값이 섞이지 않음을 확인했다.

**rule-based (전체 121행)**

| task | 성공 | 깊이 | 기울기 최대 | 관절 여유 최소 | 손잡이 힘 최대 | 1 N 넘는 접촉 |
|---|---|---|---|---|---|---|
| 45 cm (회귀 확인) | 121/121 | 11.1~12.7 cm | 2.3° | 0.12 rad | 13 N | R_link7 27 N (1행) |
| Table75, `--ki_cap 0.02` | 119/121 | 12.0~12.2 cm | 2.6° | 0.32 rad | 14 N | 없음 |
| Table75, `--ki_cap 0.04` | 121/121 | 11.9~12.3 cm | 2.1° | 0.32 rad | 14 N | 없음 |

- `--ki_cap 0.02`에서 실패한 2행은 책 z가 레인보다 3.1 cm 낮았다(허용 3 cm). 팔이 더 뻗어 있어 책 무게로 처지는 양이 커서, 적분 보정 한도(2 cm)를 넘었다. 한도를 `--ki_cap`으로 바꿀 수 있게 했다.
- 보상 합(env 평균): door_open 10.2, book_approach 3.6, book_lane 3.6, book_depth 6.0, insert_success 22.9. 45 cm와 비슷하다.
- RL smoke(OpenDoor, 256 env, 3 iteration): 정상. env.yaml에 75 cm 목표와 시작 표가 기록되어 etri exporter가 그대로 읽는다.

**학습 (v2 순서)**
```
# 1단계: 문 열린 상태에서 시작 (2500 iteration)
python scripts/rsl_rl/train.py --task Isaac-Cabinet-BookInsert-DualFR3-UMI-Table75-OpenDoor-v0 --headless --num_envs 4096 --run_name opendoor_75cm
# 2단계: 문 닫힘, 1단계 마지막 checkpoint에서 이어서
python scripts/rsl_rl/train.py --task Isaac-Cabinet-BookInsert-DualFR3-UMI-Table75-v0 --headless --num_envs 4096 --max_iterations 5000 \
  --run_name closed_door_75cm --resume --load_run <1단계 run 폴더> --checkpoint model_2499.pt
```
- 보상이 v2라 속도 문제(관절 약 2 rad/s, 7-10)는 그대로다. v3 속도 제한을 다시 설계하면 `apply_table75`를 같이 적용한다.

**실물 배치**: etri `docs/dual_fr3_cabinet_book_layout_75cm.md`. FK 노드 보정값은 `hinge_position_base: [0.6635, 0.2555, 0.54]`, `left_tcp_closed_position_base: [0.6241, -0.035, 0.5259]`이다.

### 7-13. DoorFirst: 문을 다 연 뒤 책 (2026-11-07)

**문제**: 실로봇에서 7498이 문을 다 열기 전에 책을 넣으려 했다(충돌 위험으로 정지). sim에서도 같다(121행, 결정적 행동, `sequential` 행 배정).

| | 책 앞면이 입구에 닿을 때 문 | 책 중심이 입구를 지날 때 | 문 85° 도달 | 끝 기울기 |
|---|---|---|---|---|
| 7498 | 중앙 37° (30~44°), 0.75 s | 42° | 1.75 s | 4° |
| v1 (`v1_5070` 14999) | 중앙 52° (46~70°), 1.15 s | 60° | 1.75 s | 31° |

- 원인: 책 보상은 문 60~80°에서 켜지지만(`DOOR_GATE_DEG`), 그 전에 책을 옮기는 것을 막는 항이 없다. 정책은 문을 여는 동안 책을 미리 입구까지 가져다 둔다.
- 속도(관절 약 2 rad/s)는 v1/v2 그대로 둔다(사용자 결정 2026-11-07). 실로봇 제어기(Isaac PD 모드)도 속도를 제한하지 않아 sim과 같은 속도로 움직인다.

**구현** (`apply_door_first(cfg)`, task `Isaac-Cabinet-BookInsert-DualFR3-UMI-DoorFirst{,-Play}-v0`, runner는 v2와 같은 experiment)
- `book_wait` = `mdp.book_early_intrusion`, 가중치 −20: 책 중심 y가 `BOOK_WAIT_Y_B`(−0.14)를 넘은 정도(5 cm에서 1로 포화) × (1 − 문 게이트 80~90°).
  - −0.14 = 닫힌 문의 자유단 y −0.095 − 책 두께 절반 − 여유 3 cm. 책 시작 위치(y ≤ −0.29)와 rule-based WAIT 위치는 밖이다. 수납장 y가 같아 Table75에서도 같은 값이다.
  - 책이 레인(y 0.06)에 가려면 이 선을 반드시 넘으므로, 문이 열리기 전에는 어떤 경로로도 넣을 수 없다.
- 책 보상(접근·레인·깊이) 게이트 60~80° → 80~90°.
- 성공 조건에 `min_door_deg` 85°.
- 나머지(속도, book_orientation, 접촉, 손잡이 힘)는 v2와 같다.

**확인**
- rule-based 전체 행: 121/121, book_wait 평균 −0.06(가중치 −10 기준; 최소 −0.34). SLIDE가 문 75°(`--gate_deg` 기본)에서 시작해 생기는 작은 값이다. 접촉 > 1 N 없음.
- 같은 121행에서 보상 합(가중치 −10 기준):

  | | 7498 (v2 보상) | 7498 (DoorFirst) | v1 (DoorFirst) | rule-based (DoorFirst) |
  |---|---|---|---|---|
  | book_wait | − | −17.7 (−10.2 ~ −116) | −8.8 | −0.06 |
  | 합 | 116.6 | 92.1 | 52.6 | 약 66 |

  - rule-based 합이 7498보다 낮은 것은 순서가 아니라 속도 때문이다(첫 성공 약 7 s vs 1.2 s).
  - 문을 기다리는 빠른 정책은 7498보다 약 1~1.5 s 늦게 넣는다(문 85° 1.7 s 후 레인까지 0.2 m 이동 + 삽입). 성공·깊이·레인 보상으로 약 10~13 손해다. 가중치 −10이면 7498식 선진입의 벌점(평균 17.7, 최소 10.2)과 비슷해서, 여유를 두려고 −20으로 정했다. −20에서 7498은 book_wait 평균 −27.3(최소 −20.4)이다.
- 7498은 `sequential` 121행에서 115~120/121 성공이다(DR 표본에 따라; 실패 행은 문이 33~36°에서 멈춤). 위 표의 7498 수치는 실패 행을 포함한다.
- smoke(256 env, 3 iteration, open-door 2499에서 resume): 정상. 2499는 문 닫힘 시작에서 책을 바로 들고 와서 book_wait −0.08 → −2.4로 커진다(학습으로 줄어야 할 항).

**학습** (v2 순서의 2단계만 새로. 1단계 open-door는 문이 90°라 게이트 1, book_wait 0, 성공 문 조건 만족이라 v2 1단계와 같다)
```
python scripts/rsl_rl/train.py --task Isaac-Cabinet-BookInsert-DualFR3-UMI-DoorFirst-v0 --headless --num_envs 4096 --max_iterations 5000 \
  --run_name door_first --resume --load_run open_door_transfer_2026-11-06 --checkpoint model_2499.pt
```
- 확인할 것: Episode_Reward/book_wait가 0 근처로 가는지, insert_success가 v2만큼 오르는지, 문 80° 전에 책이 대기선을 넘는 행이 없어졌는지(`scripts/tools/eval_dual_fr3_cabinet_book_door_entry.py`, 아래).
```
python scripts/tools/eval_dual_fr3_cabinet_book_door_entry.py --headless --task Isaac-Cabinet-BookInsert-DualFR3-UMI-DoorFirst-Play-v0 \
  --checkpoint logs/rsl_rl/dual_fr3_cabinet_book_insert/<run>/model_7498.pt
```
- 7498 기준값: 대기선(y −0.14)을 문 80° 전에 넘는 행 121/121.
- Table75·OpenDoor 변형 등록은 task 구성 논의(속도·손잡이 힘 등) 후에 한다.

