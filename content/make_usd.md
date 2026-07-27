# dual_fr3 USD 제작 절차

fr3_husky_description의 xacro에서 IsaacLab용 USD를 만드는 전체 파이프라인.

```
xacro 수정 → colcon build → xacro 전개(URDF) → mesh 절대경로화(abs URDF) → IsaacLab 변환(USD)
```

## 0. 워크스페이스 구성 (최초 1회)

`~/fr3_ws/src`에 세 패키지가 필요하다:

| 패키지 | 출처 | 역할 |
|---|---|---|
| `fr3_husky_description` | `~/Downloads/fr3_husky-main`에서 복사 | 최상위 xacro (dual_fr3 조립) |
| `franka_description` | https://github.com/JunHeonYoon/franka_description | fr3 팔 정의 (`franka_multi_robot.xacro` + joint_limits 등 yaml) |
| `husky` | https://github.com/husky/husky (humble-devel) | husky 몸체·휠 mesh |

- 공식 frankarobotics/franka_description(2.x)은 `franka_multi_robot.xacro`가 삭제되어 **반드시 JunHeonYoon 포크**를 사용해야 한다.
- `husky` repo는 mesh 파일 때문에 필요하며, src에 있는 이상 colcon이 의존성으로 인식하므로 `husky_description`도 같이 빌드해야 한다.

## 1. xacro 수정

`src/fr3_husky_description/robots/dual_fr3.urdf.xacro`의 주요 arg:

| arg | 기본값 | 의미 |
|---|---|---|
| `mobile` | `true` | husky 모바일 베이스 포함 여부 (false면 box 베이스 2개) |
| `hand` | `true` | franka hand(그리퍼) 장착 여부 |
| `hand_camera` | `false` | 손목 realsense d435i 장착 여부 (직접 추가한 스위치) |
| `fix_finger` | `false` | finger joint를 fixed로 만들지 여부 |

- `hand_camera`는 ee2cam yaml 로드 + `realsense_d435i` 매크로 + `ee2d435i` joint 전체를 `<xacro:if>`로 감싼 것. 기본 false.
- husky 위 `top_azure_kinect`는 `mobile:=true`일 때 항상 포함된다 (hand_camera와 무관).

## 2. 빌드 + URDF 전개

```bash
cd ~/fr3_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select husky_description franka_description fr3_husky_description
source install/setup.bash

# 기본값(mobile=true, hand_camera=false)으로 전개
xacro src/fr3_husky_description/robots/dual_fr3.urdf.xacro > dual_fr3.urdf
# 손목 카메라 포함 버전
xacro src/fr3_husky_description/robots/dual_fr3.urdf.xacro hand_camera:=true > dual_fr3.urdf
```

- 빌드는 `$(find 패키지명)`이 풀리게 하는 용도(파일 설치뿐이라 1~2초).
- **xacro를 수정하면 반드시 재빌드 후 전개**해야 install쪽에 반영된다.
- 전개 결과는 include/macro가 전부 풀린 하나의 flat URDF.

## 3. mesh 경로 절대경로화

URDF 안의 `package://` URI는 ROS 전용이라 Isaac importer가 해석하지 못한다. 실제 경로로 치환:

```bash
sed -e 's|package://franka_description/|/home/dyros/fr3_ws/src/franka_description/|g' \
    -e 's|package://fr3_husky_description/|/home/dyros/fr3_ws/src/fr3_husky_description/|g' \
    -e 's|package://husky_description/|/home/dyros/fr3_ws/src/husky/husky_description/|g' \
    dual_fr3.urdf > dual_fr3_abs.urdf

grep -c "package://" dual_fr3_abs.urdf   # 0이어야 함
```

주의: 경로가 하드코딩되므로 `~/fr3_ws/src`를 옮기면 재변환이 안 된다 (이미 만든 USD는 무관).

## 4. URDF → USD 변환

```bash
cd ~/IsaacLab
./isaaclab.sh -p scripts/tools/convert_urdf.py \
    /home/dyros/fr3_ws/dual_fr3_abs.urdf \
    /home/dyros/fr3_ws/dual_fr3_vN.usd \
    --merge-joints --headless
```

- `--merge-joints`: fixed joint로 연결된 링크(휠 4개, 카메라, ROAS 구조물 등)를 부모에 병합. rigid body 수가 줄어 시뮬이 빨라짐. 구동 joint 18개만 남는다.
- `--fix-base`: 사용 안 함. root 고정은 IsaacLab `ArticulationCfg`에서 `fix_root_link=True`로 처리 (같은 USD를 고정/모바일 겸용 가능).
- drive stiffness/damping(기본 100/1)과 target(0)은 변환기 기본값일 뿐, 스폰 시 `ArticulationCfg`의 actuator/init_state가 덮어쓴다.
- 출력은 레이어 구조(`dual_fr3_vN.usd` + `configuration/dual_fr3_vN_{robot,base,physics,sensor}.usd`). 배포용 단일 파일이 필요하면 flatten해서 저장한다.

## 최종 산출물

| 파일 | hand_camera | 크기 | 특징 |
|---|---|---|---|
| `dual_fr3_no_handeye.usd` | false | ~30MB | 손목 d435i 없음. IL/RL 학습용 기본. 시뮬 카메라는 IsaacLab `CameraCfg`로 hand link에 부착하면 되므로 mesh 불필요 |
| `dual_fr3_handeye.usd` | true | ~38MB | 손목 d435i mesh + optical frame 포함. 실물과 동일한 외형/시각화가 필요할 때 |

두 파일 공통:

- flatten된 단일 USD (configuration/ 레이어 불필요)
- 구동 joint 18개: `left/right_fr3_joint1~7` (revolute), `left/right_fr3_finger_joint1~2` (prismatic)
- 휠·카메라·구조물은 merge되어 정적 mesh (husky는 움직이지 않음)
- root는 floating → 스폰 시 `ArticulationRootPropertiesCfg(fix_root_link=True)` 필요
- URDF의 finger mimic joint는 변환에서 소실됨 → 제어에서 두 finger에 동일 명령을 주는 방식으로 대응 (공식 FRANKA_PANDA_CFG와 동일한 관행)
- **joint4 [-176.3°, -6.7°], joint6 [25.2°, 264.8°]는 0을 포함하지 않음** (FR3 하드웨어 스펙). 초기자세를 반드시 `ArticulationCfg.init_state.joint_pos`로 지정할 것 (예: joint4=-2.36, joint6=1.57)
