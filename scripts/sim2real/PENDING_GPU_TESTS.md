# ⏳ 남은 Test A — Reach joint-order 검증 (Isaac 파트)

> 갱신 2026-09-17. GPU/Isaac Lab headless smoke test는 통과했다. 다만 Reach 정책의
> articulation joint-order를 실측하는 아래 Test A는 아직 별도로 실행하지 않았다.

## 배경

Reach sim↔real 정합성(Test A) 중 **오프라인 파트는 완료(✅ ALL PASS)**:
- `reach_obs_contract_check.py` — obs 58D 레이아웃, 기본자세, npz actor(tanh) 일치 확인 완료.
- 결과 요약은 `docs/sim2real_gap.md` §8-3 표 참조.

**남은 것 = Isaac 런타임이 필요한 단 하나** → 현재 환경에서 실행 가능, 결과 확인만 남음.

## ▶ 실행

```bash
cd ~/dual_fr3_lab
~/anaconda3/envs/env_isaaclab/bin/python scripts/sim2real/reach_joint_order_check.py --headless
```

## 무엇을 확인하나 (가장 중요한 정합성 리스크)

Isaac articulation이 실제로 뱉는 joint 순서가 real 배포 가정
`[left_fr3_joint1..7, right_fr3_joint1..7]` 과 같은지 실측한다.
**여기가 어긋나면 58D 계약이 통과해도 정책 입력이 permute되어 실물에서 전혀 다르게 움직인다.**

스크립트가 출력하는 것:
1. `robot.joint_names` 전체 순서
2. obs term `SceneEntityCfg(".*_fr3_joint[1-7]")` 해석 순서 vs real 기대값 (관절별 OK/!! 표시)
3. reset 직후 policy obs 앞 14D(`joint_pos_rel`)가 `q - default`와 일치하는지, sim default == real DEFAULT 인지

### 판정
- **모두 PASS** → Reach 정합성 완결. 다음 단계(Phase 1 sysID: 같은 명령 주입 후 q/qd/τ 비교)로 진행.
- **joint 순서 FAIL** → `~/fr3_husky_etri/.../observation.py`의 `DUAL_ARM_JOINT_NAMES` /
  `DEFAULT_DUAL_ARM_JOINT_POSITION` 인덱싱을 sim 순서에 맞게 재정렬 후 npz 재검토.
- **default FAIL** → 학습 ready pose(`robots/dual_fr3.py init_state`)와 real DEFAULT 불일치. 한쪽 갱신.

> 별도 확인: 2026-09-16에는 `run_dual_fr3_sysid.py`가 RTX 5070에서 headless로 동작했고,
> 14관절 A=0.1 / dry friction=0.2 / viscous friction=0.003 PhysX readback 및 1 kHz custom
> torque path를 확인했다. 이는 SysID runner 검증이며, 이 파일의 Reach joint-order Test A를
> 대신하지는 않는다.

## 그 외 GPU/학습 PC에서 별도 확인

- 배포 정확본 `logs/.../dual_fr3_reach_sim2real_v2/2026-09-09_11-05-39/model_4999.pt` 는
  이 PC에 없음(로컬엔 `2026-08-30_18-59-08` run만). 해당 체크포인트 있는 PC에서
  `export_ppo_actor ... --output-activation tanh` 재출력 → npz(sha 150eac1b...)와 수치 일치 확인.

## 참고 파일
- 오프라인 검증: `scripts/sim2real/reach_obs_contract_check.py`
- 이 검증: `scripts/sim2real/reach_joint_order_check.py`
- 전체 맥락: `docs/sim2real_gap.md` §8 (프레임워크·real-only 요소·sysID 로드맵)
