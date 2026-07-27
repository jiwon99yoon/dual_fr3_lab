# FreeCAD script: dual-arm lift용 넓은 판 end-effector 생성
# mount_hand.step의 볼트 인터페이스(Ø6.3 관통 @ BCD Ø50 45°, Ø10.3 카운터보어,
# 머리 아래 살 10mm)를 그대로 유지 → 기존 M6 볼트 재사용 가능.
#
# 좌표계: mount_hand와 동일. z=0 = FT 센서 접촉면(장착면), +z = 물체 쪽(작업면).
import math
import Part
from FreeCAD import Vector

# ===================== 파라미터 (여기만 수정) =====================
PLATE_X = 140.0        # 판 가로 (mm)
PLATE_Y = 140.0        # 판 세로 (mm)
THICK = 25.0           # 판 두께 (mm) = UNDER_HEAD 17 + 카운터보어 8
CORNER_R = 12.0        # 모서리 라운드 (mm)

HOLE_D = 6.3           # 관통 구멍 지름 (mm)  [mount_hand 동일]
CBORE_D = 10.3         # 카운터보어 지름 (mm) [mount_hand 동일]
UNDER_HEAD = 17.0      # 장착면~카운터보어 바닥 (mm). 볼트는 기존보다 7mm 긴 것 필요
BOLT_BCD_R = 25.003    # 볼트 원 반지름 (BCD Ø50.006) [mount_hand 동일]
BOLT_ANGLE0 = 0.0      # 첫 구멍 각도(deg). 0 = 마름모(판 축 위), 45 = 정사각(원래)

CENTER_RECESS = False  # True면 mount_hand처럼 작업면에 Ø32 x 4mm 리세스 추가
OUT_DIR = "/home/dyros/fr3_ws/plate_ee"
NAME = "plate_ee_%dx%dx%d" % (PLATE_X, PLATE_Y, THICK)
# ================================================================

# 본체: 라운드 사각 판, z = 0 ~ THICK
half_x, half_y = PLATE_X / 2.0, PLATE_Y / 2.0
plate = Part.makeBox(PLATE_X, PLATE_Y, THICK, Vector(-half_x, -half_y, 0))
if CORNER_R > 0:
    edges = [e for e in plate.Edges
             if abs(e.tangentAt(e.FirstParameter).z) > 0.99]  # 수직 모서리 4개
    plate = plate.makeFillet(CORNER_R, edges)

# 볼트 구멍 4개: 관통 Ø6.3 + 카운터보어 Ø10.3 (작업면 쪽에서 파냄)
cutters = []
for i in range(4):
    ang = math.radians(BOLT_ANGLE0 + 90.0 * i)
    x, y = BOLT_BCD_R * math.cos(ang), BOLT_BCD_R * math.sin(ang)
    cutters.append(Part.makeCylinder(HOLE_D / 2.0, THICK + 2, Vector(x, y, -1)))
    cutters.append(Part.makeCylinder(CBORE_D / 2.0, THICK - UNDER_HEAD + 1,
                                     Vector(x, y, UNDER_HEAD)))
if CENTER_RECESS:
    cutters.append(Part.makeCylinder(16.0, 4 + 1, Vector(0, 0, THICK - 4)))

for c in cutters:
    plate = plate.cut(c)
plate = plate.removeSplitter()

assert len(plate.Solids) == 1, "solid가 1개가 아님"
print("Volume: %.1f mm^3  (Al 2700kg/m^3 -> %.3f kg)" %
      (plate.Volume, plate.Volume * 2.7e-6))
bb = plate.BoundBox
print("BBox: %.1f x %.1f x %.1f mm" % (bb.XLength, bb.YLength, bb.ZLength))

plate.exportStep("%s/%s.step" % (OUT_DIR, NAME))
plate.exportStl("%s/%s.stl" % (OUT_DIR, NAME))
print("saved: %s/%s.step / .stl" % (OUT_DIR, NAME))
