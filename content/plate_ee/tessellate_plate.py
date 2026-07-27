# STEP -> 삼각 메시(JSON, meter 단위) 추출
import json
import Part

shp = Part.Shape()
shp.read("/home/dyros/fr3_ws/plate_ee/plate_ee_140x140x25.step")
verts, faces = shp.tessellate(0.2)  # 0.2mm 편차
data = {
    "points": [[v.x / 1000.0, v.y / 1000.0, v.z / 1000.0] for v in verts],
    "faces": [list(f) for f in faces],
}
with open("/tmp/plate_mesh.json", "w") as fp:
    json.dump(data, fp)
print("verts:", len(verts), "tris:", len(faces))
