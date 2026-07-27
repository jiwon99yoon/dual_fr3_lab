# FreeCAD script: mount_hand.step 형상 분석 (구멍 위치/직경, 두께, 외곽 크기)
import Part

shp = Part.Shape()
shp.read("/home/dyros/fr3_ws/mount_hand.step")

bb = shp.BoundBox
print("=== BoundBox (mm) ===")
print(f"X: {bb.XMin:.3f} ~ {bb.XMax:.3f}  (len {bb.XLength:.3f})")
print(f"Y: {bb.YMin:.3f} ~ {bb.YMax:.3f}  (len {bb.YLength:.3f})")
print(f"Z: {bb.ZMin:.3f} ~ {bb.ZMax:.3f}  (len {bb.ZLength:.3f})")
print(f"Solids: {len(shp.Solids)}, Faces: {len(shp.Faces)}")

print("\n=== Cylindrical faces (holes/bosses) ===")
seen = []
for f in shp.Faces:
    surf = f.Surface
    if surf.__class__.__name__ == "Cylinder":
        c = surf.Center
        a = surf.Axis
        r = surf.Radius
        # z-range of the face
        fb = f.BoundBox
        key = (round(c.x, 2), round(c.y, 2), round(r, 3), round(a.x, 2), round(a.y, 2), round(a.z, 2))
        if key in seen:
            continue
        seen.append(key)
        print(f"R={r:7.3f}  D={2*r:7.3f}  center=({c.x:8.3f},{c.y:8.3f},{c.z:8.3f})  axis=({a.x:.2f},{a.y:.2f},{a.z:.2f})  z[{fb.ZMin:.2f}~{fb.ZMax:.2f}]")

print("\n=== Planar faces normal to Z (plate surfaces) ===")
zplanes = {}
for f in shp.Faces:
    surf = f.Surface
    if surf.__class__.__name__ == "Plane":
        n = surf.Axis
        if abs(abs(n.z) - 1.0) < 1e-6:
            z = round(f.BoundBox.ZMin, 3)
            zplanes[z] = zplanes.get(z, 0) + f.Area
for z in sorted(zplanes):
    print(f"z={z:8.3f}  total area={zplanes[z]:10.1f} mm^2")
