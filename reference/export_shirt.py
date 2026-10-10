#!/usr/bin/env python3
"""Seamly2D sample men's shirt (.sm2d + .smis) -> per-piece OBJ (metres) + combined layout OBJ + report."""
import json
import os
import sys
import numpy as np
import mapbox_earcut as earcut

sys.path.insert(0, os.path.dirname(__file__))
from sm2d_eval import Pattern, load_smis, polygon_area, perimeter, self_intersections  # noqa: E402

S = "/home/claude/fashionfreedom/seamly2d/src/app/share/samples/"
SM2D, SMIS = S + "patterns/male_shirt.sm2d", S + "measurements/individual/male_shirt.smis"
OUT = "/home/claude/output/shirt"
os.makedirs(OUT, exist_ok=True)

meas = load_smis(SMIS)
pat = Pattern(SM2D, meas)


def tri(P_m):
    V = np.array(P_m, dtype=np.float64)
    T = earcut.triangulate_float64(V, np.array([len(V)], dtype=np.uint32)).reshape(-1, 3)
    return V, T


report = {"pattern": "male_shirt.sm2d", "measurements": meas, "units": "metres in OBJ, cm in report", "pieces": {}}
combined_v, combined_f, cursor = [], [], 0.0
GAP = 5.0   # cm between pieces in the combined layout

for name in pat.piece_names():
    poly, notes, attrs = pat.piece(name)
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    x0, y1 = min(xs), max(ys)
    cm = [(x - x0, y1 - y) for x, y in poly]                  # y flipped (up); origin = bbox bottom-left? see below
    ymin = min(p[1] for p in cm)
    cm = [(x, y - ymin) for x, y in cm]                        # origin = bounding-box bottom-left corner
    w, h = max(p[0] for p in cm), max(p[1] for p in cm)
    V, T = tri([(x / 100.0, y / 100.0) for x, y in cm])

    def tarea(a, b, c):
        u, v = V[b] - V[a], V[c] - V[a]
        return abs(u[0] * v[1] - u[1] * v[0]) / 2
    mesh_cm2 = sum(tarea(*t) for t in T) * 1e4
    area = abs(polygon_area(cm))
    xings = self_intersections(cm)

    with open(f"{OUT}/{name}_m.obj", "w") as f:
        f.write(f"# {name}  units: metres  origin: bounding-box bottom-left  z=0\n")
        f.write(f"# quantity {attrs.get('quantity')}  onFold {attrs.get('onFold')}  (half pieces are NOT mirrored)\n")
        for v in V:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} 0.000000\n")
        for t in T:
            f.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")

    off = len(combined_v)
    for v in V:
        combined_v.append((v[0] + cursor / 100.0, v[1]))
    combined_f.append((name, [(t[0] + 1 + off, t[1] + 1 + off, t[2] + 1 + off) for t in T]))
    cursor += w + GAP

    report["pieces"][name] = {
        "width_cm": round(w, 3), "height_cm": round(h, 3), "area_cm2": round(area, 2),
        "mesh_area_cm2": round(float(mesh_cm2), 2), "perimeter_cm": round(perimeter(cm), 2),
        "vertices": len(V), "triangles": int(len(T)), "self_intersections": xings, "gaps": notes,
        "quantity": attrs.get("quantity"), "onFold": attrs.get("onFold"),
        "area_vs_mesh_diff_pct": round(abs(area - mesh_cm2) / area * 100, 4),
    }

with open(f"{OUT}/shirt_all_pieces_layout_m.obj", "w") as f:
    f.write("# Men's shirt, all pieces laid out left-to-right, 5 cm apart. units: metres, z=0 (flat pattern pieces)\n")
    for v in combined_v:
        f.write(f"v {v[0]:.6f} {v[1]:.6f} 0.000000\n")
    for name, faces in combined_f:
        f.write(f"g {name}\n")
        for t in faces:
            f.write(f"f {t[0]} {t[1]} {t[2]}\n")

json.dump(report, open(f"{OUT}/shirt_report.json", "w"), indent=2)
print(f"{'piece':22s} {'w':>7s} {'h':>7s} {'area':>8s} {'qty':>4s} {'tris':>5s} {'xings':>5s} {'mesh/area diff %':>16s}")
for n, r in report["pieces"].items():
    print(f"{n:22s} {r['width_cm']:7.2f} {r['height_cm']:7.2f} {r['area_cm2']:8.1f} {str(r['quantity']):>4s} "
          f"{r['triangles']:5d} {r['self_intersections']:5d} {r['area_vs_mesh_diff_pct']:16.4f}")
