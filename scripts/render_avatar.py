"""Developer tool: render an avatar folder (avatar.obj + arrangement_points.json) to a PNG with matplotlib.

Orthographic front / side / back views, painter's algorithm, points coloured by body part. Not part of the packages.
Usage: python scripts/render_avatar.py outputs/avatar/male_173 [more folders ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

GROUPS = {
    "Head": "#e6194b",
    "Neck": "#f58231",
    "Chest": "#3cb44b",
    "Waist": "#4363d8",
    "Hip": "#911eb4",
    "UpperArm": "#008080",
    "Forearm": "#46f0f0",
    "Hand": "#808000",
    "Thigh": "#f032e6",
    "Shin": "#9a6324",
    "Foot": "#000000",
}


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    v, f = [], []
    for ln in path.read_text().splitlines():
        if ln.startswith("v "):
            v.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("f "):
            f.append([int(x) - 1 for x in ln.split()[1:4]])
    return np.array(v), np.array(f)


def view_matrix(name: str) -> tuple[np.ndarray, np.ndarray]:
    """(rotation: world -> view, view direction toward the camera) for 'front' | 'side' | 'back' | 'three_quarter'."""
    yaw = {"front": 0.0, "side": 90.0, "back": 180.0, "three_quarter": 35.0}[name]
    a = np.radians(yaw)
    # camera on the +z side at yaw 0, looking toward -z; the viewer's right is +x (= the wearer's left)
    r = np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])
    return r, r.T @ np.array([0, 0, 1.0])


def render(folder: Path, out: Path | None = None) -> Path:
    v, f = load_obj(folder / "avatar.obj")
    pts = json.loads((folder / "arrangement_points.json").read_text())["points"]
    views = ["front", "three_quarter", "side", "back"]
    spans = []
    for name in views:
        px = (v @ view_matrix(name)[0].T)[:, 0]
        spans.append((float(px.min()) - 0.06, float(px.max()) + 0.06))
    widths = [b - a for a, b in spans]
    fig, axes = plt.subplots(
        1, len(views), figsize=(sum(widths) * 4.2 + 1.0, 9.0), dpi=110, gridspec_kw={"width_ratios": widths}
    )
    for ax, name, span in zip(axes, views, spans, strict=True):
        r, toward_cam = view_matrix(name)
        p = v @ r.T  # x right, y up, z toward the camera
        tri = p[f]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        if (np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]]))).sum() < 0:
            n = -n
        shade = 0.55 + 0.4 * np.clip(n[:, 2], 0, 1)
        order = np.argsort(tri[:, :, 2].mean(axis=1))
        ax.add_collection(
            PolyCollection(
                tri[order][:, :, :2],
                facecolors=[(s, s, s * 1.02) for s in shade[order].clip(0, 1)],
                edgecolors="none",
                alpha=0.9,
            )
        )
        for q in pts:
            pos = np.array(q["position_m"]) @ r.T
            nrm = np.array(q["normal"]) @ r.T
            vis = nrm[2] > -0.15
            ax.scatter(
                pos[0],
                pos[1],
                s=26 if vis else 5,
                c=GROUPS[q["bv"].split("_")[0]],
                alpha=1.0 if vis else 0.25,
                edgecolors="white" if vis else "none",
                linewidths=0.5,
                zorder=3 if vis else 2,
            )
        ax.set_xlim(*span)
        ax.set_ylim(-0.02, float(v[:, 1].max()) + 0.05)
        ax.set_aspect("equal")
        ax.set_title(
            {"front": "front (+z toward viewer)", "three_quarter": "3/4 view", "side": "side", "back": "back"}[name],
            fontsize=10,
        )
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in GROUPS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(GROUPS), fontsize=8, frameon=False)
    ext = v.max(axis=0) - v.min(axis=0)
    fig.suptitle(
        f"{folder.name}: bbox {ext[0] * 100:.1f} x {ext[1] * 100:.1f} x {ext[2] * 100:.1f} cm; "
        f"{len(pts)} arrangement points "
        "(+x = wearer's left, +y up, +z front; metres)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out = out or folder / "preview.png"
    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(render(Path(d)))
