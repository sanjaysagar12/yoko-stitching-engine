"""Runs the cloth simulation inside ZOZO's Contact Solver's bundled Python (it needs the solver's ``frontend``):

    <solver>/python/python.exe ppf_runner.py <scene.npz> <params.json> <out_dir>

Imports only numpy and the solver: it is started by ``simulator.solver`` in a subprocess and talks back through
files and stdout. Every event is one JSON line on stdout (``{"event": "stage" | "progress" | "done", ...}``); the
newest garment frame is kept in ``<out_dir>/latest.npy`` while the solver runs (the viewer shows the sewing live).
A file ``<out_dir>/cancel`` stops the solver. At the end ``<out_dir>/frames.npz`` holds the kept frames of the
garment (metres, the scene's vertex order) and ``closure`` the largest and mean stitch gap of every frame.

Scene: the garment is one Baraff-Witkin shell with the stitches as six-column rows and the flat pattern as its UV
(the warp / weft frame); interfaced pieces get a spatial map of ``bend``; the body is a pinned (fixed) collider.
Run: a sewing phase with inactive momentum (the cloth does not keep its velocity, so the stitches pull the panels
together calmly) and a 1 ms step, then normal dynamics with ``dt_after`` until the last frame.
"""

import json
import os
import sys
import time
from typing import Any

import numpy as np


def emit(**kw: Any) -> None:
    print(json.dumps(kw), flush=True)


def gaps(x: Any, si: Any, sw: Any) -> tuple[float, float]:
    pa = np.einsum("kj,kjd->kd", sw[:, :3], x[si[:, :3]])
    pb = np.einsum("kj,kjd->kd", sw[:, 3:], x[si[:, 3:]])
    g = np.linalg.norm(pa - pb, axis=1)
    return float(g.max()), float(g.mean())


def main(scene_path: str, params_path: str, out: str) -> int:
    from frontend import App
    from frontend._utils_ import Utils

    d = np.load(scene_path)
    p = json.load(open(params_path, encoding="utf-8"))
    V, F, UV = d["V"], d["F"], d["UV"]
    SI, SW = d["SI"].astype(np.int64), d["SW"].astype(np.float64)
    emit(event="stage", stage="building the solver scene", vertices=int(len(V)), triangles=int(len(F)))
    app = App.create(p["app"], cache_dir=p["cache_dir"])
    app.asset.add.tri("garment", V.astype(np.float64), F.astype(np.int64))
    app.asset.add.tri("body", d["BV"].astype(np.float64), d["BF"].astype(np.int64))
    app.asset.add.stitch("seams", (SI, SW))
    scene = app.scene.create()
    g = scene.add("garment")
    for k, v in p["garment"].items():
        g.param.set(k, v)
    g.stitch("seams")
    g.set_uv([t for t in UV[F].astype(np.float32)])
    stiff = d["STIFF"]
    if stiff.max() > 0 and p.get("stiff_bend"):
        g.set_param_spatial("bend", stiff, float(p["stiff_bend"]))
    body = scene.add("body")
    for k, v in p["body"].items():
        body.param.set(k, v)
    body.pin()
    scene = scene.build()
    gi = np.asarray(scene._map_by_name["garment"])  # the garment's vertices in the solver's output order

    session = app.session.create(scene)
    sp = session.param
    sp.set("frames", int(p["frames"])).set("dt", 1e-3).set("inactive-momentum", True)
    sp.dyn("inactive-momentum").time(float(p["sew_time"])).hold().change(False)
    sp.dyn("dt").time(float(p["sew_time"])).hold().change(float(p["dt_after"]))
    session = session.build()
    emit(event="stage", stage="simulating")
    t0 = time.time()
    session.start(blocking=False)
    last, cancelled = -1, False
    latest = os.path.join(out, "latest.npy")
    while session.is_running():
        if os.path.exists(os.path.join(out, "cancel")):
            Utils.terminate()
            cancelled = True
            break
        got = session.get.vertex()
        if got is not None and got[1] != last:
            last = int(got[1])
            x = got[0][gi]
            tmp = latest + ".tmp.npy"
            np.save(tmp, x.astype(np.float32))
            os.replace(tmp, latest)
            gmax, gmean = gaps(x.astype(np.float64), SI, SW)
            emit(
                event="progress",
                frame=last,
                frames=int(p["frames"]),
                wall_s=round(time.time() - t0, 1),
                gap_max_m=round(gmax, 5),
                gap_mean_m=round(gmean, 5),
            )
        time.sleep(0.5)
    wall = time.time() - t0
    finished = (not cancelled) and session.finished()
    keep = int(p.get("keep_every", 4))
    end = session.get.latest_frame()
    frames, idx, closure = [], [], []
    for k in range(0, end + 1):
        got = session.get.vertex(k)
        if got is None:
            continue
        x = got[0][gi].astype(np.float64)
        gmax, gmean = gaps(x, SI, SW)
        closure.append({"frame": k, "gap_max_m": round(gmax, 6), "gap_mean_m": round(gmean, 6)})
        if k % keep == 0 or k == end:
            frames.append(x.astype(np.float32))
            idx.append(k)
    if frames:
        np.savez_compressed(os.path.join(out, "frames.npz"), X=np.stack(frames), frame=np.asarray(idx))
    log = []
    if not finished and not cancelled:
        try:
            log = list(session.get.log.stderr(40)) + list(session.get.log.stdout(40))
        except Exception as exc:  # noqa: BLE001 - the log is informative only
            log = [f"log unavailable: {exc}"]
    try:
        tpf = [n for _, n in session.get.log.numbers("time-per-frame")]
    except Exception:  # noqa: BLE001
        tpf = []
    if p.get("delete_session", True) and frames:  # the frames are copied out; the solver keeps sessions in its tree
        try:
            session.delete()
        except Exception:  # noqa: BLE001 - a session left behind only costs disk space
            pass
    emit(
        event="done",
        finished=bool(finished),
        cancelled=cancelled,
        frames=end,
        wall_s=round(wall, 1),
        ms_per_frame=round(float(np.mean(tpf)), 1) if tpf else None,
        closure=closure,
        log=log[-80:],
    )
    return 0 if finished else (3 if cancelled else 1)


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
