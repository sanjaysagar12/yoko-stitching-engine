# avatar_engine

Body measurements (JSON) to an SMPL-family avatar, plus CLO3D-style **bounding volumes** and **arrangement points**.
Metres, y up, feet on y = 0, **+z front, +x the wearer's left**, zero pose (T-pose). CPU only; a build takes about 2 s.

Definitions of the twelve measurements: [docs/measurements.md](../../docs/measurements.md). Design:
[docs/LLD_avatar_alignment_web.md](../../docs/LLD_avatar_alignment_web.md).

## Install and use

```powershell
pip install -e .
avatar-engine build male_173                 # assert/body/male_173.json -> outputs/avatar/male_173
avatar-engine build my_body.json --model-dir D:\models\smplx --out outputs\avatar\mine
```

```python
from avatar_engine import AvatarEngine
result = AvatarEngine().build(body_json_text, model_dir, out_dir)   # the caller reads files, the engine writes them
```

Environment: `AVATAR_ENGINE_MODEL_DIR` (default `models/smplx`), `AVATAR_ENGINE_BODY_DIR` (default `assert/body`).

## Model files

Put the files in `models/smplx/` (or a subfolder such as `smpl/`); nothing is ever downloaded. Searched, case
insensitively and recursively: `SMPLX_<GENDER>.npz` (SMPL-X), `SMPL_<GENDER>.npz|pkl` and the original
`basicmodel_{f,m}_lbs_*.pkl` / `basicModel_neutral_*.pkl` (SMPL). `model_type` (`auto` | `smpl` | `smplx`) selects;
`auto` prefers SMPL-X. A missing model fails with `ModelFilesNotFoundError` listing everything searched. If there is no
file for the gender, `neutral` is used and a warning says so.

* `.npz` files are opened with `allow_pickle=False` and only the arrays needed are read. The `.pkl` twins of SMPL-X and
  the zip are ignored.
* An SMPL `.pkl` is converted once to an `.npz` cache (`<model dir>/.avatar_cache`) with a restricted unpickler that
  stubs `chumpy`, so `chumpy` is not needed. **This path is tested with a synthetic pickle only; no real SMPL files were
  available.**
* The model is loaded with numpy: `v = template + shapedirs @ betas`, `J = J_regressor @ v` (zero pose makes skinning
  the identity). No `torch` and no `smplx` package.
* The model is **symmetrised** about x = 0 at load time (asymmetric shape components are removed), so `_L` and `_R`
  points mirror each other.

SMPL / SMPL-X are non-commercial licensed. The model sits behind the `BodyModel` protocol
(`bodymodel.py`), so replacing it only needs another implementation of `rest(betas)`, `faces`, `skinning_weights`,
joint names and the mirror maps.

## Input: `assert/body/<name>.json` (`contracts/body_measurements.schema.json`)

```json
{
  "schema_version": 1, "name": "male_173", "gender": "male", "units": "cm",
  "measurements": {"height": 173, "chest_circ": 100, "waist_circ": 88, "hip_circ": 100},
  "options": {"num_betas": 30, "exact_circumferences": true, "model_type": "auto"}
}
```

Any subset of `height, neck_circ, shoulder_width, chest_circ, waist_circ, hip_circ, upper_arm_circ, wrist_circ,
thigh_circ, arm_length, inseam, neck_back_to_waist` may be given. `units`: `cm`, `mm` or `m`. `num_betas` defaults to
**30** (10 is a valid option); the report states how many components were actually used (the model file may provide
fewer than requested).

## Pipeline

1. **Shape fit** (`fitting.py`): bounded (+-5) L2-regularised least squares over the betas, started at zero, fully
   deterministic. Height and lengths are fitted tightly; circumferences loosely (see below).
2. **Ring correction** (`ringcorrect.py`, `exact_circumferences`): each circumference ring is scaled radially around its
   local axis, smoothly along the body part, until its length equals the target to 0.01 mm. Warns above +-10 % scale.
3. **Volumes** (`volumes.py`): one capsule per body part, from the vertices the skinning weights assign to it.
4. **Points** (`points.py` + `default_points.yaml`): rays from the volume axes to the surface, plus an offset.

Why circumferences are mostly left to step 2: the shape components put extra girth where the training bodies carry it
(a larger chest is mostly deeper). Ring correction scales isotropically, so changing `chest_circ` by 10 cm moves the
front of the chest by about 1.6 cm. The report lists errors before and after the correction.

## Outputs

`avatar.obj` (metres), `avatar_bundle.json`, `bounding_volumes.json`, `arrangement_points.json`,
`measurements_report.json` (target / before / after / errors per measurement). Floats are written with fixed precision,
so two builds give byte-identical files.

## Arrangement points (`default_points.yaml`)

Defined by bounding volume, angle around its axis, `t` along it (CLO's "Y"), offset and wrap direction; never by
coordinates, so they follow any body. 80 by default: `Head|Neck|Chest|Waist|Hip` x `Front|Back|Left|Right`,
`Chest|Waist|Hip` x `FrontSide|BackSide` x `_L|_R` (angles 40 and 140 degrees on the left, 320 and 220 on the right; for
pieces that sit on the front or back panel boundary), and `UpperArm|Forearm|Hand|Thigh|Shin|Foot` x
`Front|Back|Outer|Inner` x `_L|_R`.

* Central volumes (axis pointing up): angle 0 = front, 90 = left (+x), 180 = back, 270 = right.
* Limb volumes (axis pointing away from the torso, arms in the T-pose along +-x): 0 = front, 180 = back,
  **Outer = 270 and Inner = 90 on the left side** (for an arm "outer" is the top surface, as it would be once the arm
  hangs down); the right side is generated mirrored (angle 360 - a).
* `position_m` includes `offset_m` (default 0.03 m, limbs 0.02 m) along the surface normal, so it is exactly
  `offset_m` from the avatar surface (tested to 1 mm). `up` points up the body (toward the torso for limbs);
  `tangent = up x normal`.

## Known limits

Zero pose only (T-pose); no hands or face fitting; the arm length and shoulder width definitions are approximate
(acromion by position); SMPL on real files is untested; a fit with all 300 SMPL-X components is slow (finite
differences).
