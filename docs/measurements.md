# Avatar measurements: what each one measures and where

Compare these definitions with a tape measure before trusting a number. All values are defined **procedurally on the
model template** (no hard-coded vertex indices), so the same definitions work for SMPL (6890 vertices) and SMPL-X
(10475 vertices). Source: `src/avatar_engine/measurements.py`.

## Coordinate convention (also shown in the web UI)

| Axis | Direction |
|---|---|
| **+x** | **the wearer's left** (the viewer's right when looking at the avatar from the front) |
| **+y** | up; the feet are on y = 0 |
| **+z** | forward (the front of the body) |

Unit: metres. Pose: the model's zero pose (a T-pose). The model is symmetrised about x = 0 at load time (shape
components that make the body left/right asymmetric are removed), so `_L` and `_R` points mirror each other.

## How a measurement is evaluated

* **Circumference** = length of a closed loop of points lying on mesh edges (`Ring`), found **once** on the template by
  slicing the mesh with a plane at a landmark level and keeping the loop that encloses the body axis. Each loop point
  is `v[a] + t * (v[b] - v[a])` for a fixed edge `(a, b)`, so for any body shape the loop is the same band of the mesh,
  and its length is the sum of the straight segments between the loop points. For the template this equals the
  planar slice length; it is a polygon through surface points, so a tape measure on the real surface can differ by a
  few millimetres.
* **Length** = straight distance between two fixed landmarks, or the length of a fixed polyline of mesh vertices.
* Levels that say "narrowest" or "widest" are searched on the template (41 levels between two joints) and then frozen.

## The twelve measurements

| Name | Definition | Template value (SMPL-X male, 179.4 cm) |
|---|---|---|
| `height` | Highest vertex above the floor (top of the head, standing, zero pose). | 179.4 cm |
| `neck_circ` | Narrowest horizontal girth of the neck, searched between 25 % and 75 % of the way from the `neck` joint to the `head` joint. | 40.5 cm |
| `shoulder_width` | Straight line between the left and right **acromion points**: the highest vertex within 2 cm (in x/z) of the point 3.5 cm outside the shoulder joint. It is the top of the shoulder, not the arm tip. | 39.8 cm |
| `chest_circ` | Horizontal girth of the torso half way between the `spine2` and `spine3` joints (bust level, below the armpits, so the arms are not in the loop). | 100.6 cm |
| `waist_circ` | Narrowest horizontal torso girth between the `spine1` and `spine2` joints (natural waist, about 65 % of the height). | 92.5 cm |
| `hip_circ` | Widest horizontal girth between the `pelvis` joint and 3 cm above the crotch (hips and buttocks). | 106.4 cm |
| `upper_arm_circ` | Girth of the left upper arm half way between the shoulder and elbow joints, in the plane perpendicular to the arm. | 32.4 cm |
| `wrist_circ` | Narrowest girth of the left forearm between 60 % and 100 % of the way from the elbow to the wrist joint, perpendicular to the forearm. | 17.4 cm |
| `thigh_circ` | Girth of the left thigh 35 % of the way from the hip to the knee joint, perpendicular to the leg (upper thigh, below the gluteal fold). | 58.6 cm |
| `arm_length` | Left arm: acromion point to the elbow joint to the wrist joint (two straight segments). | 53.9 cm |
| `inseam` | Height of the crotch above the floor. The crotch is the lowest vertex on the body mid line between the knees and the pelvis (the perineum). | 80.1 cm |
| `neck_back_to_waist` | Length along the back mid line surface (a fixed chain of mid-line vertices) from C7 (the neck joint + 2 cm) down to the level of the waist ring. | 35.6 cm |

Limb circumferences (`upper_arm_circ`, `wrist_circ`, `thigh_circ`) are measured on the left limb; the right limb is
fitted to the same value (the model is symmetric).

Typical tape-measure differences: a tape measure follows the skin and is read at a landmark you find by touch. If a
real person's `waist_circ` was measured at the navel (lower than the narrowest level) it will read larger than this
definition; adjust the level in `build_definitions` if your measuring protocol differs.

## How the avatar reaches the numbers

1. **Shape fit.** `scipy.optimize.least_squares` over the shape coefficients (betas, default 30; 10 is an option),
   bounded to +-5, started at zero, with a ridge penalty of 0.3 per beta. Height and lengths are fitted tightly (0.5 mm
   and 3 mm per residual unit). Circumferences are fitted loosely (5 cm per unit) on purpose, see below.
2. **Ring correction** (`exact_circumferences: true`). Each circumference ring is scaled radially around its local
   axis (vertical for the torso and neck, the limb axis for arms and legs) with a smooth `cos^2` falloff along the
   axis and a skinning-weight mask for the body part, until its length equals the target (to 0.01 mm). A warning is
   issued if a ring needs more than +-10 % radial scale. Limb rings are corrected on both sides with the same scale.

Why circumferences are left to the ring correction: the shape components put extra girth wherever the training bodies
carry it (a larger chest is mostly deeper, front to back). Ring correction scales the surface isotropically around the
body axis, so changing `chest_circ` by 10 cm moves the front of the chest by about 10 / (2 pi) = 1.6 cm, which is
predictable and smooth. The report lists the errors before and after correction for every measurement. Lengths that
cross a ring (for example `neck_back_to_waist`) can drift by a couple of millimetres during the correction, and are
reported after it.

`measurements_report.json` holds, per measurement: `target_cm`, `pre_correction_cm`, `post_correction_cm`,
`pre_error_cm`, `post_error_cm` and, for rings, `ring_radial_scale`.
