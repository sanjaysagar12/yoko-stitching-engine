# aldrich_skirt: iteration log (real avatar female_165)

Baseline (before the loop, HEAD a305694): `A - Skirt Front` on `Waist_Front` (roll) hangs y 0.779..1.377 with the waist
point at 1.074: the piece is **centred** on its point (bbox-centre reference), top 30 cm above the waist; `valid=True`.
Image `build/_probe/img_front.png`: the skirt covers the chest down to mid thigh (the user's screenshot).

Engine finding 0 (avatar_engine): female_165 no longer built at HEAD (`PointConfigError: Shin_Front_L ... axis distance
0.038 m, radius 0.077 m`): the point sanity floor of 0.5 volume radii is wrong for a bony side (shin front; the radius
is the calf). Fixed: floor = offset + 0.2 radii.

Engine finding 1 (alignment_engine): pieces had no way to hang from a body level. New: `levels.py` (Waist = narrowest
torso girth between spine1/spine2, Hip = first girth from the pelvis down that reaches the measured hip girth, Crotch =
the inseam landmark, Knee/Ankle joints, Neck base, Shoulder joints) and `anchor: {level, piece_point | edge, offset_m}`
(registration; the solver keeps a registered piece's height: axial slide weight 1e4).

## iter_01
- JSON: front/back halves on `Waist_Front` / `Waist_Back` (roll), each with a mirror clone across its fold edge (CF
  A2-A4, CB A1-A3), reference on the fold, `anchor {Waist, A2|A1}`.
- verify: engine valid, independent FAIL: `orientation.top_up/grain` 11 deg on both back halves.
- images (sheet, front/back/left/right/top, 3/4, sections): waist edge at the waist (1.044), hem at 0.45 (knee 0.444),
  four pieces on one ring, gaps at CF, CB and the side seams; top view: concentric. The diagonal edges in the side
  views are the flared side seams, not a tilt. Nothing inside the body (heat map all green, amber on the buttocks).
- hypothesis: the verifier compares with the point's `up`, which on the lower back is tilted by the surface normal; the
  piece follows the (vertical) ring axis, which is right. Verifier bug, not a placement bug.
- change: verifier reference axis = the point's body-volume axis (whole limb for limb points).

## iter_02
- verify: PASS in arrange, replay, replay_reversed; carry 0 mm; deterministic.
- images unchanged (same placement). Run time 173 s, 100 s of it the verifier's exact distances: sped up.

## iter_03
- verify PASS everywhere (128 s). Images identical to iter_02.

Engine work for the trousers and the shirt happened after iter_03 (registration offset, edge point, `over`, solver
limit, carried needed radius, pair angles modulo 360, Hip point up, verifier distance search): the skirt is run again
on the final code.

## iter_04 (final code) and iter_05, iter_06 (confirmations): see below.

## iter_04 (final code, full): PASS in arrange, replay, replay_reversed; carry 0 mm; deterministic.
- images: waist edges (A2 / A1) exactly at the body waist 1.044 m; hem at 0.45 m, just above the knee (0.444); four
  pieces on one concentric ring; gaps at CF, CB (half a gap each side of the centre line) and wide side gaps at the
  waist (the darted waist is narrower than the hip ring); nothing rotated (grain within 5 degrees of vertical); heat map
  green with amber on the buttocks (within 1 cm above the clearance).
- fit data (ring at the waist, 1 cm below): garment 87.8 cm vs body 69.3 cm; at the hip level 102.5 vs 100.9 cm
  (the body hip level is measured where the girth reaches 98 cm, close to the crotch on this SMPL body).

## iter_05, iter_06 (confirmations): PASS everywhere, same fingerprint as iter_04 (bb549d6502815df7).
