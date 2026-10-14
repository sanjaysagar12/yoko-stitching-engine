# trousers: iteration log (real avatar male_173)

Baseline (HEAD a305694): `Front_with_pocket` on `Thigh_Front_L` (roll): `anchor auto -> root` slid it 48 cm down the
leg, y -0.47..0.63 (below the floor), invalid (STANDOFF). `Back` on `Thigh_Back_L`: y -0.35..0.77, **valid=True** (no
floor check). On `Hip_Left`: the 1.1 m panel centred on the hip point reaches the T-pose arms, `clear_radius` grows the
torso ring to R = 0.48 m to get the arms out of its footprint: the "thin strip beside the hip" of the screenshot.

Engine findings 2 and 3: new validity kinds FLOOR (piece below the floor clearance), OFF_REGION (far from its point),
REGISTRATION (registered point > 1.5 cm off its level), ORIENTATION; `Back` on the thigh is now FLOOR-invalid.

Fit facts: pattern WaistLIne -> bottom 1.08 m, WaistLIne -> HeelLine 1.07 m (`height_waist_side` 107 cm, crotch to
floor 83 cm: drafted to the floor); body waist 1.091 m, crotch 0.751 m. Pattern rise (waist to crotch line) 24 cm vs
body 34 cm (SMPL's natural waist is high).

## iter_01 (4 panels: Front_with_pocket + mirror on the legs' fronts, Back + mirror on the backs, WaistLIne at Waist)
- verify FAIL (16): hems at y 0.006 / 0.010 (floor 3 cm); fronts 4.4 mm apart and backs 11 mm apart at the crotch;
  orientation 6.8 deg on all; singular values 0.3 % on the fronts.
- images: legs as wide tubes (R 0.21-0.24) around each leg, waist at the waist, hems on the floor; front panels
  touching at the centre front/crotch (front view, top view: the two front arcs meet at x = 0).
- hypotheses: (a) floor: the pattern is full length, it cannot keep 3 cm off the floor with its waist line at the
  natural waist; (b) gap at the crotch: `pair_slide` keeps a side pair apart from x = 0 but measures only local row
  y = 0 ("the axis has no x component") - wrong for a leg axis, which leans 4.5 deg sideways; (c) 6.8 deg: verifier
  compared with the thigh volume (9 deg), the ring is the whole leg (4.5 deg); (d) singular values: sliver triangles
  of the self-crossing front outline (the engine's own size check: 0.0014 %).
- changes: engine `_pair_clear` maps every sample when the axis leans (engine finding 4); verifier: whole-limb axis,
  slivers (min altitude < 0.5 mm) left out of the per-triangle Jacobian check; `anchor.offset_m` (declared
  registration offset, also in `expect.registration`); JSON: waist line worn 2.5 cm above the natural waist (fit
  decision, reported: the trousers are about 2 cm too long for this body barefoot).

## iter_02
- verify: only `engine.honesty` (engine invalid: NOT_AT_POINT with limit max_shift on both fronts, slide 9 cm of 25).
- engine finding 5: stage 1 of the solver caps slides at 5 cm and records `max_shift` when it reaches that cap; later
  stages only clear it on pieces that still move, so a converged piece kept a stale limit -> false NOT_AT_POINT. Fixed:
  each solver stage starts with `max_shift` cleared on its moving pieces.
- images / flat outlines (`build/_probe/flat.png`): the drafted `Front_with_pocket` has its crotch point at +x, so on a
  front point (piece +x -> world +x) it is a RIGHT-leg front; `Back` (piece +x -> world -x on a back point) is a
  left-leg back. iter_01/02 had the front on the wrong leg (crotch outward), which is why it had to slide 9 cm.
- change: front on `Thigh_Front_R`, its mirror `Front_L` on `Thigh_Front_L`.

## iter_03
- verify PASS (fast mode). Images: fronts with the crotch toward the centre, backs too, left/right symmetric, hems
  3.2 cm off the floor, wide leg tubes clear of the feet; sections: four arcs, two per leg, never crossing x = 0.

## iter_04 (+ pockets via the new `over` tool, waistbands, Detail)
- engine tool: `over: {piece, point}` lays a piece on another one's surface with a shared named point coinciding (A47
  on the fronts and the pocket pieces), layering lifts it out.
- verify FAIL: region 36-38 cm on PocketMouth/PocketPanel (layers 3 and 4 at R 0.33-0.37), Detail 6.3 deg off
  upright (flat on the belly normal), WaistBand1 bottom 2.5 cm off its level, NOT_CONCENTRIC warnings for the pocket
  layers, engine honesty mismatch.
- images: the four pocket pieces stack into a 4-layer fan at each hip (top view, front_left); waistbands a ring above
  the trouser tops; Detail a small strip at centre front.
- hypotheses: (a) engine and verifier disagree on the region of a laid-over piece (engine skipped it, verifier used
  the arrangement point 33 cm below) -> region of an `over` piece = its host; (b) a band rolled on the slightly tilted
  torso ring is not level: the engine registered "the" lowest outline vertex (an arbitrary corner), the verifier the
  lowest world point -> both now use one defined edge point (extreme height, nearest the middle across); (c) 4 layers
  of pocket bags and facings in one spot is clutter, not information.
- changes: the above (engine + verifier), JSON: pocket bags only (front and back bag per side), Detail rolled.

## iter_05 (fast)
- verify FAIL (2): NOT_CONCENTRIC warnings: the back pocket bag (layer 2 over the front) lies beside the back panel at
  a larger radius - by design.
- images (front_left, sections): pockets at the outer hip of each front (A47 is near the side seam), legs as tubes
  around each leg, front + back of a leg never cross x = 0 (sections at waist, hip, crotch, knee, ankle), hems 3 cm off
  the floor, waistbands as a ring just above the trouser tops. Crotch line 12.5 cm above the body crotch: reported
  (pattern rise 24 cm vs body 34 cm).
- change (JSON): pocket layers declared (front bag 1, back bag 2).

## iter_06 (full)
- crash in replay_reversed: a pocket arrives before its host: `over` raised UnknownPieceError -> engine finding: a
  missing host is a warning (OVER_PENDING), the piece stays on its point.
- rerun: primary and replay_reversed PASS; replay (listed order): engine invalid, verifier valid (honesty).
- engine finding 7: a carried laid-over piece measured its slide from its arrangement point (33 cm, NOT_AT_POINT):
  `_over` now sets the anchor of a placed piece too (its position kept).
- engine finding 8: a carried piece had lost why it sits at its radius (the body clearance radius is only computed
  for new pieces; recomputed at the final position it is smaller, because the solver pushes out before it slides and
  never comes back in): its standoff turned into STANDOFF_EXCEEDED as soon as another piece was added, without moving.
  The answer now carries `solved.needed_radius_m` and the engine reuses it; a layered piece is explained by its ring's
  innermost piece plus its layers.

## iter_07 (full)
- primary, replay PASS; replay_reversed: engine NOT_AT_POINT on the pockets (32-36 cm).
- hypothesis: in reverse order the pockets are placed (on their thigh points) before the fronts; when the front comes,
  the carried pocket's anchor becomes the host position and its unchanged place reads as a 36 cm slide.
- engine finding 9: a laid-over piece whose host is new is solved again onto its host (like the placed half of a pair
  whose other half is new).

## iter_08 (full): PASS in arrange, replay, replay_reversed; carry 0 mm; deterministic. 910 s.

Engine findings 10 and 11 (from the orientation tests, after iter_08): `tilt` broke on Hip points because every
`Hip_*` arrangement point had its `up` pointing to the floor (avatar_engine; the Hip volume runs pelvis -> crotch).
That is also part of the original report: the trouser panel on `Hip_Left` was upside down as well as far out. Fixed
in avatar_engine 0.2.1 (central points' up is up); the verifier's torso reference is now world-up. The verifier's
inside test and nearest-triangle search were made robust (global orientation by signed volume; candidates by bounding
spheres of triangles split to 2 cm), found on the synthetic body.

## iter_09 (final code, full): PASS in arrange, replay, replay_reversed; carry 0 mm; deterministic (same fingerprint
as iter_08: the avatar fix does not touch leg or waist points).
- images: legs as tubes around each leg (R 0.24-0.26 at the crotch row), fronts with the crotch curve toward the
  centre, left/right mirror images, never crossing x = 0 at waist / hip / crotch / knee / ankle (sections); hems 3.2 cm
  (fronts) and 3.98 cm (backs) above the floor, clear of the feet; pockets bags (layers 1, 2) over the outer hip of each
  front; both waistbands as one ring just above the trouser tops; the fly piece at centre front; heat map all green.
- fit data: WaistLIne 2.2-2.9 cm above the natural waist (declared +2.5 cm: the pattern is full length), CrotchLine
  12-13 cm above the body crotch (pattern rise 24 cm, body 34 cm), HipLine 15.5 cm above the body hip level, KneeLine
  5.7-6.4 cm above the knee joints, HeelLine 2.1-2.8 cm below the ankle joints.

## iter_10, iter_11 (confirmations): PASS everywhere, same fingerprint as iter_09 (babc161b7d15b5d4).
