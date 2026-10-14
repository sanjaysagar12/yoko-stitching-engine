# male_shirt: iteration log (real avatar male_173; regression garment)

Construction (from the flat outlines, `build/_probe/flat.png`): FrontPanel CF = A33-A32 (x = 0, straight, full
height). BackPanel: the straight full-height edge A34-A35 at x = 0.347 is the centre back; x = 0 carries the armhole
(A24 .. A27): `tests/test_layout.py` mirrors the back at x = 0, i.e. across the side seam (wrong for a garment; the
test only checks the solver). Yoke CB = A28-A25 (back neckline A15 .. A28). CollarBase CB D2-D3, CollarTop CB D4-D1.
PlacketTop / PlacketUnder are 15-20 cm long: sleeve plackets (the sleeve has the slit C18). Variants: FullSleeve (not
ShortSleeve), Pocket + PocketFlap (not the round ones); interfacings are inner layers of collar and cuff, left out.

Registration: the shirt's `Waistline` lies 42 cm below the yoke seam; at SMPL's natural waist (1.091) the back panel
top would be at 1.51 m, above the shoulders (1.40): the pattern is drafted on a body with a lower waist. The shirt is
registered by its high shoulder point (front A22, yoke A15) at the neck base; the back follows its drafting heights
(`pattern_origin_m`, datum). This is also why the default level registration (schema 5) applies to bottoms only.

## iter_01 (fast)
- verify FAIL (4): `size.outline_edges` FullSleeve_L 0.3 %, `pairs.mirror_image` FullSleeve | FullSleeve_L 9.2 cm,
  NOT_CONCENTRIC (plackets, pocket flap), engine honesty.
- images: panels on one torso ring, gaps at CF/CB/sides; yoke over the back top (layer 1); collars at the neck;
  sleeves rolled on the arms, cap at the shoulder, hem past the wrist over the hand (the 62.5 cm sleeve is longer than
  the arm: fit data); cuffs at the wrists; pocket + flap on the wearer's left chest; plackets left and right not
  symmetric (right one under the sleeve at layer 0, left one at layer 1).
- hypotheses: (a) the sleeves are exact mirror images (bounding boxes identical to 1 um after x -> -x): the verifier's
  "longest boundary loop" is broken by the slit's pinch vertex and traces different partial loops -> verifier bug;
  (b) the plackets are not recognised as a pair: `_find_pairs` compares the rotations raw, a mirrored quarter turn is
  270, not -90 (engine finding 6); (c) small pieces need declared layers.
- changes: verifier boundary checks on all boundary edges, located on the outline by arc position; engine compares
  angles modulo 360; JSON: layers (cuffs, plackets, pocket 1; flap 2).

## iter_02 (fast): PASS. Images: as iter_01, plackets now one mirror pair on top of the forearms (top view).

## iter_03 (full: arrange, replay, reversed replay, carry, determinism): PASS. 662 s.

## iter_04 (final code, full; yoke layer declared after the synthetic run flagged NOT_CONCENTRIC front | yoke)
- PASS in arrange, replay, replay_reversed; carry 0 mm; deterministic.
- images: as iter_02/03: panels on one torso ring, gaps at CF, CB and the side seams; yoke one layer over the back top;
  collar band and collar at the neck, CB on the back centre; sleeves wrapped on the arms, caps at the shoulders, left
  and right mirror images; cuffs over the sleeve hems at the wrists; plackets on the forearms; pocket and flap on the
  wearer's left chest; heat map all green.
- fit data: A22 / A15 (high shoulder points) at the neck base exactly; the shirt's Waistline 16 cm and BustLine 17 cm
  below SMPL's natural waist / chest levels (reported as check, the shirt is drafted for a longer back); the sleeve
  (62.5 cm) reaches past the wrist over the hand. The torso "ease" at the chest level is not meaningful: the body slice
  at 1.331 m includes the T-pose upper arms.

## iter_05, iter_06 (confirmations): PASS everywhere, same fingerprint as iter_04 (a80f14117d9db615).
