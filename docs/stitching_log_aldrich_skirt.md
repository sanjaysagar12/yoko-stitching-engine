# Stitching loop: Aldrich skirt

`assert/stitchings/aldrich_skirt.stitching.json`, 10 seams.

## Reading the pieces

* The job (`assert/jobs/aldrich_skirt.json`) has only the two skirt blocks: no waistband, facing or zip piece. The
  waist edge is therefore free (`ignore_edges`, reason given); a waistband needs a piece in the job and the assembly.
* The alignment puts the pattern's blocks on the wearer's right (-x) and their mirrors (`Skirt_Front_R`,
  `Skirt_Back_R`) on the left.
* Side seams: front A16 -> A16e -> A16b -> A7 -> A8c 60.07 cm, back A10 -> A10b -> A7 -> A8a 60.06 cm.
* The darts are not cut into the outline: the waist only carries the dart caps (A17b-A17e-A17a, A14b-A12a-A14a,
  A13b-A11b-A13a, legs of about 1 cm). Their apexes are interior named points (front A18, back A13 and A14, 12-14 cm
  below the waist). Added `line` sides (straight polylines through named points, put onto the placed surface) to sew
  each dart leg to its apex: front 12.06 / 12.06, back 14.04 / 14.04 and 13.04 / 13.04 cm.
* `suggest` did not find the CF / CB mirror joins (their mirror edge passes A6 / A6a): fixed in `suggest` (the shorter
  way round between the mirror edge's ends).

## Iterations

| iter | result |
|---|---|
| 1 (synthetic) | pass at once: 0 crossings, 0 sewing-rule failures, 0 unstitched edges, 28 ignored (waist, hems). Close-ups: darts converge to their apexes, CF / CB ladders, side seams joined waist to hem (5 cm apart at the waist, 21 cm at the flared hem) |
| 2 (synthetic) | the same after the engine changes made for the trousers (`fold`, `allow_crossings`) |
| real 1, 2 (SMPL-X female_165) | pass at once; side seams 4.9-20.1 cm apart; every close-up looked at |
