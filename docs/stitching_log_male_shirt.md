# Stitching loop: male shirt

`assert/stitchings/male_shirt.stitching.json`, 29 seams. Output of each iteration in `build/stitching/male_shirt/`.

## Reading the pieces (before iteration 1)

* `suggest` found the mirror joins, the side seams, the back fish-eye dart, yoke to back; it cannot find the armhole
  (cap against front + yoke + back), the neck (stand against front + yoke), the cuff (gathered) or the pocket.
* The alignment puts the pattern's own pieces (FrontPanel, BackPanel, Yoke, CollarBase, CollarTop) on the wearer's
  left (+x) and their `*_L` mirrors on the right; FullSleeve_L, Cuff_L and the `*_L` plackets are on the left. Seam
  suffixes `_L` / `_R` follow the wearer.
* Lengths: armhole front 27.42 + yoke 7.36 + back 25.69 = 60.47 cm against the cap C5 -> C4 60.50 cm; shoulder 18.00
  both; side 43.35 / 43.36; underarm 49.01 both; back top 26.64 (with the 2.9 cm box pleat A34-A25) against the yoke's
  lower edge 23.96 (gather 1.11); sleeve hem 44.92 (two pleats and both sides of the placket slit) against the cuff 29.0
  (gather 1.55); stand lower edge 23.17 against the neckline from mid button stand (front 13.55 + yoke 10.21 = 23.77).
* Decisions: the fronts' button stands A32-A33 and the cuff ends are closures, sewn shut for the simulation (CF as
  `join_mirror`) or left free (cuff ends); the pocket and its flap have no outline on the front: `attach` seams onto the
  front's surface; the sleeve slit's right side (C18 -> C29) runs on into 6.13 cm of hem, and the stand ends mid button
  stand (CF, 2.75 cm in from A32): `trim_end_m` / `trim_start_m`. The cap's top C lands 2.9 cm before the shoulder
  point A23 along the armhole (uniform arc mapping); accepted.

## Iterations (synthetic avatar)

| iter | what was wrong | fix |
|---|---|---|
| 0 (probe) | side seam with b reversed: 1683 crossings, hint "probably reverse" (as intended); "through body" 88 segments, 10 cm deep, on a side seam clearly outside the body | through-body test used vertex normals; the synthetic body is wound inward: switched to `assembler.verify.Surf` (exact signed distance) |
| 1 | `CollarBase_L` chain `[D13, D10]` ambiguous: D3/D13 coincide and the mirror lists them in the other order | resolver: points coinciding with a neighbour do not count when matching a chain to one way exactly |
| 1 | ignore_edges ids of the two fronts swapped (`A32-A20` / `A20-A32`) | ignore by chain `[A32, A20]` |
| 1 | `placket_top` end "1.5 mm off": the verifier interpolated on the outline vertices (chord of a wrapped straight edge) | verifier checks trimmed ends against the `world_edges` samples |
| 1 | shoulder and neck gaps up to 49 cm (yoke on the back, fronts on the front) | `allow_far` on both, with the reason in the label |
| 2 | armhole and neck "twisted" in both orientations (589 / 687, 259 / 79 crossings) | the cap rings the T-pose arm, the stand reaches CB -> side only: segments of the rotated layout cross although the seams run the same way round. Added the sewing rule (pieces on opposite sides of a join) as the orientation test, and `allow_crossings: "<reason>"` for such layouts |
| 2 | plackets: 0 crossings but the sewing rule fails | a placket is sewn and folded over the slit: kind `fold` (same side); E1 -> E |
| 3, 4 | pass: 0 errors, 0 judged crossings, 0 sewing-rule failures, 0 unstitched edges, 50 ignored edges (hems, cuff ends and lower edge, placket outer edges, stand front ends, collar outer edges, pocket mouth, flap free edges, button stand tops) | all close-ups looked at |
| 5 | plackets became `allow_far` after the SMPL-X run (below); synthetic still passes | |

Warnings left (report only): sewing segments through the body on the shoulders (over the neck, 2.5 cm), underarms
(1.8 cm) and plackets (1 cm); the cuff gather 1.55 > 1.5.

## SMPL-X (male_173)

| iter | what was wrong | fix |
|---|---|---|
| real 1 | top plackets 31-35 cm from the sleeve slit (`seam.gap`): placed at their own forearm point, the real arms are longer | `allow_far` on both plackets, with the reason in the label |
| real 2 | pass: 0 errors, 0 judged crossings, 0 sewing-rule failures, 0 unstitched edges; every close-up looked at, all as on the synthetic body | |
