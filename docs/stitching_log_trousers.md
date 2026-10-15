# Stitching loop: trousers

`assert/stitchings/trousers.stitching.json`, 23 seams.

## Reading the pieces

* Wearer's left (+x): Front_L, Back, PocketFront_L, PocketBack_L, WaistBand1; right: Front_with_pocket, Back_R,
  PocketFront, PocketBack, Waistband2. Not in the assembly: PocketMouth, PocketPanel, Front_Pleats, Front_no_pocket.
* Inseam front CrotchPointFront -> A7 -> A20 -> A24 -> A57 84.23 cm, back CrotchPointBack -> A8 -> A21 -> A25 -> A58
  84.21 cm.
* Slant pocket: the front is cut away along Pocket1 -> Pocket2. The pocket back piece (`PocketBack`) is the side piece
  behind the opening: its side A54 -> A53 (19.23 cm) continues the outseam from the waist down to the front's A46.
  Outseam: pocket back A54 -> A53 + front A46 -> hem = 108.65 cm against back BackSide -> hem 108.64 cm.
* The bag front's facing edge A43 -> A53 (20.0 cm) goes to the front's opening Pocket1 -> Pocket2 -> A46 (19.76 cm);
  the two bag halves join round A53 -> A59 -> A51 (24.25 cm both); the bag back's inner edge lies over the bag front:
  `attach`.
* Darts: back two per side with equal legs (11.18 and 8.0 cm; the CB one is a zero-width notch, A39 = A40); front one
  per side at the CF with legs 6.0 / 5.5 cm as drafted (eased: gather 1.1).
* Rise: CF (crotch + zip fly, sewn shut for the simulation) and CB as mirror joins. The fly shield (`Detail`) is
  attached onto the right front along B -> B1.
* Waistbands: band 1 runs CF B11 -> CB B2 round the left, band 2 CF B9 -> CB B7 round the right. Waist per side: front
  FrontCenter -> A26 and A32 -> Pocket1, pocket back A43 -> A54, back BackSide -> A37, A38 -> A39, A40 -> CB (darts
  closed first) = 52.8 cm. Band 1 52.0 cm; band 2 56.0 cm of which the first 4 cm at the CF are an underlap extension.

## Iterations (synthetic avatar)

| iter | what was wrong | fix |
|---|---|---|
| 1 | pocket facing and pocket bag broke the sewing rule | a facing is turned to the inside and a bag is two layers sewn round their edge: both pieces end on the same side. Kind `fold` (same side in the sewing rule); the shirt plackets became `fold` too |
| 1 | inseam gap 42-44 cm | the leg panels lie on the front and back of the thigh; `allow_far` with the reason |
| 2 | waistbands: 44 crossings (754 reversed) | the CF line continues 1.7 cm above the waist to FrontCenter: the CF seam now ends at FrontCenter and the waist starts there |
| 3 | waistband_R ratio 1.06 (band 56.0 cm, waist 52.8 cm) | the band's 4 cm CF underlap: `trim_start_m` 0.04, the rest of that edge in ignore_edges |
| 3 | 30 / 39 waistband crossings left, all at the outline's jogs (FrontCenter -> A26 back up toward the CF, A43 -> A52 up 1 cm) and at the step from the front (layer 0) to the pocket back piece (layer 2) | `allow_crossings` with that reason (sewing rule judged) |
| 4 | pass: 0 errors, 0 judged crossings (61 reported), 0 sewing-rule failures, 0 unstitched edges, 32 ignored | all close-ups looked at |
| 5 | outseams given `allow_crossings` after the SMPL-X run (below); synthetic still passes | |

## SMPL-X (male_173)

| iter | what was wrong | fix |
|---|---|---|
| real 1 | one crossing per outseam (9737 reversed): exactly at the part boundary, where side a steps 6 cm from the pocket back piece (layer 2, A53) to the front (layer 0, A46) | `allow_crossings` on the outseams with that reason (sewing rule judged) |
| real 2 | pass: 0 errors, 0 judged crossings (29 reported), 0 sewing-rule failures, 0 unstitched edges; inseams up to 51 cm apart (`allow_far`); every close-up looked at | |
