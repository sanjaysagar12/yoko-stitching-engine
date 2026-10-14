# verify: trousers iter_09

**overall: PASS**

| run | engine valid | independent | fails | checks |
|---|---|---|---|---|
| primary | True | True | 0 | 216 |
| replay | True | True | 0 | 219 |
| replay_reversed | True | True | 0 | 218 |

| global check | ok | value |
|---|---|---|
| carry.moves | True | 0.0 |
| determinism | True | babc161b7d15b5d4 |

## worst offenders (primary)

| id | piece | value | limit |
|---|---|---|---|
| clearance.body | Waistband2 | 0.030316 | 0.03 |
| clearance.body | WaistBand1 | 0.041556 | 0.03 |
| clearance.body | Front_with_pocket | 0.048199 | 0.03 |
| clearance.body | Front_L | 0.048203 | 0.03 |
| clearance.pieces | Back | Back_R | 0.030297 | 0.03 |
| clearance.pieces | Front_with_pocket | Front_L | 0.030316 | 0.03 |
| clearance.pieces | PocketFront | PocketBack | 0.030334 | 0.03 |
| clearance.pieces | PocketFront_L | PocketBack_L | 0.030334 | 0.03 |
| clearance.floor | Front_with_pocket | 0.031178 | 0.03 |
| clearance.floor | Front_L | 0.03118 | 0.03 |
| clearance.floor | Back_R | 0.039815 | 0.03 |
| clearance.floor | Back | 0.039823 | 0.03 |

## registration and level offsets (primary)

| piece | line | level | y | body y | offset |
|---|---|---|---|---|---|
| Front_with_pocket | HeelLine | Ankle | 0.0463 | 0.0744 | -0.0281 |
| Front_with_pocket | WaistLIne | Waist | 1.1131 | 1.0911 | 0.022 |
| Front_with_pocket | CrotchLine | Crotch | 0.8738 | 0.7515 | 0.1223 (check) |
| Front_with_pocket | HipLine | Hip | 0.9535 | 0.7982 | 0.1554 (check) |
| Front_with_pocket | KneeLine | Knee | 0.5448 | 0.488 | 0.0568 |
| Front_L | HeelLine | Ankle | 0.0463 | 0.0744 | -0.0281 |
| Front_L | WaistLIne | Waist | 1.1131 | 1.0911 | 0.022 |
| Front_L | CrotchLine | Crotch | 0.8738 | 0.7515 | 0.1223 (check) |
| Front_L | HipLine | Hip | 0.9536 | 0.7982 | 0.1554 (check) |
| Front_L | KneeLine | Knee | 0.5448 | 0.488 | 0.0568 |
| Back | HeelLine | Ankle | 0.0531 | 0.0744 | -0.0213 |
| Back | WaistLIne | Waist | 1.1199 | 1.0911 | 0.0288 |
| Back | CrotchLine | Crotch | 0.8806 | 0.7515 | 0.1291 (check) |
| Back | HipLine | Hip | 0.9604 | 0.7982 | 0.1622 (check) |
| Back | KneeLine | Knee | 0.5516 | 0.488 | 0.0636 |
| Back_R | HeelLine | Ankle | 0.0531 | 0.0744 | -0.0213 |
| Back_R | WaistLIne | Waist | 1.1199 | 1.0911 | 0.0287 |
| Back_R | CrotchLine | Crotch | 0.8806 | 0.7515 | 0.1291 (check) |
| Back_R | HipLine | Hip | 0.9603 | 0.7982 | 0.1621 (check) |
| Back_R | KneeLine | Knee | 0.5516 | 0.488 | 0.0636 |
| PocketFront | WaistLIne | Waist | 1.1114 | 1.0911 | 0.0203 |
| PocketFront | HipLine | Hip | 0.9519 | 0.7982 | 0.1537 (check) |
| PocketFront_L | WaistLIne | Waist | 1.1114 | 1.0911 | 0.0203 |
| PocketFront_L | HipLine | Hip | 0.9519 | 0.7982 | 0.1537 (check) |
| Front_with_pocket | registration | | | | -0.003047 (pass) WaistLIne at 1.113 m, Waist at 1.091 m |
| Front_L | registration | | | | -0.003045 (pass) WaistLIne at 1.113 m, Waist at 1.091 m |
| Back | registration | | | | 0.003756 (pass) WaistLIne at 1.120 m, Waist at 1.091 m |
| Back_R | registration | | | | 0.003747 (pass) WaistLIne at 1.120 m, Waist at 1.091 m |
| WaistBand1 | registration | | | | 0.0 (pass) bottom at 1.166 m, Waist at 1.091 m |
| Waistband2 | registration | | | | 0.00041 (pass) bottom at 1.167 m, Waist at 1.091 m |

## ease (primary, informational)

| ring | level | y | garment | body | ease | coverage |
|---|---|---|---|---|---|---|

## engine warnings (primary)

- PocketBack: SIDE_UNLIKELY: 'PocketBack' looks like a back piece but is placed on Thigh_Front_R
- PocketBack_L: SIDE_UNLIKELY: 'PocketBack_L' looks like a back piece but is placed on Thigh_Front_L
