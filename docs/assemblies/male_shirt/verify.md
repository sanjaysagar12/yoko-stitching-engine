# verify: male_shirt iter_04

**overall: PASS**

| run | engine valid | independent | fails | checks |
|---|---|---|---|---|
| primary | True | True | 0 | 346 |
| replay | True | True | 0 | 344 |
| replay_reversed | True | True | 0 | 341 |

| global check | ok | value |
|---|---|---|
| carry.moves | True | 0.0 |
| determinism | True | a80f14117d9db615 |

## worst offenders (primary)

| id | piece | value | limit |
|---|---|---|---|
| clearance.body | CollarTop | 0.03641 | 0.03 |
| clearance.body | CollarTop_L | 0.03641 | 0.03 |
| clearance.body | BackPanel | 0.036527 | 0.03 |
| clearance.body | BackPanel_L | 0.036538 | 0.03 |
| clearance.pieces | FrontPanel | Pocket | 0.030299 | 0.03 |
| clearance.pieces | BackPanel | Yoke | 0.030299 | 0.03 |
| clearance.pieces | BackPanel_L | Yoke_L | 0.030299 | 0.03 |
| clearance.pieces | Pocket | PocketFlap | 0.030299 | 0.03 |
| clearance.floor | BackPanel | 0.664265 | 0.03 |
| clearance.floor | BackPanel_L | 0.664265 | 0.03 |
| clearance.floor | FrontPanel | 0.688729 | 0.03 |
| clearance.floor | FrontPanel_L | 0.688729 | 0.03 |

## registration and level offsets (primary)

| piece | line | level | y | body y | offset |
|---|---|---|---|---|---|
| BackPanel | BustLine | Chest | 1.1564 | 1.3309 | -0.1744 (check) |
| BackPanel | Waistline | Waist | 0.9289 | 1.0911 | -0.1622 (check) |
| BackPanel_L | BustLine | Chest | 1.1564 | 1.3309 | -0.1744 (check) |
| BackPanel_L | Waistline | Waist | 0.9289 | 1.0911 | -0.1622 (check) |
| FrontPanel | registration | | | | -0.0 (pass) A22 at 1.431 m, Neck at 1.431 m |
| FrontPanel_L | registration | | | | -0.0 (pass) A22 at 1.431 m, Neck at 1.431 m |
| Yoke | registration | | | | -2.1e-05 (pass) A15 at 1.431 m, Neck at 1.431 m |
| Yoke_L | registration | | | | -2.1e-05 (pass) A15 at 1.431 m, Neck at 1.431 m |

## ease (primary, informational)

| ring | level | y | garment | body | ease | coverage |
|---|---|---|---|---|---|---|
| torso | Chest | 1.3309 | 1.1093 | 1.5088 | -0.3994 | 0.698 |
| torso | Waist | 1.0911 | 1.4105 | 0.8743 | 0.5363 | 0.893 |

## engine warnings (primary)

- ARM_ZONE: FrontPanel reaches 5.9 cm into the space around the left arm; a sleeve there slides outward along the arm
- ARM_ZONE: FrontPanel_L reaches 5.9 cm into the space around the right arm; a sleeve there slides outward along the arm
- ARM_ZONE: BackPanel reaches 1.4 cm into the space around the left arm; a sleeve there slides outward along the arm
- ARM_ZONE: BackPanel_L reaches 1.5 cm into the space around the right arm; a sleeve there slides outward along the arm
