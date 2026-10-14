# verify: aldrich_skirt iter_04

**overall: PASS**

| run | engine valid | independent | fails | checks |
|---|---|---|---|---|
| primary | True | True | 0 | 78 |
| replay | True | True | 0 | 82 |
| replay_reversed | True | True | 0 | 78 |

| global check | ok | value |
|---|---|---|
| carry.moves | True | 0.0 |
| determinism | True | bb549d6502815df7 |

## worst offenders (primary)

| id | piece | value | limit |
|---|---|---|---|
| clearance.body | A - Skirt Back | 0.030667 | 0.03 |
| clearance.body | Skirt_Back_R | 0.030676 | 0.03 |
| clearance.body | Skirt_Front_R | 0.045918 | 0.03 |
| clearance.body | A - Skirt Front | 0.045947 | 0.03 |
| clearance.pieces | A - Skirt Front | Skirt_Front_R | 0.030573 | 0.03 |
| clearance.pieces | A - Skirt Back | Skirt_Back_R | 0.030573 | 0.03 |
| clearance.floor | A - Skirt Front | 0.449704 | 0.03 |
| clearance.floor | Skirt_Front_R | 0.449704 | 0.03 |
| clearance.floor | A - Skirt Back | 0.460709 | 0.03 |
| clearance.floor | Skirt_Back_R | 0.460709 | 0.03 |

## registration and level offsets (primary)

| piece | line | level | y | body y | offset |
|---|---|---|---|---|---|
| A - Skirt Front | registration | | | | -0.0 (pass) A2 at 1.044 m, Waist at 1.044 m |
| Skirt_Front_R | registration | | | | -0.0 (pass) A2 at 1.044 m, Waist at 1.044 m |
| A - Skirt Back | registration | | | | -0.0 (pass) A1 at 1.044 m, Waist at 1.044 m |
| Skirt_Back_R | registration | | | | -0.0 (pass) A1 at 1.044 m, Waist at 1.044 m |

## ease (primary, informational)

| ring | level | y | garment | body | ease | coverage |
|---|---|---|---|---|---|---|
| skirt | Waist | 1.0341 | 0.8779 | 0.6925 | 0.1853 | 0.661 |
| skirt | Hip | 0.7751 | 1.0251 | 1.0086 | 0.0165 | 0.773 |

## engine warnings (primary)

- none
