# Tunnel score feasibility (2025, sequence pairs)

Regular-season pitches with a previous pitch in the same PA: 524,357. Commit point 175 ms before the current pitch reaches the plate; tracking limit 100 ms.

## 1. Noise floor and free variation

|                |   d_dec_deg p10 |   d_dec_deg p50 |   d_dec_deg p90 |   d_track_in p10 |   d_track_in p50 |   d_track_in p90 |   d_plate_in p10 |   d_plate_in p50 |   d_plate_in p90 |
|:---------------|----------------:|----------------:|----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|
| different type |            0.77 |            2.06 |            4.39 |             5.07 |            13.09 |            24.51 |             7.28 |            18.58 |            34.34 |
| same type      |            0.6  |            1.59 |            3.17 |             4.19 |            10.95 |            21.19 |             5.43 |            14.2  |            27.31 |

Share of commit-point separation (different-type pairs) predictable from other inputs, held-out R²:

|                                 |    R² |
|:--------------------------------|------:|
| types + both plate locations    | 0.706 |
| + plate separation (d_plate_in) | 0.714 |

## 2. Raw relationship (different-type pairs, no model)

**By decile of d_dec_deg**

| bin              |   pitches |   swing% |   whiff/swing% |   CSW% |
|:-----------------|----------:|---------:|---------------:|-------:|
| (0.00311, 0.772] |     33954 |     64.6 |           21.4 |   24.5 |
| (0.772, 1.133]   |     33954 |     62   |           22.1 |   24.7 |
| (1.133, 1.446]   |     33954 |     59   |           23   |   24.7 |
| (1.446, 1.748]   |     33954 |     56.8 |           23.7 |   24.5 |
| (1.748, 2.065]   |     33954 |     54.1 |           24   |   24   |
| (2.065, 2.417]   |     33954 |     51.4 |           25.2 |   23.6 |
| (2.417, 2.843]   |     33954 |     48.4 |           26.1 |   23.3 |
| (2.843, 3.417]   |     33954 |     46.3 |           27.1 |   22.8 |
| (3.417, 4.389]   |     33954 |     43.7 |           27.8 |   22.6 |
| (4.389, 42.007]  |     33954 |     42.7 |           27.6 |   24.4 |

**By decile of d_track_in**

| bin              |   pitches |   swing% |   whiff/swing% |   CSW% |
|:-----------------|----------:|---------:|---------------:|-------:|
| (0.0498, 5.07]   |     33954 |     65.7 |           21.9 |   27.8 |
| (5.07, 7.379]    |     33954 |     63.7 |           22.2 |   27.4 |
| (7.379, 9.346]   |     33954 |     61.8 |           22.8 |   27   |
| (9.346, 11.215]  |     33954 |     60   |           22.9 |   26.3 |
| (11.215, 13.091] |     33954 |     57.5 |           23.4 |   25.4 |
| (13.091, 15.102] |     33954 |     54.8 |           24.5 |   24.5 |
| (15.102, 17.376] |     33954 |     50.9 |           25.1 |   23.5 |
| (17.376, 20.216] |     33954 |     46.2 |           27.7 |   22.5 |
| (20.216, 24.509] |     33954 |     40.1 |           28.9 |   19.9 |
| (24.509, 83.911] |     33954 |     28.4 |           32.8 |   14.7 |

**Whiff/swing % by early (commit-point) x late (tracking-limit) separation terciles**

| dec3        |   late close |   late mid |   late far |
|:------------|-------------:|-----------:|-----------:|
| early close |         21.2 |       23.6 |       24.6 |
| early mid   |         22   |       24   |       29.1 |
| early far   |         26.4 |       24.5 |       29.7 |

## 3. Held-out lift and permutation null (train Mar-Jul, test Aug-Oct)

Log-loss improvement x 1000 over the base model. Null = tunnel features shuffled within pitcher x pitch type, 5 draws.

| outcome       |   test n |   base log-loss |   +tunnel (no prev) |   +prev |   +prev +tunnel | null +prev +tunnel (mean, max)   |
|:--------------|---------:|----------------:|--------------------:|--------:|----------------:|:---------------------------------|
| whiff | swing |    92493 |          0.4678 |               -0.11 |    1.39 |            0.86 | 0.52, 1.00                       |
| swing         |   173377 |          0.4347 |                0.55 |    2.14 |            1.73 | 1.28, 1.59                       |
| CSW           |   173377 |          0.4788 |               -0.65 |    0.18 |           -0.05 | -0.35, -0.04                     |

## 4. Pitcher x pitch-type level (whiff | swing)

|                                                             | value       |
|:------------------------------------------------------------|:------------|
| pitcher x type groups (>=75 swings per half)                | 447         |
| SD of mean tunnel value, whiff pct points (>=150 swings)    | 0.7         |
| split-half r, tunnel value (Spearman-Brown)                 | 0.88 (0.93) |
| split-half r, residual whiff% (Spearman-Brown)              | 0.37 (0.54) |
| tunnel value in one half vs residual whiff% in the other, r | 0.015       |

**Largest pitcher x pitch-type tunnel values (>=150 swings), whiff pct points**

|                |   n |   tv |
|:---------------|----:|-----:|
| (605135, 'CU') | 188 | 6.69 |
| (808967, 'CU') | 170 | 3.48 |
| (571510, 'CH') | 331 | 2.15 |
| (624133, 'CU') | 164 | 1.81 |
| (445276, 'FC') | 309 | 1.71 |
| (682052, 'SL') | 225 | 1.43 |
| (664208, 'CU') | 203 | 1.42 |
| (669920, 'CH') | 211 | 1.36 |
| (608337, 'CH') | 265 | 1.35 |
| (607259, 'CH') | 224 | 1.32 |

