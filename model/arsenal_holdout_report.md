# Arsenal tunnel check, 2025 -> 2026

1,379 pitcher x pitch type x side groups in both seasons; 245,134 2026 pitches.

Year-to-year correlation of the group tunnel features: a_dec_deg 0.85, a_track_in 0.63, a_plate_in 0.68

Log-loss gain x 1000 on 2026 of +2025 arsenal tunnel over stuff + location + count + fastball differentials (model trained on 2025):

| outcome       |   2026 n |   gain x1000 |   null mean |   null max | share of nulls beaten   |
|:--------------|---------:|-------------:|------------:|-----------:|:------------------------|
| whiff | swing |   117474 |         0.1  |       -0.38 |      -0.12 | 100%                    |
| swing         |   245134 |         0.51 |       -2.46 |      -2.22 | 100%                    |
| CSW           |   245134 |         0.19 |       -1.23 |      -1.1  | 100%                    |

Group level (1065 groups with >= 40 swings in 2026): rank correlation of 2025 tunnel features with 2026 residual whiff/swing (tunneling predicts negative for a_dec_deg):

|            |   Spearman r |
|:-----------|-------------:|
| a_dec_deg  |       -0.07  |
| a_track_in |       -0.045 |
| a_plate_in |       -0.107 |

Mean 2026 residual whiff/swing (pct points) by quartile of 2025 commit-point angle to the fastball:

| dec_q    |   groups |   resid_whiff_pct |
|:---------|---------:|------------------:|
| closest  |      267 |             -3.55 |
| 2        |      266 |             -2.75 |
| 3        |      266 |             -4.07 |
| farthest |      266 |             -4.77 |

Closest minus farthest quartile: 1.22 pct points (95% bootstrap interval 0.20 to 2.23).

