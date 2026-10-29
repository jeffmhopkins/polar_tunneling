# Arsenal tunnel check (2026, vs primary fastball)

1,882 pitcher x pitch type x side groups (>= 50 pitches Mar-Jul), 332,520 pitches. Group features from Mar-Jul; test on Aug-Oct.

Log-loss gain x 1000 of +arsenal tunnel over stuff + location + count + fastball differentials:

| outcome       |   test n |   gain x1000 |   null mean |   null max | share of nulls beaten   |
|:--------------|---------:|-------------:|------------:|-----------:|:------------------------|
| whiff | swing |    44746 |        -0.04 |       -0.3  |       0.06 | 90%                     |
| swing         |    93972 |         0.34 |        0.19 |       0.4  | 90%                     |
| CSW           |    93972 |         0.21 |       -0.35 |      -0.08 | 100%                    |

Group level (359 groups with >= 40 Aug-Oct swings): rank correlation of Mar-Jul tunnel features with Aug-Oct residual whiff/swing (tunneling predicts negative for a_dec_deg):

|            |   Spearman r |
|:-----------|-------------:|
| a_dec_deg  |        0.011 |
| a_track_in |       -0.005 |
| a_plate_in |        0.042 |

Mean residual whiff/swing (pct points) by quartile of commit-point angle to the fastball:

| dec_q    |   groups |   resid_whiff_pct |
|:---------|---------:|------------------:|
| closest  |       90 |             -1    |
| 2        |       90 |             -0.04 |
| 3        |       89 |              0.28 |
| farthest |       90 |             -1.21 |

