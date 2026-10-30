# Arsenal tunnel check (2025, vs primary fastball)

1,768 pitcher x pitch type x side groups (>= 50 pitches Mar-Jul), 315,598 pitches. Group features from Mar-Jul; test on Aug-Oct.

Log-loss gain x 1000 of +arsenal tunnel over stuff + location + count + fastball differentials:

| outcome       |   test n |   gain x1000 |   null mean |   null max | share of nulls beaten   |
|:--------------|---------:|-------------:|------------:|-----------:|:------------------------|
| whiff | swing |    42936 |         0.41 |        0.02 |       0.32 | 100%                    |
| swing         |    88905 |        -0.36 |       -0.21 |       0.04 | 20%                     |
| CSW           |    88905 |        -0.23 |       -0.09 |       0.33 | 30%                     |

Group level (364 groups with >= 40 Aug-Oct swings): rank correlation of Mar-Jul tunnel features with Aug-Oct residual whiff/swing (tunneling predicts negative for a_dec_deg):

|            |   Spearman r |
|:-----------|-------------:|
| a_dec_deg  |       -0.081 |
| a_track_in |        0.044 |
| a_plate_in |        0.06  |

Mean residual whiff/swing (pct points) by quartile of commit-point angle to the fastball:

| dec_q    |   groups |   resid_whiff_pct |
|:---------|---------:|------------------:|
| closest  |       91 |              1.54 |
| 2        |       91 |              1.11 |
| 3        |       91 |             -0.71 |
| farthest |       91 |             -0.04 |

