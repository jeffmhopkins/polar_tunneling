# Contact quality in the sequencer (pre-registered, single run)

Test pitches: 401,479 from 265 pitchers, hitters with >= 200 pitches in 2025. The run-value pick differs from the CSW pick on 45.9% of them.

## Primary: run-value pick vs CSW pick where they differ

97,780 pitches where the picks differ and the pitch thrown was one of the two (57% were the run-value pick).

Within-count realized run value (xwOBA on contact), RV pick minus CSW pick: **-0.27 runs per 100 pitches** (95% CI -0.48 to -0.08). Negative is fewer runs. Pass: **True**.

| count   |   pitches |   runs per 100, RV pick minus CSW pick |
|:--------|----------:|---------------------------------------:|
| 0-0     |     21998 |                                  -0.03 |
| 0-1     |     13542 |                                  -0.48 |
| 0-2     |      7074 |                                  -0.43 |
| 1-0     |      9920 |                                  -0.58 |
| 1-1     |     10335 |                                   0.02 |
| 1-2     |      8806 |                                  -0.02 |
| 2-0     |      4216 |                                  -1.03 |
| 2-1     |      6418 |                                  -0.16 |
| 2-2     |      6378 |                                  -0.53 |
| 3-0     |       399 |                                  -0.22 |
| 3-1     |      3418 |                                  -1.47 |
| 3-2     |      5276 |                                   0.56 |

## Secondary

- Actual delta_run_exp: -0.38 runs per 100 (95% CI -0.67 to -0.09).
- CSW: -5.21 pts (95% CI -5.93 to -4.41).
- xwOBA on balls in play: -0.026 (95% CI -0.038 to -0.014).
- Picks differ on 45.9% of test pitches.
- 2026 R^2 of xwOBA on balls in play: 0.0324 with all features vs 0.0296 with CONTROLS + SPEED only.

