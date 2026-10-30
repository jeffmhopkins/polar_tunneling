# Per-hitter decision point (pre-registered, single run)

Train 705,020 pitches (2025), test 710,809 (2026).

## Fit on 2025

League out-of-fold log-likelihood by tau (higher is better):

|   tau ms |   total log-lik |   vs best |
|---------:|----------------:|----------:|
|      125 |         -457959 |     -37.5 |
|      150 |         -457922 |       0   |
|      175 |         -458221 |    -299.5 |
|      200 |         -458129 |    -207.4 |
|      225 |         -458135 |    -213.3 |

League best tau*: 150 ms.

Fitted tau across 673 hitters: median 155 ms, 10th-90th pct 134-182 ms; hitters with >= 1,000 pitches: 129-194 ms.

Split-half reliability (odd vs even games, 287 hitters with >= 500 pitches in each half): r = 0.04.

Correlation with bat-tracking tau (hitters with >= 1,000 pitches): r = -0.07.

## 2026 test (gain over arm A, current timing)

|                          | whiff per swing                   | swing                             |
|:-------------------------|:----------------------------------|:----------------------------------|
| 2026 log-loss A          | 0.45035                           | 0.44483                           |
| B fitted gain x1000      | -0.64                             | -0.87                             |
| C league tau* gain x1000 | 0.28                              | -0.34                             |
| null gains               | -0.10, -0.56, -0.31, -0.49, -0.08 | -0.70, -0.82, -0.81, -0.45, -1.05 |
| pass                     | False                             | False                             |

**Part 1 passes: False.**

