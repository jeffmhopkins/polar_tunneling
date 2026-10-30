# Perceptual tunnel model: 2025 -> 2026 (pre-registered, single run)

Train 705,020 pitches (2025), test 710,809 (2026).

Feature coverage in 2026 (share non-missing): tau_b 1.00, miss_x_in 1.00, miss_z_in 1.00, miss_t_ms 1.00, ang_speed 1.00, ang_speed_vs_fb 1.00, sep_rate_vs_fb 1.00, post_fb_commit 0.99, post_true_commit 0.97, post_fb_release 0.99, post_true_release 0.97, surprise_count 1.00, surprise_prev 0.74, fb_seen_pa 1.00, sep_last_fb 0.59, n_priorpa_thisgame 1.00, pitch_number 1.00

## Primary endpoints and ablations (log-loss gain x 1000 over M0, 2026)

|                         | whiff | swing                  | swing                             |
|:------------------------|:-------------------------------|:----------------------------------|
| 2026 n                  | 339082                         | 710809                            |
| M0 log-loss             | 0.4524                         | 0.4509                            |
| M1 gain x1000           | 1.83                           | 5.86                              |
| +P1 timing              | -0.01                          | 0.37                              |
| +P2 fastball-prior miss | 0.36                           | 1.14                              |
| +P3 optical motion      | 0.9                            | 4.4                               |
| +P4 ambiguity           | 0.6                            | 0.43                              |
| +P5 expectation         | 0.36                           | 0.61                              |
| +P6 at-bat build-up     | 0.95                           | 1.53                              |
| null gains              | 0.02, 0.08, -0.07, 0.05, -0.03 | -0.22, -0.07, -0.47, -0.33, -0.21 |
| pass                    | True                           | True                              |

## Secondary endpoint

607 pitcher x pitch type groups with >= 75 swings per half. Split-half r of P value: 0.94. P value in one half vs M0 residual whiff/swing in the other: r = -0.019 (pass if >= 0.10).

## M1 (whiff | swing) gain importance, % of total

|                    |   % gain |
|:-------------------|---------:|
| ang_speed          |     4.94 |
| tau_b              |     2.45 |
| miss_x_in          |     1.38 |
| miss_z_in          |     1.2  |
| ang_speed_vs_fb    |     0.91 |
| surprise_prev      |     0.83 |
| sep_rate_vs_fb     |     0.82 |
| sep_last_fb        |     0.8  |
| post_true_commit   |     0.66 |
| post_fb_commit     |     0.64 |
| surprise_count     |     0.51 |
| post_true_release  |     0.51 |
| miss_t_ms          |     0.49 |
| post_fb_release    |     0.45 |
| n_priorpa_thisgame |     0.28 |
| pitch_number       |     0.22 |
| fb_seen_pa         |     0.18 |

All perceptual features together: 17.3% of total gain.

