# Speed check: perceptual signal with stronger speed controls (2025 -> 2026, single run)

Speed controls added to M0: plate_speed_mph, flight_ms, prev_dv, prev_dtf_ms, fb_dtf_ms.

Log-loss gain x 1000 on 2026. Rows marked (expl.) are exploratory and not part of pass/fail.

|                                 | whiff | swing                | swing                             |
|:--------------------------------|:-----------------------------|:----------------------------------|
| 2026 n                          | 339082                       | 710809                            |
| M0s log-loss                    | 0.4525                       | 0.4505                            |
| M1s gain x1000                  | 2.14                         | 5.67                              |
| share of original gain          | 117%                         | 97%                               |
| +P1 timing                      | 0.3                          | 0.05                              |
| +P2 fastball-prior miss         | 0.86                         | 1.13                              |
| +P3 optical motion              | 1.61                         | 4.02                              |
| +P4 ambiguity                   | 0.52                         | 0.35                              |
| +P5 expectation                 | 0.93                         | -0.3                              |
| +P6 at-bat build-up             | 0.75                         | 0.82                              |
| null gains                      | 0.54, 0.39, 0.38, 0.19, 0.31 | -0.80, -0.53, -0.78, -0.67, -0.67 |
| pass                            | True                         | True                              |
| (expl.) +kinematics over M0s    | 2.45                         | 6.07                              |
| (expl.) +P over M0s+kinematics  | 0.89                         | 2.01                              |
| (expl.) +P3 over M0s+kinematics | -0.01                        | 0.72                              |
