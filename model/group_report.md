# Group-reference read: 2025 -> 2026 (pre-registered, single run)

> How every number below is computed is in [How the model works](#how-the-model-works), and every variable is defined in [Variables](#variables).

Train 705,020 pitches (2025), test 710,809 (2026).

Feature coverage in 2026 (share non-missing): gmiss_x_in 0.99, gmiss_z_in 0.99, gmiss_t_ms 0.99, gspread_in 0.99, miss_x_in 1.00, miss_z_in 1.00, miss_t_ms 1.00

Median in 2026 by pitch type (swings), group read vs fastball read:

| pitch_type   |      n |   abs_fbmiss_in |   abs_gmiss_in |   gspread_in |   whiff_pct |
|:-------------|-------:|----------------:|---------------:|-------------:|------------:|
| CH           |  40056 |            2.23 |           1.33 |         1.73 |       31.38 |
| CU           |  18278 |            5.28 |           2.77 |         2.02 |       30.55 |
| FC           |  28232 |            2.35 |           1.44 |         1.7  |       22.83 |
| FF           | 107582 |            0.48 |           1.06 |         1.56 |       21.81 |
| FS           |  11792 |            2.62 |           1.25 |         1.66 |       34.33 |
| KC           |   5286 |            5.37 |           2.79 |         2.16 |       33.92 |
| SI           |  54372 |            0.69 |           1.48 |         1.76 |       13.58 |
| SL           |  45256 |            3.72 |           1.8  |         1.81 |       33.37 |
| ST           |  26605 |            4.8  |           2.49 |         2.07 |       31.21 |
| SV           |   1327 |            5.23 |           2.54 |         2.14 |       29.92 |

## Endpoints (log-loss gain x 1000, 2026)

Primary: whiff | swing, M1 + G over M1. Pass needs >= 0.3 and above all 5 nulls. The swing column and the head-to-head and swap rows are secondary.

|                              | whiff \| swing               | swing                            |
|:-----------------------------|:-----------------------------|:---------------------------------|
| 2026 n                       | 339082                       | 710809                           |
| M1 log-loss                  | 0.4505                       | 0.4451                           |
| M1 + G gain x1000            | 0.52                         | 0.18                             |
| M0 + P2 gain x1000 (over M0) | 0.36                         | nan                              |
| M0 + G gain x1000 (over M0)  | 0.91                         | nan                              |
| M1 with G for P2 gain x1000  | 0.37                         | nan                              |
| null gains                   | 0.26, 0.03, 0.11, 0.27, 0.13 | -0.33, -0.26, 0.06, -0.06, -0.31 |
| pass                         | True                         | False                            |


## Reading

- **Primary: passed.** Adding the group read to the full perceptual model (M1) lowered 2026 whiff-per-swing
  log-loss by 0.52 per 1000. All 5 nulls were lower (largest 0.27), so roughly half of it is above what
  shuffled values of the same features give.
- **Head to head, the group read is the better reference.** On its own over M0 it adds 0.91 per 1000,
  against 0.36 for the fastball read (P2). Swapping it in for P2 inside M1 adds 0.37.
- **Swing (secondary): did not pass.** 0.18 per 1000, under the 0.3 bar, though above every null. The
  group read is about whether a swing misses, not whether the hitter swings.
- For breaking balls and offspeed pitches, the group read lands about half as far from the actual
  crossing as the fastball read (sliders 1.8 in vs 3.7 in). The miss that's left is what the hitter
  couldn't know at commit from the look-alikes.
- **Scale.** This is a small forecast gain, similar to the other single perceptual blocks (0.4 to 1.0).
  It has not been tested in the sequencer, and every earlier perceptual block improved forecasts without
  changing picks. The gains are observational: pitchers and catchers choose pitches too.

## How the model works

Same data, controls, M0 and M1 as the [perceptual model](perceptual_report.md#how-the-model-works).
With $`T`$ the flight time, $`\tau_b`$ the hitter's commit time and $`t_c = \max(T - \tau_b, 0.05)`$:

1. **Types.** For each pitcher x batter side, the types with >= 20 pitches in the season. Each gets its mean
   acceleration $`\mathbf a_k`$, and its cloud of eye-frame angles $`(\alpha, \epsilon)`$ 175 ms before
   arrival: mean $`\boldsymbol\mu_k`$, covariance $`\Sigma_k + 0.1^2 I`$, prior $`\pi_k`$ = usage share
   (the P4 observer). Built from trajectories and labels only, within each season.
2. **Group weights.** For a pitch seen at $`\mathbf x`$,
   $`w_k = \pi_k \mathcal N(\mathbf x; \boldsymbol\mu_k, \Sigma_k + 0.01 I) / \sum_j \pi_j \mathcal N(\mathbf x; \boldsymbol\mu_j, \Sigma_j + 0.01 I)`$.
   With one qualifying type, $`w = 1`$.
3. **Each member's read.** From the actual $`\mathbf p(t_c)`$ and $`\mathbf v(t_c)`$, $`s_k`$ solves
   $`p_y + v_y s + \tfrac12 a_{k,y} s^2 = y_p`$, and
   $`\hat x_k = p_x + v_x s_k + \tfrac12 a_{k,x} s_k^2`$, $`\hat z_k = p_z + v_z s_k + \tfrac12 a_{k,z} s_k^2`$.
4. **Group read.** $`\bar x = \sum_k w_k \hat x_k`$, likewise $`\bar z`$, $`\bar s`$.

```math
\text{gmiss\_x\_in} = 12\, s_b (x_p - \bar x), \quad \text{gmiss\_z\_in} = 12 (z_p - \bar z), \quad
\text{gmiss\_t\_ms} = 1000 (\tau_b - \bar s)
```

```math
\text{gspread\_in} = 12 \sqrt{\textstyle\sum_k w_k \big[(\hat x_k - \bar x)^2 + (\hat z_k - \bar z)^2\big]}
```

For a pitch the observer is sure is the primary fastball, this is P2's fastball read (r = 0.9998 on 2025
pitches with `post_fb_commit` > 0.99).

**Models.** LightGBM with `feasibility.PARAMS`, 400 rounds, trained on 2025 and scored on 2026 once. Gain
x1000 is $`1000(L_{\text{without}} - L_{\text{with}})`$ on 2026. Nulls refit M1 + G with the four G columns
permuted within pitcher x pitch type in train and test (seed 11).

## Variables

| Variable | Meaning | Units | Source |
|---|---|---|---|
| `pitcher` | MLBAM id of the pitcher | id | Statcast column `pitcher` (`datamining/outputs`) |
| `ax_ft_s2` | Constant acceleration of the fit, x ($`a_x`$) | ft/s² | Statcast column `ax` (`datamining/outputs`) |
| `ay_ft_s2` | Constant acceleration of the fit, y ($`a_y`$, positive: drag) | ft/s² | Statcast column `ay` (`datamining/outputs`) |
| `az_ft_s2` | Constant acceleration of the fit, z ($`a_z`$, gravity plus lift) | ft/s² | Statcast column `az` (`datamining/outputs`) |
| `plate_x_ft` | Horizontal position where the pitch crosses the front of the plate ($`x_p`$, catcher's view) | ft | Statcast column `plate_x` (`datamining/outputs`) |
| `plate_z_ft` | Height where the pitch crosses the front of the plate ($`z_p`$) | ft | Statcast column `plate_z` (`datamining/outputs`) |
| `y_p` | Depth of the front edge of the plate, 17/12 ft, where `plate_x_ft`/`plate_z_ft` are measured | ft | `reconstruct_trajectory.PLATE_FRONT_Y` |
| `tf` | Flight time $`T`$ from release to the front of the plate. Pitches with $`T \ge 0.6`$ s are dropped | s | `features.add_state` |
| `p(t)` | Ball position $`t`$ seconds after release, $`\mathbf r_0 + \mathbf v_r t + \tfrac12 \mathbf a t^2`$ | ft | `features.position` |
| `E` | Batter's eye: $`(s_b \cdot (-35.8),\ -28.3,\ 65.0)`$ in, i.e. 35.8 in off the plate's center line on the batter's side, 28.3 in behind the back of the plate, 65 in high | ft | `features.EYE_*`, `to_polar.ORIGIN_*` |
| `pitch_type` | Statcast pitch type code: FF four-seam, SI sinker, FC cutter, SL slider, ST sweeper, CU curve, KC knuckle curve, SV slurve, CH changeup, FS splitter, FO forkball, KN knuckleball. PO (pitchout), EP (eephus), FA (position players), CS (slow curve), SC (screwball) and UN (unknown) are dropped | category | Statcast column `pitch_type` (`datamining/outputs`) |
| `stand` | Batter side, R or L. Sets the eye position and the sign $`s_b`$ | category | Statcast column `stand` (`datamining/outputs`) |
| `s_b` | Batter-side sign: +1 for a right-handed batter, -1 for a left-handed one. Multiplying an x value by it makes + mean away from the batter | ±1 | `stand` |
| `tau_b` | P1. Hitter's commit time from bat tracking: $`0.175 \times \tilde\rho_b/\bar\rho`$, clipped to 0.125-0.225 s, where $`\rho`$ = swing length / bat speed (swings over 40 mph), $`\bar\rho`$ the league mean and $`\tilde\rho_b`$ the batter's mean shrunk to it with 50 prior swings. 0.175 s if he has none | s | `perceptual.batter_tau` |
| `t_c` | Commit time after release, $`T - \tau`$ (floored at 0 in the geometric features, at 0.05 s in the perceptual ones) | s | `features.add_features`, `perceptual.commit_features` |
| `v(t)` | Ball velocity $`t`$ seconds after release, $`\mathbf v_r + \mathbf a t`$ | ft/s | `perceptual.velocity` |
| `miss_x_in` | P2. Horizontal error of a fastball read: actual plate crossing minus where the ball would cross if, from the commit point, it kept the primary fastball's acceleration. Batter-oriented ($`s_b`$), + is away | in | `perceptual.commit_features` |
| `miss_z_in` | P2. Vertical error of the same fastball read; + means the pitch finished higher than the read | in | `perceptual.commit_features` |
| `miss_t_ms` | P2. Arrival-time error: actual time left at commit ($`\tau_b`$) minus the fastball read's time to the plate. + means the pitch arrives later than a fastball would | ms | `perceptual.commit_features` |
| `post_fb_commit` | P4. Posterior probability, from the ball's direction 175 ms before arrival, that the pitch is the primary fastball (Gaussian observer over the pitcher x side's types) | probability | `perceptual.posteriors` |
| `post_true_commit` | P4. Posterior probability, from the same view, of the pitch's true type | probability | `perceptual.posteriors` |
| `M0` | Control model: LightGBM on CONTROLS = BASE + PREV + fastball differentials + `arm_angle_deg` + batter rates | - | `perceptual.CONTROLS` |
| `M1` | M0 plus all 17 perceptual features (P1-P6) | - | `perceptual.ALL_P` |
| `LightGBM` | Gradient-boosted trees. Settings everywhere unless stated: binary log-loss objective, learning rate 0.05, 63 leaves, >= 200 rows per leaf, 80% of features and 80% of rows per tree, 400 trees, seed 7 | - | `feasibility.PARAMS`, `feasibility.ROUNDS` |
| `log-loss` | $`-\frac1n\sum_i [y_i\log\hat p_i + (1-y_i)\log(1-\hat p_i)]`$ on held-out pitches; lower is better | nats per pitch | `sklearn.metrics.log_loss` |
| `gain x1000` | $`1000\,(L_{\text{without}} - L_{\text{with}})`$: how much adding the features lowers held-out log-loss. + helps | nats per 1000 pitches | computed in each script |
| `null` | The same model refit after shuffling the tested features among pitches of the same pitcher x pitch type (or group), which keeps each feature's spread but breaks its link to the pitch | - | `feasibility.shuffled`, `perceptual.shuffled` |
| `null gains` | Gain x1000 of each shuffled-feature refit; the real gain has to beat all of them | nats per 1000 pitches | computed in each script |
| `abs_fbmiss_in`, `abs_gmiss_in` | Distance at the plate between the actual crossing and the fastball read (`miss_x_in`, `miss_z_in`) or the group read (`gmiss_x_in`, `gmiss_z_in`) | in | `group_read.main` |
| `whiff_pct` | Whiffs per swing | % | `group_read.main` |
| `M1 log-loss` | 2026 log-loss of M1 | nats per pitch | `group_read.main` |
| `pass` | Pre-registered pass rule met (stated in How the model works) | True/False | computed in the script |
| `w_k` | G. Posterior weight of type $`k`$ from the ball's direction 175 ms before arrival: the P4 Gaussian observer's full posterior over the pitcher x side's types (>= 20 pitches each) | probability | `group_read.group_features` |
| `a_k` | G. Mean acceleration of type $`k`$ for the pitcher x side, within season | ft/s² | `group_read.group_features` |
| `x_hat_k`, `z_hat_k`, `s_k` | G. Where the ball would cross the front of the plate, and when, if from the commit point it kept type $`k`$'s acceleration | ft, ft, s | `group_read.group_features` |
| `gmiss_x_in` | G. Horizontal miss of the group read: actual crossing minus the weighted mean of `x_hat_k`. Batter-oriented, + away | in | `group_read.group_features` |
| `gmiss_z_in` | G. Vertical miss of the group read; + means the pitch finished higher than the read | in | `group_read.group_features` |
| `gmiss_t_ms` | G. Arrival-time miss of the group read: $`\tau_b`$ minus the weighted mean of `s_k`. + means later than the read | ms | `group_read.group_features` |
| `gspread_in` | G. Weighted spread of the group's crossings around the group read | in | `group_read.group_features` |
