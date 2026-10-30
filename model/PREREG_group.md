# Group-reference read: pre-registration

Written and committed before any outcome is computed for these features. The
2026 test is run once. Every result is reported, including failures.

## Why

Every post-commit feature so far measures the pitch against one reference: the
pitcher's primary fastball (P2 miss, P3 separation rate, P6 last fastball). P4
already asks how much a pitch looks like each of the pitcher's types at the
decision point, but nothing has asked what the hitter should expect *after*
commit given that look. A slider that looks like the pitcher's cutter at commit
is fooling the hitter about the cutter's path, not the four-seamer's.

The question: if the reference is the group of the pitcher's pitches that look
like this one at commit, weighted by how much they look like it, does the
post-commit miss from that group's path predict whiffs better than the miss
from the fastball's path, and beyond everything in the perceptual model?

## Data

- Train: 2025 regular season, test: 2026 regular season, scored once. Same rows
  and controls as `perceptual.py` (`PREREG.md`).
- As in P2 and P4, the per-type clouds and mean accelerations are built within
  each season from trajectories and pitch labels only (no outcomes).

## Group-reference features (G)

For each pitcher x batter side, take the types with >= 20 pitches (the P4
types). If only one type qualifies, it gets weight 1.

1. **Who's in the group.** The P4 Gaussian observer gives, from the ball's
   direction 175 ms before arrival, a posterior weight $`w_k`$ for every type
   $`k`$, not only the fastball's and the true type's.
2. **Each member's path after commit.** Each type has a mean acceleration
   $`\mathbf a_k`$ (pitcher x side x type). From the hitter's commit point
   $`t_c = \max(T - \tau_b, 0.05)`$, the ball's actual position and velocity
   are extrapolated to the front of the plate with $`\mathbf a_k`$, exactly as
   P2 does with the fastball's acceleration. That gives a crossing
   $`(\hat x_k, \hat z_k)`$ and a time to the plate $`s_k`$.
3. **The group read** is the weighted mean $`\bar x = \sum_k w_k \hat x_k`$,
   and likewise $`\bar z`$ and $`\bar s`$.

Features:

- `gmiss_x_in` $`= 12\,s_b\,(x_p - \bar x)`$, `gmiss_z_in` $`= 12\,(z_p - \bar z)`$,
  `gmiss_t_ms` $`= 1000\,(\tau_b - \bar s)`$. Same signs as P2.
- `gspread_in` $`= 12\sqrt{\sum_k w_k\big[(\hat x_k - \bar x)^2 + (\hat z_k - \bar z)^2\big]}`$:
  how far apart the look-alike group's paths end up, so how much is left to
  guess after commit.

## Models and endpoints

LightGBM with the same parameters as `feasibility.py`. M0 and M1 are the
perceptual models (M1 = M0 + P1-P6).

- **Primary:** whiff | swing, 2026 log-loss gain of M1 + G over M1.
  **Pass** if the gain is >= 0.3 per 1000 AND above all 5 null draws (G
  permuted within pitcher x pitch type in train and test). For scale, P2 alone
  over M0 added 0.36 and P4 0.60 in the perceptual run.
- **Secondary (head to head):** whiff | swing gain over M0 of + G alone vs + P2
  alone. The group read beats the fastball read if its gain is larger.
- **Secondary (swap):** whiff | swing, M1 with P2 replaced by G, gain over M1.
- **Secondary (swing):** swing, gain of M1 + G over M1, with the same 5-draw
  null.

The question of whether a better trajectory change could be designed from this
(pitch design) is out of scope and is only worth asking if the primary passes.

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
| `pass` | Pre-registered pass rule met (stated in How the model works) | True/False | computed in the script |
| `w_k` | G. Posterior weight of type $`k`$ from the ball's direction 175 ms before arrival: the P4 Gaussian observer's full posterior over the pitcher x side's types (>= 20 pitches each) | probability | `group_read.weights` |
| `a_k` | G. Mean acceleration of type $`k`$ for the pitcher x side, within season | ft/s² | `group_read.type_accel` |
| `x_hat_k`, `z_hat_k`, `s_k` | G. Where the ball would cross the front of the plate, and when, if from the commit point it kept type $`k`$'s acceleration | ft, ft, s | `group_read.group_features` |
| `gmiss_x_in` | G. Horizontal miss of the group read: actual crossing minus the weighted mean of `x_hat_k`. Batter-oriented, + away | in | `group_read.group_features` |
| `gmiss_z_in` | G. Vertical miss of the group read; + means the pitch finished higher than the read | in | `group_read.group_features` |
| `gmiss_t_ms` | G. Arrival-time miss of the group read: $`\tau_b`$ minus the weighted mean of `s_k`. + means later than the read | ms | `group_read.group_features` |
| `gspread_in` | G. Weighted spread of the group's crossings around the group read | in | `group_read.group_features` |
