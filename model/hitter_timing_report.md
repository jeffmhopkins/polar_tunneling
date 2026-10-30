# Per-hitter decision point (pre-registered, single run)

> How every number below is computed is in [How the model works](#how-the-model-works), and every variable is defined in [Variables](#variables).

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

## How the model works

This pre-registered test (`PREREG_hitter.md`, Part 1) asks whether each hitter has his own decision point
that can be fitted from his results.

**Pitch features.** Every pitch is rebuilt from Statcast's 9-parameter fit, viewed from the batter's eye
and described by the controls, speed controls and perceptual features of `perceptual_report.md` and
`speed_check_report.md` (each of those reports walks through the formulas). In short: the trajectory is
$`\mathbf p(t) = \mathbf r_0 + \mathbf v_r t + \tfrac12\mathbf a t^2`$ from release to the front of the plate
at $`t = T`$, the hitter's eye is at $`\mathbf E`$, and the perceptual features describe the ball at the
commit time $`t_c = \max(T - \tau_b, 0.05)`$.

**Commit-time features at any $`\tau`$** (`hitter_timing.set_tau`). For a commit time $`\tau`$ (one value,
or one per hitter), every commit-time feature is recomputed at $`t_c = \max(T - \tau, 0.05)`$: `tau_b`
$`= \tau`$, P2 (`miss_*`), P3 (`ang_speed*`, `sep_rate_vs_fb`) and `sep_last_fb`. The P4 commit posteriors
are computed at each grid value $`\tau_j \in \{0.125, 0.150, 0.175, 0.200, 0.225\}`$ s and interpolated
linearly between the two nearest grid values:

```math
\text{post}(\tau) = (1 - w)\,\text{post}(\tau_{j-1}) + w\,\text{post}(\tau_j), \qquad w = \frac{\tau - \tau_{j-1}}{\tau_j - \tau_{j-1}}
```

**Model** (`feasibility.PARAMS`). Each outcome $`y \in \{0, 1\}`$ gets a LightGBM classifier: an additive
sum of $`K = 400`$ regression trees on the features $`\mathbf x`$, passed through the logistic function,

```math
\hat p(\mathbf x) = \sigma\Big(\sum_{k=1}^{K} f_k(\mathbf x)\Big), \qquad \sigma(u) = \frac{1}{1 + e^{-u}}
```

where each tree $`f_k`$ is fitted to the gradient of the log-loss of the trees before it (learning rate
0.05, at most 63 leaves, at least 200 training rows per leaf, a random 80% of features and of rows per
tree, seed 7). Categorical columns (`pitch_type`, `stand`, `p_throws`, `prev_pitch_type`, `prev_desc`)
are split on directly. Missing values are allowed: a tree learns which side of each split they go to.

**Fit on 2025** (`oof_loglik`, `posterior_tau`).

1. At each grid $`\tau_j`$, set every hitter to $`\tau_j`$ and fit the swing and whiff \| swing models
   (CONTROLS + SPEED + all P) with 2-fold cross-fitting by `game_pk` parity, so each pitch gets an
   out-of-fold log-likelihood $`\ell^{s}_i(\tau_j) = y_i \log \hat p_i + (1 - y_i)\log(1 - \hat p_i)`$ for swing,
   and $`\ell^{w}_i(\tau_j)`$ for whiff on swings.
2. League curve: $`L(\tau_j) = \sum_i \ell^s_i(\tau_j) + \sum_{i \in \text{swings}} \ell^w_i(\tau_j)`$.
   $`\tau^* = \arg\max_j L(\tau_j)`$.
3. Centered per-pitch score, so a league-wide shift does not move individual hitters:
   $`c_i(\tau_j) = (\ell^s_i - \overline{\ell^s}) + \mathbb 1[\text{swing}_i]\,(\ell^w_i - \overline{\ell^w})`$, means
   taken at the same $`\tau_j`$.
4. Hitter score $`S_b(\tau_j) = \sum_{i \in b} c_i(\tau_j)`$, and a posterior over the grid with a normal prior
   (sd 25 ms) at $`\tau^*`$:
   ```math
   w_b(\tau_j) \propto \exp\Big(S_b(\tau_j) - \tfrac12\Big(\frac{\tau_j - \tau^*}{0.025}\Big)^2\Big), \qquad
   \hat\tau_b = \sum_j w_b(\tau_j)\,\tau_j
   ```
   Hitters without 2025 pitches get $`\tau^*`$.
5. Split-half reliability: the same fit on odd and on even 2025 games separately, correlated over hitters
   with >= 500 pitches in each half.

**Score: held-out log-loss gain.** On the $`n`$ test pitches,

```math
L = -\frac{1}{n} \sum_{i=1}^{n} \big[ y_i \log \hat p_i + (1 - y_i) \log (1 - \hat p_i) \big],
\qquad \text{gain} \times 1000 = 1000\,(L_{\text{without}} - L_{\text{with}})
```

A positive gain means the added features improved the forecast of pitches the model never saw. For scale,
adding the previous pitch's type, location and result to the controls gained 1.4 (whiff per swing) and
2.1 (swing) per 1000 in 2025.

**2026 test.** All arms trained on 2025 and scored on 2026, for whiff \| swing and swing:

- A, current: bat-tracking $`\tau_b`$ (see `perceptual_report.md`), commit posteriors at 175 ms
- B, fitted: $`\tau = \hat\tau_b`$ for every hitter, all commit features at it
- C, league: everyone at $`\tau^*`$
- nulls: B with the fitted $`\hat\tau_b`$ shuffled among hitters, 5 draws

Gains are $`1000\,(L_A - L_{\text{arm}})`$. An outcome passes if B gains >= 0.3 and beats every null; Part 1
passes if one outcome passes and neither has B worse than A by more than 0.3.

## Variables

| Variable | Meaning | Units | Source |
|---|---|---|---|
| `pitcher` | MLBAM id of the pitcher | id | Statcast column `pitcher` (`datamining/outputs`) |
| `batter` | MLBAM id of the batter | id | Statcast column `batter` (`datamining/outputs`) |
| `game_pk` | MLBAM game id. Its parity (odd or even) splits a season into two halves of games | id | Statcast column `game_pk` (`datamining/outputs`) |
| `at_bat_number` | Plate appearance number within the game; with `game_pk` it identifies a PA | count | Statcast column `at_bat_number` (`datamining/outputs`) |
| `pitch_number` | Pitch number within the plate appearance (1 for the first pitch). Also a P6 feature | count | Statcast column `pitch_number` (`datamining/outputs`) |
| `game_type` | R = regular season. Only R rows are used, except the one game a game page is built for | category | Statcast column `game_type` (`datamining/outputs`) |
| `release_pos_x_ft` | Release point, horizontal (catcher's view, + toward the first-base side) | ft | Statcast column `release_pos_x` (`datamining/outputs`) |
| `release_pos_y_ft` | Release point, distance from the back of home plate ($`y_r`$) | ft | Statcast column `release_pos_y` (`datamining/outputs`) |
| `release_pos_z_ft` | Release height | ft | Statcast column `release_pos_z` (`datamining/outputs`) |
| `vx0_ft_s` | Velocity of Statcast's fit at y = 50 ft, x component ($`v_{x0}`$) | ft/s | Statcast column `vx0` (`datamining/outputs`) |
| `vy0_ft_s` | Velocity of the fit at y = 50 ft, y component ($`v_{y0}`$, negative: toward the plate) | ft/s | Statcast column `vy0` (`datamining/outputs`) |
| `vz0_ft_s` | Velocity of the fit at y = 50 ft, z component ($`v_{z0}`$) | ft/s | Statcast column `vz0` (`datamining/outputs`) |
| `ax_ft_s2` | Constant acceleration of the fit, x ($`a_x`$) | ft/s² | Statcast column `ax` (`datamining/outputs`) |
| `ay_ft_s2` | Constant acceleration of the fit, y ($`a_y`$, positive: drag) | ft/s² | Statcast column `ay` (`datamining/outputs`) |
| `az_ft_s2` | Constant acceleration of the fit, z ($`a_z`$, gravity plus lift) | ft/s² | Statcast column `az` (`datamining/outputs`) |
| `plate_x_ft` | Horizontal position where the pitch crosses the front of the plate ($`x_p`$, catcher's view) | ft | Statcast column `plate_x` (`datamining/outputs`) |
| `plate_z_ft` | Height where the pitch crosses the front of the plate ($`z_p`$) | ft | Statcast column `plate_z` (`datamining/outputs`) |
| `y_p` | Depth of the front edge of the plate, 17/12 ft, where `plate_x_ft`/`plate_z_ft` are measured | ft | `reconstruct_trajectory.PLATE_FRONT_Y` |
| `t(y)` | Time, counted from y = 50 ft, at which the fit reaches depth y (negative before y = 50) | s | `reconstruct_trajectory._time_at_y` |
| `r_0` | Ball position at release (`rx`, `ry`, `rz`), rebuilt from the fit | ft | `reconstruct_trajectory.release_state_from_fit` |
| `v_r` | Ball velocity at release (`rvx`, `rvy`, `rvz`) | ft/s | `reconstruct_trajectory.release_state_from_fit` |
| `tf` | Flight time $`T`$ from release to the front of the plate. Pitches with $`T \ge 0.6`$ s are dropped | s | `features.add_state` |
| `p(t)` | Ball position $`t`$ seconds after release, $`\mathbf r_0 + \mathbf v_r t + \tfrac12 \mathbf a t^2`$ | ft | `features.position` |
| `E` | Batter's eye: $`(s_b \cdot (-35.8),\ -28.3,\ 65.0)`$ in, i.e. 35.8 in off the plate's center line on the batter's side, 28.3 in behind the back of the plate, 65 in high | ft | `features.EYE_*`, `to_polar.ORIGIN_*` |
| `theta(a,b)` | Angle between two ball positions as seen from the eye: $`\arccos\big((\mathbf a-\mathbf E)\cdot(\mathbf b-\mathbf E)/(\lVert\mathbf a-\mathbf E\rVert\,\lVert\mathbf b-\mathbf E\rVert)\big)`$ | deg | `features.angle_deg` |
| `pitch_type` | Statcast pitch type code: FF four-seam, SI sinker, FC cutter, SL slider, ST sweeper, CU curve, KC knuckle curve, SV slurve, CH changeup, FS splitter, FO forkball, KN knuckleball. PO (pitchout), EP (eephus), FA (position players), CS (slow curve), SC (screwball) and UN (unknown) are dropped | category | Statcast column `pitch_type` (`datamining/outputs`) |
| `stand` | Batter side, R or L. Sets the eye position and the sign $`s_b`$ | category | Statcast column `stand` (`datamining/outputs`) |
| `p_throws` | Pitcher hand, R or L. Sets the sign $`s_a`$ | category | Statcast column `p_throws` (`datamining/outputs`) |
| `release_speed_mph` | Release speed | mph | Statcast column `release_speed` (`datamining/outputs`) |
| `pfx_x_ft` | Horizontal movement relative to a spinless pitch, catcher's view | ft | Statcast column `pfx_x` (`datamining/outputs`) |
| `pfx_z_ft` | Vertical movement relative to a spinless pitch (induced vertical break) | ft | Statcast column `pfx_z` (`datamining/outputs`) |
| `release_spin_rate_rpm` | Spin rate at release | rpm | Statcast column `release_spin_rate` (`datamining/outputs`) |
| `release_extension_ft` | Release extension toward the plate | ft | Statcast column `release_extension` (`datamining/outputs`) |
| `sz_top_ft` | Top of the batter's strike zone for this pitch | ft | Statcast column `sz_top` (`datamining/outputs`) |
| `sz_bot_ft` | Bottom of the batter's strike zone for this pitch | ft | Statcast column `sz_bot` (`datamining/outputs`) |
| `balls` | Balls before the pitch | 0-3 | Statcast column `balls` (`datamining/outputs`) |
| `strikes` | Strikes before the pitch | 0-2 | Statcast column `strikes` (`datamining/outputs`) |
| `description` | Result of the pitch (ball, called_strike, swinging_strike, foul, hit_into_play, ...) | category | Statcast column `description` (`datamining/outputs`) |
| `s_b` | Batter-side sign: +1 for a right-handed batter, -1 for a left-handed one. Multiplying an x value by it makes + mean away from the batter | ±1 | `stand` |
| `s_a` | Pitcher-hand sign: +1 for a right-hander, -1 for a left-hander. Multiplying an x value by it puts both hands on one convention | ±1 | `p_throws` |
| `mov_x` | Horizontal movement mirrored for lefties, $`s_a\,`$`pfx_x_ft`. Arm-side run is negative for both hands (a right-hander's four-seamer averages -0.65 ft) | ft | `features.add_outcomes_and_controls` |
| `rel_x` | Horizontal release point mirrored for lefties, $`s_a\,`$`release_pos_x_ft`. Negative is toward the arm side | ft | `features.add_outcomes_and_controls` |
| `loc_x` | Plate location across the plate, $`s_b\,x_p`$. + is away from the batter | ft | `features.add_outcomes_and_controls` |
| `loc_z` | Plate height scaled to the batter's zone, $`(z_p - z_{bot})/(z_{top} - z_{bot})`$: 0 at the bottom of the zone, 1 at the top | zone heights | `features.add_outcomes_and_controls` |
| `platoon` | 1 if batter and pitcher are on the same side (`stand == p_throws`) | 0/1 | `features.add_outcomes_and_controls` |
| `prev_pitch_type` | Type of the previous pitch in the same PA (`none` on the first pitch) | category | `features.add_previous` |
| `prev_desc` | Result of the previous pitch, grouped: whiff, called, foul, ball, other, none | category | `feasibility.prep`, `perceptual.season` |
| `prev_loc_x` | Previous pitch's `loc_x`, using the current batter side | ft | `feasibility.prep`, `perceptual.season` |
| `prev_loc_z` | Previous pitch's height scaled to the current pitch's zone | zone heights | `feasibility.prep`, `perceptual.season` |
| `swing` | 1 if the batter swung: swinging_strike(_blocked), foul_tip, foul, foul_bunt, hit_into_play, missed_bunt, bunt_foul_tip (`features.SWING`) | 0/1 | `features.add_outcomes_and_controls` |
| `whiff` | 1 if the batter swung and missed: swinging_strike(_blocked), foul_tip, missed_bunt (`features.WHIFF`). Modelled on swings only, so the model is P(whiff \| swing) | 0/1 | `features.add_outcomes_and_controls` |
| `csw` | Called strike or whiff: 1 if `whiff` or `description == called_strike` | 0/1 | `features.add_outcomes_and_controls` |
| `CSW` | Share of pitches that are a called strike or whiff (`csw` averaged) | % or pts | `csw` |
| `primary fastball` | For a pitcher x batter side: the most-used of FF, SI and FC (to that side, in the data used for the features) | - | `arsenal.group_table`, `perceptual.primary_fastball` |
| `fb_dv` | Release speed minus the primary fastball's mean release speed | mph | `arsenal.group_table`, `perceptual.season` |
| `fb_dmx` | `mov_x` minus the primary fastball's mean `mov_x` | ft | `arsenal.group_table`, `perceptual.season` |
| `fb_dmz` | `pfx_z_ft` minus the primary fastball's mean `pfx_z_ft` | ft | `arsenal.group_table`, `perceptual.season` |
| `fb_drx` | `rel_x` minus the primary fastball's mean `rel_x` | ft | `arsenal.group_table`, `perceptual.season` |
| `fb_drz` | `release_pos_z_ft` minus the primary fastball's mean release height | ft | `arsenal.group_table`, `perceptual.season` |
| `arm_angle_deg` | Pitcher arm angle at release | deg | Statcast column `arm_angle` (`datamining/outputs`) |
| `bat_whiff_rate` | Batter's whiff per swing in the same season, leaving out the current game, shrunk to the league rate with 100 prior swings | rate | `perceptual.batter_rates` |
| `bat_swing_rate` | Batter's swing rate in the same season, leaving out the current game, shrunk with 100 prior pitches | rate | `perceptual.batter_rates` |
| `tau_b` | P1. Hitter's commit time from bat tracking: $`0.175 \times \tilde\rho_b/\bar\rho`$, clipped to 0.125-0.225 s, where $`\rho`$ = swing length / bat speed (swings over 40 mph), $`\bar\rho`$ the league mean and $`\tilde\rho_b`$ the batter's mean shrunk to it with 50 prior swings. 0.175 s if he has none | s | `perceptual.batter_tau` |
| `bat_speed_mph` | Bat speed on a tracked swing | mph | Statcast column `bat_speed` (`datamining/outputs`) |
| `swing_length_ft` | Swing length on a tracked swing | ft | Statcast column `swing_length` (`datamining/outputs`) |
| `t_c` | Commit time after release, $`T - \tau`$ (floored at 0 in the geometric features, at 0.05 s in the perceptual ones) | s | `features.add_features`, `perceptual.commit_features` |
| `v(t)` | Ball velocity $`t`$ seconds after release, $`\mathbf v_r + \mathbf a t`$ | ft/s | `perceptual.velocity` |
| `miss_x_in` | P2. Horizontal error of a fastball read: actual plate crossing minus where the ball would cross if, from the commit point, it kept the primary fastball's acceleration. Batter-oriented ($`s_b`$), + is away | in | `perceptual.commit_features` |
| `miss_z_in` | P2. Vertical error of the same fastball read; + means the pitch finished higher than the read | in | `perceptual.commit_features` |
| `miss_t_ms` | P2. Arrival-time error: actual time left at commit ($`\tau_b`$) minus the fastball read's time to the plate. + means the pitch arrives later than a fastball would | ms | `perceptual.commit_features` |
| `ang_speed` | P3. Angular speed of the ball as seen from the eye at commit, $`\lVert\mathbf v_\perp\rVert/r`$ | deg/s | `perceptual.angular_speed` |
| `ang_speed_vs_fb` | P3. `ang_speed` minus the angular speed of the primary fastball's mean path at the same time since release | deg/s | `perceptual.commit_features` |
| `sep_rate_vs_fb` | P3. Rate of change of the angle between the pitch and the fastball's mean path at commit, central difference with h = 5 ms | deg/s | `perceptual.commit_features` |
| `post_fb_commit` | P4. Posterior probability, from the ball's direction 175 ms before arrival, that the pitch is the primary fastball (Gaussian observer over the pitcher x side's types) | probability | `perceptual.posteriors` |
| `post_true_commit` | P4. Posterior probability, from the same view, of the pitch's true type | probability | `perceptual.posteriors` |
| `post_fb_release` | P4. As `post_fb_commit`, from the direction of the release point | probability | `perceptual.posteriors` |
| `post_true_release` | P4. As `post_true_commit`, from the direction of the release point | probability | `perceptual.posteriors` |
| `surprise_count` | P5. Surprisal of the type given the count, $`-\log_2 P(\text{type} \mid \text{pitcher, side, count})`$, usage smoothed with 5 prior pitches | bits | `perceptual.surprisal` |
| `count` | `balls-strikes` before the pitch, e.g. `1-2` (12 values) | category | built from `balls`, `strikes` in `features.add_outcomes_and_controls` |
| `surprise_prev` | P5. Surprisal given the previous pitch type, $`-\log_2 P(\text{type} \mid \text{pitcher, side, previous type})`$; missing on a PA's first pitch | bits | `perceptual.surprisal` |
| `fb_seen_pa` | P6. Fastballs (FF, SI, FC) already thrown earlier in the PA | count | `perceptual.season` |
| `sep_last_fb` | P6. Angle from the eye at commit between this pitch and the most recent earlier fastball in the PA (same time since release); missing if none | deg | `perceptual.season` |
| `n_priorpa_thisgame` | Times the batter has already faced this pitcher in the game (times through the order minus 1). Also a P6 feature | count | Statcast column `n_priorpa_thisgame_player_at_bat` (`datamining/outputs`) |
| `plate_speed_mph` | Speed of the fit at the front of the plate, $`\lVert\mathbf v(T)\rVert/1.46667`$ | mph | `speed_check.add_speed` |
| `flight_ms` | Flight time $`1000\,T`$ | ms | `speed_check.add_speed` |
| `prev_dv` | Release speed minus the previous pitch's release speed | mph | `speed_check.add_speed` |
| `prev_dtf_ms` | Flight time minus the previous pitch's flight time | ms | `speed_check.add_speed` |
| `fb_dtf_ms` | Flight time minus the primary fastball's mean flight time | ms | `speed_check.add_speed` |
| `M0` | Control model: LightGBM on CONTROLS = BASE + PREV + fastball differentials + `arm_angle_deg` + batter rates | - | `perceptual.CONTROLS` |
| `M1` | M0 plus all 17 perceptual features (P1-P6) | - | `perceptual.ALL_P` |
| `M0s` | M0 plus the five speed controls | - | `speed_check.SPEED` |
| `M1s` | M0s plus all perceptual features | - | `speed_check.main` |
| `LightGBM` | Gradient-boosted trees. Settings everywhere unless stated: binary log-loss objective, learning rate 0.05, 63 leaves, >= 200 rows per leaf, 80% of features and 80% of rows per tree, 400 trees, seed 7 | - | `feasibility.PARAMS`, `feasibility.ROUNDS` |
| `log-loss` | $`-\frac1n\sum_i [y_i\log\hat p_i + (1-y_i)\log(1-\hat p_i)]`$ on held-out pitches; lower is better | nats per pitch | `sklearn.metrics.log_loss` |
| `gain x1000` | $`1000\,(L_{\text{without}} - L_{\text{with}})`$: how much adding the features lowers held-out log-loss. + helps | nats per 1000 pitches | computed in each script |
| `null` | The same model refit after shuffling the tested features among pitches of the same pitcher x pitch type (or group), which keeps each feature's spread but breaks its link to the pitch | - | `feasibility.shuffled`, `perceptual.shuffled` |
| `tau` | A commit time on the grid {125, 150, 175, 200, 225} ms | ms | `hitter_timing.GRID` |
| `tau*` | League-best grid commit time: the grid value with the highest total out-of-fold log-likelihood on 2025 | ms | `hitter_timing.main` |
| `tau_fit` | A hitter's fitted commit time: posterior mean over the grid | s | `hitter_timing.posterior_tau`, `batter_tau_2025.csv` |
| `tau_bat` | The bat-tracking commit time (`tau_b`) for the same hitter | s | `batter_tau_2025.csv` |
| `n_2025` | Pitches the hitter saw in 2025 | count | `batter_tau_2025.csv` |
| `S_b(tau)` | Hitter score at a grid tau: sum over his pitches of the pitch's out-of-fold log-likelihood minus the league per-pitch mean at that tau | nats | `hitter_timing.oof_loglik` |
| `total log-lik` | Sum of out-of-fold log-likelihoods for swing (all pitches) and whiff (swings) at one tau | nats | `hitter_timing.oof_loglik` |
| `tau ms` | Grid commit time | ms | `hitter_timing.GRID` |
| `vs best` | Total log-lik minus the best grid value's | nats | `hitter_timing.main` |
| `split-half r` | Pearson correlation of a group's value in odd-`game_pk` games with its value in even-`game_pk` games | r | computed in each script |
| `2026 log-loss A` | 2026 log-loss of arm A (current timing) | nats per pitch | `hitter_timing.main` |
| `B fitted gain x1000` | Gain x1000 of arm B (fitted per-hitter timing) over A | nats per 1000 pitches | `hitter_timing.main` |
| `C league tau* gain x1000` | Gain x1000 of arm C (everyone at tau*) over A | nats per 1000 pitches | `hitter_timing.main` |
| `null gains` | Gain x1000 of each shuffled-feature refit; the real gain has to beat all of them | nats per 1000 pitches | computed in each script |
| `pass` | Pre-registered pass rule met (stated in How the model works) | True/False | computed in the script |
