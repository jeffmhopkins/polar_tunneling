# Speed check: perceptual signal with stronger speed controls (2025 -> 2026, single run)

> How every number below is computed is in [How the model works](#how-the-model-works), and every variable is defined in [Variables](#variables).

Speed controls added to M0: plate_speed_mph, flight_ms, prev_dv, prev_dtf_ms, fb_dtf_ms.

Log-loss gain x 1000 on 2026. Rows marked (expl.) are exploratory and not part of pass/fail.

|                                 | whiff \| swing                | swing                             |
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

## How the model works

This pre-registered check (`PREREG_speed.md`) asks whether the perceptual gain in `perceptual_report.md`
is really speed and flight-time information that the controls missed.

**Data.** As in the perceptual study: train all of 2025, test once on 2026.

**Coordinates.** Statcast's catcher's-view frame, in feet: $`x`$ across the plate (+ toward the
first-base side), $`y`$ from the back point of home plate toward the mound, $`z`$ up. The front edge of
the plate is at $`y_p = 17/12`$ ft.

**Rebuild the trajectory** (`polar_transformations/reconstruct_trajectory.py`, `features.add_state`).
Statcast fits every pitch with a constant acceleration $`\mathbf a = (a_x, a_y, a_z)`$ and publishes the
fit's velocity $`\mathbf v_{50} = (v_{x0}, v_{y0}, v_{z0})`$ at $`y = 50`$ ft, the release depth $`y_r`$ and
the plate crossing $`(x_p, z_p)`$ at $`y_p`$. The time, counted from $`y = 50`$, at which the fit reaches
depth $`y`$ is the root of $`50 + v_{y0} t + \tfrac12 a_y t^2 = y`$ that the ball reaches first:

```math
t(y) = \frac{-v_{y0} - \sqrt{v_{y0}^2 - 2 a_y (50 - y)}}{a_y}
```

With $`t_p = t(y_p)`$ (positive) and $`t_r = t(y_r)`$ (negative, release is before $`y = 50`$), the fit's
$`x`$ and $`z`$ at $`y = 50`$ follow from the plate crossing, and the state at release from running the fit
back to $`t_r`$:

```math
x_{50} = x_p - v_{x0} t_p - \tfrac12 a_x t_p^2, \qquad z_{50} = z_p - v_{z0} t_p - \tfrac12 a_z t_p^2
```

```math
\mathbf r_0 = \big(x_{50} + v_{x0} t_r + \tfrac12 a_x t_r^2,\ \ y_r,\ \ z_{50} + v_{z0} t_r + \tfrac12 a_z t_r^2\big), \qquad
\mathbf v_r = \mathbf v_{50} + \mathbf a\, t_r, \qquad T = t_p - t_r
```

$`T`$ is the flight time from release to the front of the plate. From release, the ball's position and
velocity at time $`t \in [0, T]`$ are

```math
\mathbf p(t) = \mathbf r_0 + \mathbf v_r t + \tfrac12 \mathbf a t^2, \qquad \mathbf v(t) = \mathbf v_r + \mathbf a t
```

Pitches with $`T \ge 0.6`$ s (lobs and bad fits) are dropped, as are pitch types PO, EP, FA, CS, SC and UN
and any row missing a fit column.

**View it from the batter's eye** (`features.EYE_*`). The eye sits 35.8 in off the plate's center line
on the batter's side, 28.3 in behind the back point of the plate and 65 in high:

```math
\mathbf E = \tfrac{1}{12}\,\big(-35.8\, s_b,\ -28.3,\ 65.0\big)\ \text{ft}, \qquad s_b = +1 \text{ (right-handed batter)},\ -1 \text{ (left-handed)}
```

The angle between two ball positions $`\mathbf a`$ and $`\mathbf b`$ as the hitter sees them is

```math
\theta(\mathbf a, \mathbf b) = \frac{180}{\pi} \arccos \frac{(\mathbf a - \mathbf E)\cdot(\mathbf b - \mathbf E)}{\lVert \mathbf a - \mathbf E\rVert\, \lVert \mathbf b - \mathbf E\rVert}\ \ \text{deg}
```

**Outcomes** (`features.add_outcomes_and_controls`):

```math
\text{swing} = \mathbb 1[\text{description} \in \text{SWING}], \quad
\text{whiff} = \mathbb 1[\text{description} \in \text{WHIFF}], \quad
\text{csw} = \text{whiff} \lor \mathbb 1[\text{description} = \text{called\_strike}]
```

Whiff is modelled on swings only, so its model estimates $`P(\text{whiff} \mid \text{swing})`$; swing and
CSW are modelled on all pitches.

**Controls (M0)** (`perceptual.CONTROLS`): BASE + PREV as in the feasibility study (`prev_desc` also has a
`none` level for a PA's first pitch), plus

- fastball differentials against the pitcher's primary fastball to that side (`perceptual.primary_fastball`:
  his most-used FF, SI or FC to that side in the season, averaged): `fb_dv` (release speed), `fb_dmx`,
  `fb_dmz` (movement), `fb_drx`, `fb_drz` (release point)
- `arm_angle_deg`
- batter rates within the season, leaving out the current game $`g`$, shrunk to the league rate $`\ell`$ with
  $`k = 100`$ prior swings (or pitches):
  ```math
  \text{bat\_whiff\_rate}_{b,g} = \frac{(W_b - W_{b,g}) + k\,\ell_w}{(S_b - S_{b,g}) + k}, \qquad
  \text{bat\_swing\_rate}_{b,g} = \frac{(S_b - S_{b,g}) + k\,\ell_s}{(N_b - N_{b,g}) + k}
  ```
  with $`W`$ whiffs, $`S`$ swings and $`N`$ pitches, for the batter's season and for game $`g`$.

**Speed controls (S)** (`speed_check.add_speed`), added to M0 to make M0s:

```math
\text{plate\_speed\_mph} = \frac{\lVert \mathbf v(T) \rVert}{1.46667}, \quad
\text{flight\_ms} = 1000\,T, \quad
\text{prev\_dv} = v_{rel} - v'_{rel}, \quad
\text{prev\_dtf\_ms} = 1000\,(T - T'), \quad
\text{fb\_dtf\_ms} = 1000\,(T - T_{fb})
```

where primes are the previous pitch in the PA, $`v_{rel}`$ is release speed (mph) and $`T_{fb}`$ the primary
fastball's mean flight time (1.46667 ft/s per mph). Release speed and the speed differential to the
fastball are already in M0.

**Kinematics block (K, exploratory).** The ball's batter-oriented position and velocity at the hitter's
commit point $`t_c = \max(T - \tau_b, 0.05)`$:
$`(k_x, k_y, k_z) = (s_b\, p_x, p_y, p_z)`$ and $`(k_{vx}, k_{vy}, k_{vz}) = (s_b\, v_x, v_y, v_z)`$. Optical motion
(P3) is a function of these six numbers and the eye position, so M0s + K asks whether expressing them as
angles adds anything.

**Perceptual features (P1-P6).** Exactly as defined in `perceptual_report.md` (How the model works): timing
`tau_b`, fastball-prior miss, optical motion, ambiguity posteriors, surprisal and at-bat build-up. Their
variables are listed below.

**Model** (`feasibility.PARAMS`). Each outcome $`y \in \{0, 1\}`$ gets a LightGBM classifier: an additive
sum of $`K = 400`$ regression trees on the features $`\mathbf x`$, passed through the logistic function,

```math
\hat p(\mathbf x) = \sigma\Big(\sum_{k=1}^{K} f_k(\mathbf x)\Big), \qquad \sigma(u) = \frac{1}{1 + e^{-u}}
```

where each tree $`f_k`$ is fitted to the gradient of the log-loss of the trees before it (learning rate
0.05, at most 63 leaves, at least 200 training rows per leaf, a random 80% of features and of rows per
tree, seed 7). Categorical columns (`pitch_type`, `stand`, `p_throws`, `prev_pitch_type`, `prev_desc`)
are split on directly. Missing values are allowed: a tree learns which side of each split they go to.

**Score: held-out log-loss gain.** On the $`n`$ test pitches,

```math
L = -\frac{1}{n} \sum_{i=1}^{n} \big[ y_i \log \hat p_i + (1 - y_i) \log (1 - \hat p_i) \big],
\qquad \text{gain} \times 1000 = 1000\,(L_{\text{without}} - L_{\text{with}})
```

A positive gain means the added features improved the forecast of pitches the model never saw. For scale,
adding the previous pitch's type, location and result to the controls gained 1.4 (whiff per swing) and
2.1 (swing) per 1000 in 2025.

**Models.** For whiff \| swing and swing, all trained on 2025 and scored on 2026:

| Row | Compared | Gain |
|---|---|---|
| M1s gain | M0s + all P vs M0s | $`1000\,(L_{M0s} - L_{M1s})`$ |
| +P1 ... +P6 | M0s + one block vs M0s | $`1000\,(L_{M0s} - L_{M0s + block})`$ |
| share of original gain | M1s gain / M1 gain in `perceptual_report.md` (1.83, 5.86) | % |
| (expl.) +kinematics over M0s | M0s + K vs M0s | $`1000\,(L_{M0s} - L_{M0s+K})`$ |
| (expl.) +P over M0s+kinematics | M0s + K + all P vs M0s + K | $`1000\,(L_{M0s+K} - L_{M0s+K+P})`$ |
| (expl.) +P3 over M0s+kinematics | M0s + K + P3 vs M0s + K | $`1000\,(L_{M0s+K} - L_{M0s+K+P3})`$ |

**Permutation null.** To check that a gain is more than the extra flexibility of more columns, the
tested features are shuffled among pitches of the same pitcher x pitch type, and the model is refit and
scored the same way. Shuffling keeps each pitcher's pitch types and the features' spread, and breaks only
the link between a feature value and the pitch it belongs to. A real gain has to beat every null draw. All P features are shuffled within pitcher x pitch type in train and test and M1s is refit, 5 draws.

**Pass** if the M1s gain is >= 1.0 per 1000 and above all 5 nulls on that endpoint.

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
| `k_x` | Ball position at commit, batter-oriented: $`(s_b p_x, p_y, p_z)`$ as `k_x, k_y, k_z` | ft | `speed_check.add_speed` |
| `k_vx` | Ball velocity at commit, batter-oriented: $`(s_b v_x, v_y, v_z)`$ as `k_vx, k_vy, k_vz` | ft/s | `speed_check.add_speed` |
| `M0` | Control model: LightGBM on CONTROLS = BASE + PREV + fastball differentials + `arm_angle_deg` + batter rates | - | `perceptual.CONTROLS` |
| `M0s` | M0 plus the five speed controls | - | `speed_check.SPEED` |
| `M1s` | M0s plus all perceptual features | - | `speed_check.main` |
| `LightGBM` | Gradient-boosted trees. Settings everywhere unless stated: binary log-loss objective, learning rate 0.05, 63 leaves, >= 200 rows per leaf, 80% of features and 80% of rows per tree, 400 trees, seed 7 | - | `feasibility.PARAMS`, `feasibility.ROUNDS` |
| `log-loss` | $`-\frac1n\sum_i [y_i\log\hat p_i + (1-y_i)\log(1-\hat p_i)]`$ on held-out pitches; lower is better | nats per pitch | `sklearn.metrics.log_loss` |
| `gain x1000` | $`1000\,(L_{\text{without}} - L_{\text{with}})`$: how much adding the features lowers held-out log-loss. + helps | nats per 1000 pitches | computed in each script |
| `null` | The same model refit after shuffling the tested features among pitches of the same pitcher x pitch type (or group), which keeps each feature's spread but breaks its link to the pitch | - | `feasibility.shuffled`, `perceptual.shuffled` |
| `null gains` | Gain x1000 of each shuffled-feature refit; the real gain has to beat all of them | nats per 1000 pitches | computed in each script |
| `share of nulls beaten` | Share of null draws whose gain is below the real gain | % | computed in each script |
| `2026 n` | 2026 pitches scored (swings only for whiff \| swing) | count | computed in the script |
| `M0s log-loss` | 2026 log-loss of M0s | nats per pitch | `speed_check.main` |
| `M1s gain x1000` | Gain x1000 of M1s over M0s on 2026 | nats per 1000 pitches | `speed_check.main` |
| `share of original gain` | M1s gain divided by the M1 gain from `perceptual_report.md` (1.83 whiff, 5.86 swing) | % | `speed_check.ORIGINAL` |
| `+P1 ... +P6` | Gain x1000 over the baseline of adding one block only | nats per 1000 pitches | `perceptual.main` |
| `pass` | Pre-registered pass rule met (stated in How the model works) | True/False | computed in the script |
