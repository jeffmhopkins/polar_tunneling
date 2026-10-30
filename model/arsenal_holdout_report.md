# Arsenal tunnel check, 2025 -> 2026

> How every number below is computed is in [How the model works](#how-the-model-works), and every variable is defined in [Variables](#variables).

1,379 pitcher x pitch type x side groups in both seasons; 245,134 2026 pitches.

Year-to-year correlation of the group tunnel features: a_dec_deg 0.85, a_track_in 0.63, a_plate_in 0.68

Log-loss gain x 1000 on 2026 of +2025 arsenal tunnel over stuff + location + count + fastball differentials (model trained on 2025):

| outcome       |   2026 n |   gain x1000 |   null mean |   null max | share of nulls beaten   |
|:--------------|---------:|-------------:|------------:|-----------:|:------------------------|
| whiff \| swing |   117474 |         0.1  |       -0.38 |      -0.12 | 100%                    |
| swing         |   245134 |         0.51 |       -2.46 |      -2.22 | 100%                    |
| CSW           |   245134 |         0.19 |       -1.23 |      -1.1  | 100%                    |

Group level (1065 groups with >= 40 swings in 2026): rank correlation of 2025 tunnel features with 2026 residual whiff/swing (tunneling predicts negative for a_dec_deg):

|            |   Spearman r |
|:-----------|-------------:|
| a_dec_deg  |       -0.07  |
| a_track_in |       -0.045 |
| a_plate_in |       -0.107 |

Mean 2026 residual whiff/swing (pct points) by quartile of 2025 commit-point angle to the fastball:

| dec_q    |   groups |   resid_whiff_pct |
|:---------|---------:|------------------:|
| closest  |      267 |             -3.55 |
| 2        |      266 |             -2.75 |
| 3        |      266 |             -4.07 |
| farthest |      266 |             -4.77 |

Closest minus farthest quartile: 1.22 pct points (95% bootstrap interval 0.20 to 2.23).

## How the model works

This is the out-of-season version of the arsenal check: do group tunnel features measured in 2025 predict
2026 results beyond 2026 stuff, location and fastball differentials?

**Data.** Regular-season pitches with a spin rate, all of 2025 and all of 2026. Only groups present in both
seasons are scored.

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

**Commit point.** The hitter has to decide $`\tau`$ seconds before the ball reaches the front of the plate
($`\tau = 0.175`$ s, `features.TAU`). For a pitch with flight time $`T`$ that moment is
$`t_c = \max(T - \tau, 0)`$ after release. Two pitches are compared at the same time since their own
release, so the hitter's clock, not the pitch's share of its flight, sets the comparison. The tracking
limit is 0.100 s before arrival (`features.TRACK`).

**Group features** (`arsenal.group_table`). Groups are pitcher x batter side x pitch type with
>= 50 pitches in the feature months. For each group, the release state, flight time, plate crossing,
speed, movement and release point are averaged over its pitches, giving an average path
$`\bar{\mathbf p}(t)`$ with flight time $`\bar T`$. Each pitcher x side's primary fastball is its highest-usage
FF, SI or FC group. For every other type (path $`\bar{\mathbf p}`$) against the fastball
($`\bar{\mathbf p}_{fb}`$, $`\bar T_{fb}`$):

1. On the fastball's clock: $`t^* = \max(\bar T_{fb} - \tau, 0)`$,
   ```math
   a_{dec} = \theta\big(\bar{\mathbf p}(\min(t^*, \bar T)),\ \bar{\mathbf p}_{fb}(t^*)\big)\ \text{deg}
   ```
2. At equal depth at the tracking limit: $`y^* = \bar p_y(\bar T - 0.1)`$, $`t'`$ the time the fastball
   reaches $`y^*`$,
   ```math
   a_{track} = 12\sqrt{\big(\bar p_x(\bar T - 0.1) - \bar p_{fb,x}(t')\big)^2 + \big(\bar p_z(\bar T - 0.1) - \bar p_{fb,z}(t')\big)^2}\ \text{in}
   ```
3. At the plate: $`a_{plate} = 12\sqrt{(\bar x_p - \bar x_{p,fb})^2 + (\bar z_p - \bar z_{p,fb})^2}`$ in.

Fastball differentials (DIFF) are group means minus the fastball's: speed `fb_dv`, movement `fb_dmx`,
`fb_dmz`, and release `fb_drx`, `fb_drz`. Every pitch inherits its group's features, so all of them are
constant within a group.

**What comes from which season.** The group tunnel features (`a_dec_deg`, `a_track_in`, `a_plate_in`)
come from all of 2025. In the 2026 test rows, the fastball differentials come from 2026 groups (current
stuff) while the tunnel features are the 2025 values. The year-to-year line correlates each tunnel feature
of a group in 2025 with the same feature computed from 2026.

**Outcomes** (`features.add_outcomes_and_controls`):

```math
\text{swing} = \mathbb 1[\text{description} \in \text{SWING}], \quad
\text{whiff} = \mathbb 1[\text{description} \in \text{WHIFF}], \quad
\text{csw} = \text{whiff} \lor \mathbb 1[\text{description} = \text{called\_strike}]
```

Whiff is modelled on swings only, so its model estimates $`P(\text{whiff} \mid \text{swing})`$; swing and
CSW are modelled on all pitches.

**Controls (BASE)** (`feasibility.BASE`, `features.add_outcomes_and_controls`): `pitch_type`, `stand`,
`p_throws`, `release_speed_mph`, `mov_x` $`= s_a \cdot`$ `pfx_x_ft`, `pfx_z_ft`, `release_spin_rate_rpm`,
`release_extension_ft`, `release_pos_z_ft`, `rel_x` $`= s_a \cdot`$ `release_pos_x_ft`, `loc_x` $`= s_b x_p`$,
`loc_z` $`= (z_p - z_{bot})/(z_{top} - z_{bot})`$, `balls`, `strikes`, `platoon`. Here $`s_a = +1`$ for a
right-handed pitcher and $`-1`$ for a lefty, so both hands share one sign convention.

**Previous-pitch controls (PREV)**: `prev_pitch_type`, `prev_desc` (whiff, called, foul, ball, other),
`prev_loc_x`, `prev_loc_z`.

**Model** (`feasibility.PARAMS`). Each outcome $`y \in \{0, 1\}`$ gets a LightGBM classifier: an additive
sum of $`K = 400`$ regression trees on the features $`\mathbf x`$, passed through the logistic function,

```math
\hat p(\mathbf x) = \sigma\Big(\sum_{k=1}^{K} f_k(\mathbf x)\Big), \qquad \sigma(u) = \frac{1}{1 + e^{-u}}
```

where each tree $`f_k`$ is fitted to the gradient of the log-loss of the trees before it (learning rate
0.05, at most 63 leaves, at least 200 training rows per leaf, a random 80% of features and of rows per
tree, seed 7). Categorical columns (`pitch_type`, `stand`, `p_throws`, `prev_pitch_type`, `prev_desc`)
are split on directly. Missing values are allowed: a tree learns which side of each split they go to.

**Test.** Models trained on 2025 pitches (with 2025 group features) and scored on 2026: base = BASE +
DIFF (the previous-pitch controls are not used here), full = BASE + DIFF + tunnel features. $`\text{gain} \times 1000 = 1000\,(L_{\text{base}} - L_{\text{full}})`$
on 2026.

**Null.** The 2025 tunnel features are shuffled among 2026 groups of the same pitch type, in the test rows
only, and scored with the full model trained on real features (10 draws). Because the trained model partly
uses the group features as identifiers, scrambling them at test time hurts more than leaving them out, so
these nulls are a low bar.

**Group-level check.** On 2026 swings, residual whiff
$`resid_i = whiff_i - \hat p_i`$ against the BASE + DIFF model is averaged per group (groups with >= 40
2026 swings). The table gives the Spearman rank correlation of each group feature with the mean residual
(tunneling predicts a negative correlation for `a_dec_deg`: closer at commit, more whiffs than expected),
and the mean residual (x100) by quartile of `a_dec_deg`.

**Bootstrap of the quartile gap.** Groups are resampled with replacement 2,000 times; each time the mean
residual of the closest quartile minus the farthest (x100) is recomputed. The interval is the 2.5th to
97.5th percentile.

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
| `swing` | 1 if the batter swung: swinging_strike(_blocked), foul_tip, foul, foul_bunt, hit_into_play, missed_bunt, bunt_foul_tip (`features.SWING`) | 0/1 | `features.add_outcomes_and_controls` |
| `whiff` | 1 if the batter swung and missed: swinging_strike(_blocked), foul_tip, missed_bunt (`features.WHIFF`). Modelled on swings only, so the model is P(whiff \| swing) | 0/1 | `features.add_outcomes_and_controls` |
| `csw` | Called strike or whiff: 1 if `whiff` or `description == called_strike` | 0/1 | `features.add_outcomes_and_controls` |
| `CSW` | Share of pitches that are a called strike or whiff (`csw` averaged) | % or pts | `csw` |
| `month` | Calendar month of the game; Mar-Jul vs Aug-Oct splits a season in time | 1-12 | `game_date` |
| `TAU` | Commit point: how long before the ball reaches the front of the plate the hitter must decide. Fixed at 0.175 s unless stated | s | `features.TAU` |
| `TRACK` | Tracking limit: 0.100 s before arrival, the last moment the hitter can still follow the ball | s | `features.TRACK` |
| `t_c` | Commit time after release, $`T - \tau`$ (floored at 0 in the geometric features, at 0.05 s in the perceptual ones) | s | `features.add_features`, `perceptual.commit_features` |
| `primary fastball` | For a pitcher x batter side: the most-used of FF, SI and FC (to that side, in the data used for the features) | - | `arsenal.group_table`, `perceptual.primary_fastball` |
| `a_dec_deg` | Angle from the eye between a pitch type's average path and the primary fastball's average path, at the fastball's commit point ($`T_{fb}-\tau`$, same time since release) | deg | `arsenal.group_table` |
| `a_track_in` | x-z distance between the type's average path and the fastball's at the depth the type reaches 0.100 s before arrival | in | `arsenal.group_table` |
| `a_plate_in` | Distance between the type's and the fastball's average plate crossings | in | `arsenal.group_table` |
| `fb_dv` | Release speed minus the primary fastball's mean release speed | mph | `arsenal.group_table`, `perceptual.season` |
| `fb_dmx` | `mov_x` minus the primary fastball's mean `mov_x` | ft | `arsenal.group_table`, `perceptual.season` |
| `fb_dmz` | `pfx_z_ft` minus the primary fastball's mean `pfx_z_ft` | ft | `arsenal.group_table`, `perceptual.season` |
| `fb_drx` | `rel_x` minus the primary fastball's mean `rel_x` | ft | `arsenal.group_table`, `perceptual.season` |
| `fb_drz` | `release_pos_z_ft` minus the primary fastball's mean release height | ft | `arsenal.group_table`, `perceptual.season` |
| `resid` | Residual: actual outcome (0/1) minus a model's predicted probability from a model without the features being tested | probability (shown in pct points) | `feasibility`, `arsenal`, `perceptual` |
| `dec_q` | Quartile of `a_dec_deg` across groups: closest, 2, 3, farthest | category | `arsenal.main` |
| `resid_whiff_pct` | Mean residual whiff per swing of the groups in a quartile, x100 | pct points | `arsenal.main` |
| `LightGBM` | Gradient-boosted trees. Settings everywhere unless stated: binary log-loss objective, learning rate 0.05, 63 leaves, >= 200 rows per leaf, 80% of features and 80% of rows per tree, 400 trees, seed 7 | - | `feasibility.PARAMS`, `feasibility.ROUNDS` |
| `log-loss` | $`-\frac1n\sum_i [y_i\log\hat p_i + (1-y_i)\log(1-\hat p_i)]`$ on held-out pitches; lower is better | nats per pitch | `sklearn.metrics.log_loss` |
| `gain x1000` | $`1000\,(L_{\text{without}} - L_{\text{with}})`$: how much adding the features lowers held-out log-loss. + helps | nats per 1000 pitches | computed in each script |
| `null` | The same model refit after shuffling the tested features among pitches of the same pitcher x pitch type (or group), which keeps each feature's spread but breaks its link to the pitch | - | `feasibility.shuffled`, `perceptual.shuffled` |
| `null gains` | Gain x1000 of each shuffled-feature refit; the real gain has to beat all of them | nats per 1000 pitches | computed in each script |
| `share of nulls beaten` | Share of null draws whose gain is below the real gain | % | computed in each script |
| `2026 n` | 2026 pitches scored (swings only for whiff \| swing) | count | computed in the script |
| `null mean` | Mean gain x1000 over the null draws | nats per 1000 pitches | computed in the script |
| `null max` | Largest gain x1000 over the null draws | nats per 1000 pitches | computed in the script |
| `95% CI` | 2.5th to 97.5th percentile of the statistic over a pitcher bootstrap: pitchers drawn with replacement, each pitcher's pitches weighted by how often he was drawn | - | `validate_sequencer.boot` |
