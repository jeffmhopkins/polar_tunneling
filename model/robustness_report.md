# Lift robustness (log-loss gain x 1000 of +tunnel over +prev, test Aug-Oct)

> How every number below is computed is in [How the model works](#how-the-model-works), and every variable is defined in [Variables](#variables).

|   commit ms | pairs                | outcome       |   gain x1000 |   null mean |   null max |   null draws | share of nulls beaten   |
|------------:|:---------------------|:--------------|-------------:|------------:|-----------:|-------------:|:------------------------|
|         125 | all pairs            | whiff \| swing |        -0.58 |       -0.86 |      -0.57 |            3 | 67%                     |
|         125 | all pairs            | swing         |        -0.06 |       -0.75 |      -0.65 |            3 | 100%                    |
|         125 | different-type pairs | whiff \| swing |         0.02 |       -0.61 |      -0.54 |            3 | 100%                    |
|         125 | different-type pairs | swing         |         0.51 |       -0.69 |      -0.56 |            3 | 100%                    |
|         175 | all pairs            | whiff \| swing |        -0.54 |       -0.69 |      -0.3  |           10 | 70%                     |
|         175 | all pairs            | swing         |        -0.41 |       -0.89 |      -0.67 |           10 | 100%                    |
|         175 | different-type pairs | whiff \| swing |        -0.14 |       -0.34 |       0.11 |            3 | 67%                     |
|         175 | different-type pairs | swing         |         0.19 |       -0.6  |      -0.34 |            3 | 100%                    |
|         225 | all pairs            | whiff \| swing |        -0.61 |       -0.78 |      -0.71 |            3 | 100%                    |
|         225 | all pairs            | swing         |        -0.11 |       -0.69 |      -0.58 |            3 | 100%                    |
|         225 | different-type pairs | whiff \| swing |        -0.08 |       -0.67 |      -0.46 |            3 | 100%                    |
|         225 | different-type pairs | swing         |         0.02 |       -0.47 |      -0.28 |            3 | 100%                    |

## How the model works

This repeats check 3 of the feasibility study (`feasibility_report.md`) under different settings, to see
whether its near-zero lift depends on the choices made there.

**Data.** 2025 regular-season pitches with a previous pitch in the same PA. Train March to July, test
August to October.

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

**Commit point varied.** The whole feature build is rerun with $`\tau \in \{0.125, 0.175, 0.225\}`$ s
(`features.TAU`), so the commit time is $`t_c = \max(T - \tau, 0)`$ for each value.

**Sequence tunnel features** (`features.add_features`). For the current pitch (flight time $`T`$,
path $`\mathbf p`$) and the previous pitch in the same plate appearance ($`T'`$, $`\mathbf p'`$):

1. Early separation, at the commit point. The current pitch is at $`t_c = \max(T - \tau, 0)`$; the previous
   pitch is taken at the same time since its own release, capped at its flight time,
   $`t'_c = \min(t_c, T')`$:
   ```math
   d_{dec} = \theta\big(\mathbf p(t_c),\ \mathbf p'(t'_c)\big)\ \text{deg}
   ```
2. Late separation, at the tracking limit and at equal depth (path, not timing). Let
   $`y^* = p_y(T - 0.100)`$ be the depth the current pitch reaches 0.100 s before arrival, and $`t'`$ the time
   the previous pitch reaches that depth (the formula for $`t(y)`$ above, from its release state). Then
   ```math
   d_{track} = 12\sqrt{\big(p_x(T - 0.1) - p'_x(t')\big)^2 + \big(p_z(T - 0.1) - p'_z(t')\big)^2}\ \text{in}
   ```
3. Separation at the plate:
   ```math
   d_{plate} = 12\sqrt{(x_p - x'_p)^2 + (z_p - z'_p)^2}\ \text{in}
   ```

The tunneling hypothesis is that a small $`d_{dec}`$ (the two pitches look alike when the hitter decides)
together with a large $`d_{track}`$ or $`d_{plate}`$ (they split late) produces more whiffs and called
strikes than stuff and location explain.

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

**Comparison.** For each $`\tau`$, pair set (all pairs, or different-type pairs only, where tunneling
should matter most) and outcome (whiff \| swing, swing), two models are trained on March to July and scored
on August to October:

```math
\text{gain} \times 1000 = 1000\,\big(L_{\text{BASE + PREV}} - L_{\text{BASE + PREV + tunnel}}\big)
```

So the tunnel features are judged on what they add beyond the previous pitch's type, location and
result.

**Permutation null.** To check that a gain is more than the extra flexibility of more columns, the
tested features are shuffled among pitches of the same pitcher x pitch type, and the model is refit and
scored the same way. Shuffling keeps each pitcher's pitch types and the features' spread, and breaks only
the link between a feature value and the pitch it belongs to. A real gain has to beat every null draw. The three tunnel features are shuffled within pitcher x pitch type (train and test) and the
`+tunnel` model refit: 10 draws at 175 ms on all pairs, 3 elsewhere. `null mean` and `null max` are the
mean and largest null gain over `+prev`, and `share of nulls beaten` the share of draws the real gain
exceeds.

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
| `prev_pitch_type` | Type of the previous pitch in the same PA (`none` on the first pitch) | category | `features.add_previous` |
| `prev_desc` | Result of the previous pitch, grouped: whiff, called, foul, ball, other, none | category | `feasibility.prep`, `perceptual.season` |
| `prev_loc_x` | Previous pitch's `loc_x`, using the current batter side | ft | `feasibility.prep`, `perceptual.season` |
| `prev_loc_z` | Previous pitch's height scaled to the current pitch's zone | zone heights | `feasibility.prep`, `perceptual.season` |
| `TAU` | Commit point: how long before the ball reaches the front of the plate the hitter must decide. Fixed at 0.175 s unless stated | s | `features.TAU` |
| `TRACK` | Tracking limit: 0.100 s before arrival, the last moment the hitter can still follow the ball | s | `features.TRACK` |
| `t_c` | Commit time after release, $`T - \tau`$ (floored at 0 in the geometric features, at 0.05 s in the perceptual ones) | s | `features.add_features`, `perceptual.commit_features` |
| `d_dec_deg` | Angle from the eye between this pitch at its commit point $`t_c = T-\tau`$ and the previous pitch at the same time since its release (capped at its own $`T`$) | deg | `features.add_features` |
| `d_track_in` | Distance in the x-z plane between this pitch and the previous one at the same depth: the depth this pitch reaches 0.100 s before arrival | in | `features.add_features` |
| `d_plate_in` | Distance between the two pitches' plate crossings, $`12\sqrt{\Delta x_p^2+\Delta z_p^2}`$ | in | `features.add_features` |
| `same_type` | 1 if the pitch and the previous pitch have the same `pitch_type` | 0/1 | `feasibility.prep` |
| `LightGBM` | Gradient-boosted trees. Settings everywhere unless stated: binary log-loss objective, learning rate 0.05, 63 leaves, >= 200 rows per leaf, 80% of features and 80% of rows per tree, 400 trees, seed 7 | - | `feasibility.PARAMS`, `feasibility.ROUNDS` |
| `log-loss` | $`-\frac1n\sum_i [y_i\log\hat p_i + (1-y_i)\log(1-\hat p_i)]`$ on held-out pitches; lower is better | nats per pitch | `sklearn.metrics.log_loss` |
| `gain x1000` | $`1000\,(L_{\text{without}} - L_{\text{with}})`$: how much adding the features lowers held-out log-loss. + helps | nats per 1000 pitches | computed in each script |
| `null` | The same model refit after shuffling the tested features among pitches of the same pitcher x pitch type (or group), which keeps each feature's spread but breaks its link to the pitch | - | `feasibility.shuffled`, `perceptual.shuffled` |
| `null gains` | Gain x1000 of each shuffled-feature refit; the real gain has to beat all of them | nats per 1000 pitches | computed in each script |
| `share of nulls beaten` | Share of null draws whose gain is below the real gain | % | computed in each script |
| `commit ms` | Commit point $`\tau`$ used to build the features | ms | `robustness.main` |
| `pairs` | all pairs, or different-type pairs only (`same_type` = 0) | category | `robustness.main` |
| `null mean` | Mean gain x1000 over the null draws | nats per 1000 pitches | computed in the script |
| `null max` | Largest gain x1000 over the null draws | nats per 1000 pitches | computed in the script |
| `null draws` | Number of shuffles | count | `robustness.N_NULL` |
