# Tunnel score feasibility (2025, sequence pairs)

> How every number below is computed is in [How the model works](#how-the-model-works), and every variable is defined in [Variables](#variables).

Regular-season pitches with a previous pitch in the same PA: 524,357. Commit point 175 ms before the current pitch reaches the plate; tracking limit 100 ms.

## 1. Noise floor and free variation

|                |   d_dec_deg p10 |   d_dec_deg p50 |   d_dec_deg p90 |   d_track_in p10 |   d_track_in p50 |   d_track_in p90 |   d_plate_in p10 |   d_plate_in p50 |   d_plate_in p90 |
|:---------------|----------------:|----------------:|----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|
| different type |            0.77 |            2.06 |            4.39 |             5.07 |            13.09 |            24.51 |             7.28 |            18.58 |            34.34 |
| same type      |            0.6  |            1.59 |            3.17 |             4.19 |            10.95 |            21.19 |             5.43 |            14.2  |            27.31 |

Share of commit-point separation (different-type pairs) predictable from other inputs, held-out R²:

|                                 |    R² |
|:--------------------------------|------:|
| types + both plate locations    | 0.706 |
| + plate separation (d_plate_in) | 0.714 |

## 2. Raw relationship (different-type pairs, no model)

**By decile of d_dec_deg**

| bin              |   pitches |   swing% |   whiff/swing% |   CSW% |
|:-----------------|----------:|---------:|---------------:|-------:|
| (0.00311, 0.772] |     33954 |     64.6 |           21.4 |   24.5 |
| (0.772, 1.133]   |     33954 |     62   |           22.1 |   24.7 |
| (1.133, 1.446]   |     33954 |     59   |           23   |   24.7 |
| (1.446, 1.748]   |     33954 |     56.8 |           23.7 |   24.5 |
| (1.748, 2.065]   |     33954 |     54.1 |           24   |   24   |
| (2.065, 2.417]   |     33954 |     51.4 |           25.2 |   23.6 |
| (2.417, 2.843]   |     33954 |     48.4 |           26.1 |   23.3 |
| (2.843, 3.417]   |     33954 |     46.3 |           27.1 |   22.8 |
| (3.417, 4.389]   |     33954 |     43.7 |           27.8 |   22.6 |
| (4.389, 42.007]  |     33954 |     42.7 |           27.6 |   24.4 |

**By decile of d_track_in**

| bin              |   pitches |   swing% |   whiff/swing% |   CSW% |
|:-----------------|----------:|---------:|---------------:|-------:|
| (0.0498, 5.07]   |     33954 |     65.7 |           21.9 |   27.8 |
| (5.07, 7.379]    |     33954 |     63.7 |           22.2 |   27.4 |
| (7.379, 9.346]   |     33954 |     61.8 |           22.8 |   27   |
| (9.346, 11.215]  |     33954 |     60   |           22.9 |   26.3 |
| (11.215, 13.091] |     33954 |     57.5 |           23.4 |   25.4 |
| (13.091, 15.102] |     33954 |     54.8 |           24.5 |   24.5 |
| (15.102, 17.376] |     33954 |     50.9 |           25.1 |   23.5 |
| (17.376, 20.216] |     33954 |     46.2 |           27.7 |   22.5 |
| (20.216, 24.509] |     33954 |     40.1 |           28.9 |   19.9 |
| (24.509, 83.911] |     33954 |     28.4 |           32.8 |   14.7 |

**Whiff/swing % by early (commit-point) x late (tracking-limit) separation terciles**

| dec3        |   late close |   late mid |   late far |
|:------------|-------------:|-----------:|-----------:|
| early close |         21.2 |       23.6 |       24.6 |
| early mid   |         22   |       24   |       29.1 |
| early far   |         26.4 |       24.5 |       29.7 |

## 3. Held-out lift and permutation null (train Mar-Jul, test Aug-Oct)

Log-loss improvement x 1000 over the base model. Null = tunnel features shuffled within pitcher x pitch type, 5 draws.

| outcome       |   test n |   base log-loss |   +tunnel (no prev) |   +prev |   +prev +tunnel | null +prev +tunnel (mean, max)   |
|:--------------|---------:|----------------:|--------------------:|--------:|----------------:|:---------------------------------|
| whiff \| swing |    92493 |          0.4678 |               -0.11 |    1.39 |            0.86 | 0.52, 1.00                       |
| swing         |   173377 |          0.4347 |                0.55 |    2.14 |            1.73 | 1.28, 1.59                       |
| CSW           |   173377 |          0.4788 |               -0.65 |    0.18 |           -0.05 | -0.35, -0.04                     |

## 4. Pitcher x pitch-type level (whiff | swing)

|                                                             | value       |
|:------------------------------------------------------------|:------------|
| pitcher x type groups (>=75 swings per half)                | 447         |
| SD of mean tunnel value, whiff pct points (>=150 swings)    | 0.7         |
| split-half r, tunnel value (Spearman-Brown)                 | 0.88 (0.93) |
| split-half r, residual whiff% (Spearman-Brown)              | 0.37 (0.54) |
| tunnel value in one half vs residual whiff% in the other, r | 0.015       |

**Largest pitcher x pitch-type tunnel values (>=150 swings), whiff pct points**

|                |   n |   tv |
|:---------------|----:|-----:|
| (605135, 'CU') | 188 | 6.69 |
| (808967, 'CU') | 170 | 3.48 |
| (571510, 'CH') | 331 | 2.15 |
| (624133, 'CU') | 164 | 1.81 |
| (445276, 'FC') | 309 | 1.71 |
| (682052, 'SL') | 225 | 1.43 |
| (664208, 'CU') | 203 | 1.42 |
| (669920, 'CH') | 211 | 1.36 |
| (608337, 'CH') | 265 | 1.35 |
| (607259, 'CH') | 224 | 1.32 |

## How the model works

This study asks whether a geometric tunnel score between consecutive pitches carries any information
that stuff, location and count don't. It uses 2025 only: train on March to July, test on August to
October.

**Data.** 2025 regular-season pitches (`game_type == "R"`) that have a previous pitch in the same plate
appearance (524,357 pitches).

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

**Check 1. Noise floor and free variation.** The table gives the 10th, 50th and 90th percentiles of each
separation for same-type and different-type pairs. Then, on different-type pairs, a LightGBM regression
(squared-error loss, otherwise the settings below) predicts $`d_{dec}`$ from the two pitch types, batter
side and both plate locations (`pitch_type`, `prev_pitch_type`, `stand`, `loc_x`, `loc_z`, `prev_loc_x`,
`prev_loc_z`), trained on March to July and scored on August to October with

```math
R^2 = 1 - \frac{\sum_i (d_i - \hat d_i)^2}{\sum_i (d_i - \bar d)^2}
```

A high $`R^2`$ means most of the early separation is fixed by which two pitches were thrown and where they
ended up, so little is left for a pitcher to control.

**Check 2. Raw relationship.** No model. Different-type pairs are cut into ten equal-count bins of
`d_dec_deg` (and of `d_track_in`), and each bin reports swing %, whiff per swing % and CSW %. The last
table crosses terciles of early separation with terciles of late separation, on swings, and reports
whiff per swing %. Tunneling predicts the most whiffs in the early close, late far cell.

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

**Check 3. Held-out lift.** For each outcome (whiff \| swing on swings; swing and CSW on all pitches),
four models are trained on March to July and scored on August to October:

| Model | Features |
|---|---|
| base | BASE |
| +tunnel (no prev) | BASE + `d_dec_deg`, `d_track_in`, `d_plate_in` |
| +prev | BASE + PREV |
| +prev +tunnel | BASE + PREV + tunnel features |

Each column of the table is $`1000\,(L_{\text{base}} - L_{\text{model}})`$. The tunnel features are judged
by `+prev +tunnel` against `+prev`.

**Permutation null.** To check that a gain is more than the extra flexibility of more columns, the
tested features are shuffled among pitches of the same pitcher x pitch type, and the model is refit and
scored the same way. Shuffling keeps each pitcher's pitch types and the features' spread, and breaks only
the link between a feature value and the pitch it belongs to. A real gain has to beat every null draw. Here the three tunnel features are shuffled within pitcher x pitch type across the whole season
(train and test), and the `+prev +tunnel` model is refit, 5 times. The table gives the mean and maximum
null gain over base.

**Check 4. Pitcher x pitch type.** Could the tunnel score rank pitchers even if it doesn't help
pitch-level forecasts? On swings, with $`M`$ the `+prev +tunnel` whiff model:

1. Tunnel value of each swing: the model's whiff probability with the pitch's real tunnel features minus
   with the three features set to the median for its pitch type,
   $`tv_i = M(\mathbf x_i) - M(\mathbf x_i^{\text{median}})`$.
2. Residual whiff of each swing against a model without tunnel features (BASE + PREV), cross-fitted so no
   swing is scored by a model that saw it: trained on March to July to score August to October, and the
   reverse. $`resid_i = whiff_i - \hat p_i`$.
3. Games are split by `game_pk` parity into halves A and B. Each pitcher x pitch type with >= 75 swings in
   both halves gets its mean $`tv`$ and mean $`resid`$ in each half.
4. Split-half reliability: $`r_{tv} = \text{corr}(tv_A, tv_B)`$, stepped up to the full season with
   Spearman-Brown, $`2r/(1 + r)`$. The same for residual whiff.
5. Predictive value: $`\tfrac12[\text{corr}(tv_A, resid_B) + \text{corr}(tv_B, resid_A)]`$. A real pitcher
   skill would show up here: a pitcher whose pitches tunnel well in one half should beat his expected
   whiffs in the other.

The SD row is the spread of mean $`tv`$ (x100, pct points) over pitcher x types with >= 150 swings, and
the last table lists the largest of them.

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
| `tv` | Tunnel value of a swing: predicted whiff probability with the pitch's real tunnel features minus with the features set to its pitch type's median | probability (shown in pct points) | `feasibility.check_pitchers` |
| `resid` | Residual: actual outcome (0/1) minus a model's predicted probability from a model without the features being tested | probability (shown in pct points) | `feasibility`, `arsenal`, `perceptual` |
| `LightGBM` | Gradient-boosted trees. Settings everywhere unless stated: binary log-loss objective, learning rate 0.05, 63 leaves, >= 200 rows per leaf, 80% of features and 80% of rows per tree, 400 trees, seed 7 | - | `feasibility.PARAMS`, `feasibility.ROUNDS` |
| `log-loss` | $`-\frac1n\sum_i [y_i\log\hat p_i + (1-y_i)\log(1-\hat p_i)]`$ on held-out pitches; lower is better | nats per pitch | `sklearn.metrics.log_loss` |
| `gain x1000` | $`1000\,(L_{\text{without}} - L_{\text{with}})`$: how much adding the features lowers held-out log-loss. + helps | nats per 1000 pitches | computed in each script |
| `null` | The same model refit after shuffling the tested features among pitches of the same pitcher x pitch type (or group), which keeps each feature's spread but breaks its link to the pitch | - | `feasibility.shuffled`, `perceptual.shuffled` |
| `R²` | $`1 - \sum(y-\hat y)^2/\sum(y-\bar y)^2`$ on held-out rows | share | `sklearn.metrics.r2_score` |
| `split-half r` | Pearson correlation of a group's value in odd-`game_pk` games with its value in even-`game_pk` games | r | computed in each script |
| `Spearman-Brown` | $`2r/(1+r)`$: the split-half r stepped up to the reliability of the full season | r | `feasibility.check_pitchers` |
| `p10` | 10th percentile (likewise p50 median, p90 90th) | - | - |
| `pts` | Percentage points (a rate difference x100) | pts | - |
| `test n` | Held-out pitches scored (swings only for whiff \| swing) | count | computed in the script |
| `base log-loss` | Held-out log-loss of the base model | nats per pitch | computed in the script |
| `+tunnel (no prev)` | Gain x1000 of BASE + tunnel features over BASE | nats per 1000 pitches | `feasibility.check_lift` |
| `+prev` | Gain x1000 of BASE + PREV over BASE | nats per 1000 pitches | `feasibility.check_lift` |
| `+prev +tunnel` | Gain x1000 of BASE + PREV + tunnel features over BASE | nats per 1000 pitches | `feasibility.check_lift` |
| `null +prev +tunnel (mean, max)` | Mean and largest gain x1000 over BASE of the +prev +tunnel model refit with shuffled tunnel features | nats per 1000 pitches | `feasibility.check_lift` |
| `dec3, late3` | Terciles of `d_dec_deg` (early close, mid, far) and of `d_track_in` (late close, mid, far) | category | `feasibility.check_raw` |
