# Model comparison and game report (pre-registered in PREREG_game.md)

> How every number below is computed is in [How the model works](#how-the-model-works), and every variable is defined in [Variables](#variables).

All models trained on 2025 only. Pickers: A = Perceptual sequencer; B = Standard pitch model; C = Stuff-only (Stuff+-style); D = Best pitch for the count; E = Pitcher's habit.

## Season comparison: matched minus unmatched within-count CSW, 2026

Test pitches: 470,321 from 265 pitchers (2026 regular season).

| picker | pitches | share thrown | CSW when thrown | gap pts (95% CI) |
|:--|--:|--:|--:|:--|
| A. Perceptual sequencer | 470,321 | 28.5% | 31.1% | +4.49 (+4.01 to +4.97) |
| B. Standard pitch model | 470,321 | 25.8% | 31.4% | +4.33 (+3.84 to +4.83) |
| C. Stuff-only (Stuff+-style) | 470,321 | 23.5% | 31.5% | +4.03 (+3.36 to +4.74) |
| D. Best pitch for the count | 417,749 | 31.7% | 29.8% | +2.67 (+2.17 to +3.19) |
| E. Pitcher's habit | 470,321 | 44.7% | 28.9% | +0.52 (+0.07 to +0.97) |

## A minus each other picker (same pitches)

| comparison | pitches | picks agree | A gap minus other gap (95% CI) |
|:--|--:|--:|:--|
| A vs B. Standard pitch model **(primary)** | 470,321 | 82% | +0.16 (-0.06 to +0.38) |
| A vs C. Stuff-only (Stuff+-style) | 470,321 | 44% | +0.46 (-0.12 to +1.04) |
| A vs D. Best pitch for the count | 417,749 | 36% | +1.85 (+1.26 to +2.41) |
| A vs E. Pitcher's habit | 470,321 | 31% | +3.97 (+3.35 to +4.60) |

Primary (A beats the standard model): **False**.

## Predicting the 2026 pitches actually thrown (log-loss, lower is better)

710,486 pitches. Gain x1000 vs the stuff-only model in brackets.

| model | swing | whiff (per pitch) | CSW |
|:--|--:|--:|--:|
| A. Perceptual sequencer | 0.4448 (+242.0) | 0.3138 (+41.9) | 0.4727 (+118.6) |
| B. Standard pitch model | 0.4505 (+236.4) | 0.3155 (+40.1) | 0.4748 (+116.5) |
| C. Stuff-only (Stuff+-style) | 0.6869 (+0.0) | 0.3557 (+0.0) | 0.5913 (+0.0) |

## Game: Regular Season, 2026-09-25: Baltimore Orioles 10 @ New York Yankees 2

323 pitches, 226 with picks (pitcher has a grid for that side and count). Actual CSW on those: 27.9%; model A's expected CSW of the pitches thrown: 27.3%.

| picker | matched / scored | CSW when matched | CSW otherwise | A's expected CSW if followed |
|:--|--:|--:|--:|--:|
| A. Perceptual sequencer | 59 / 226 | 29% | 28% | 32.7% |
| B. Standard pitch model | 63 / 226 | 32% | 26% | 32.4% |
| C. Stuff-only (Stuff+-style) | 42 / 226 | 33% | 27% | 31.0% |
| D. Best pitch for the count | 63 / 204 | 35% | 25% | 28.7% |
| E. Pitcher's habit | 113 / 226 | 27% | 28% | 27.6% |

CSW log-loss on the 323 game pitches with full features: A 0.5356, B 0.5433, C 0.6019.

## How the model works

This pre-registered comparison (`PREREG_game.md`) asks whether the perceptual features change which pitch
is best, by putting five pickers side by side on 2026, and builds the game page.

**Pitch features.** Every pitch is rebuilt from Statcast's 9-parameter fit, viewed from the batter's eye
and described by the controls, speed controls and perceptual features of `perceptual_report.md` and
`speed_check_report.md` (each of those reports walks through the formulas). In short: the trajectory is
$`\mathbf p(t) = \mathbf r_0 + \mathbf v_r t + \tfrac12\mathbf a t^2`$ from release to the front of the plate
at $`t = T`$, the hitter's eye is at $`\mathbf E`$, and the perceptual features describe the ball at the
commit time $`t_c = \max(T - \tau_b, 0.05)`$.

**Model** (`feasibility.PARAMS`). Each outcome $`y \in \{0, 1\}`$ gets a LightGBM classifier: an additive
sum of $`K = 400`$ regression trees on the features $`\mathbf x`$, passed through the logistic function,

```math
\hat p(\mathbf x) = \sigma\Big(\sum_{k=1}^{K} f_k(\mathbf x)\Big), \qquad \sigma(u) = \frac{1}{1 + e^{-u}}
```

where each tree $`f_k`$ is fitted to the gradient of the log-loss of the trees before it (learning rate
0.05, at most 63 leaves, at least 200 training rows per leaf, a random 80% of features and of rows per
tree, seed 7). Categorical columns (`pitch_type`, `stand`, `p_throws`, `prev_pitch_type`, `prev_desc`)
are split on directly. Missing values are allowed: a tree learns which side of each split they go to.

**Pickers.** All trained on 2025 only. A, B and C are LightGBM models (swing, whiff \| swing, CSW) placed
into the same grid; D and E are rules.

| Picker | Features or rule | Pick |
|---|---|---|
| A. Perceptual sequencer | CONTROLS + SPEED + all P | highest $`\widehat{\text{CSW}}`$ |
| B. Standard pitch model | CONTROLS + SPEED | highest $`\widehat{\text{CSW}}`$ |
| C. Stuff-only | `pitch_type`, hands, speed, `mov_x`, `pfx_z_ft`, spin, extension, release height and `rel_x`, arm angle, fastball differentials | highest $`\widehat{\text{CSW}}`$ |
| D. Best pitch for the count | the naive pick of `validation_report.md` | - |
| E. Pitcher's habit | his most-used 2026 type to that side in that count | - |

**The sequencer grid** (`sequencing.build_cells`). For each pitcher with >= 1,000 2026 pitches and each
batter side he faced >= 150 times, the menu is every type with >= 40 pitches and >= 4% usage to that
side. Each type is represented by up to 20 of his real 2026 pitches to that side (their actual shapes,
release and plate locations), and each of those pitches is placed into every count (12) and every
previous type in the menu (`none` at 0-0):

- `balls`, `strikes`, `count` from the cell, `pitch_number` $`= b + s + 1`$, `n_priorpa_thisgame` = 1
- an average hitter: `tau_b` = 0.175 s and the training-data median batter rates
- previous pitch = the pitcher's average pitch of that type to that side (its full release state, plate
  location and speed), `prev_desc` = the most common result leading into that count in the training data
- the last fastball seen = the previous pitch if it was a fastball, `fb_seen_pa` = 1 then, else 0
- `surprise_count`, `surprise_prev` from his 2026 usage, with the same smoothing as P5
- the commit-point features (P2, P3, `sep_last_fb`) recomputed for that previous pitch

The models predict for every row; a cell's value for a type is the mean over its candidate pitches:

```math
\widehat{\text{swing}} = \hat p_{\text{swing}}, \qquad
\widehat{\text{whiff}} = \hat p_{\text{swing}} \cdot \hat p_{\text{whiff}\mid\text{swing}}, \qquad
\widehat{\text{CSW}} = \hat p_{\text{csw}}
```

The top pick of a cell is the type with the highest $`\widehat{\text{CSW}}`$.

If a pitch's previous type is not in its count's grid, the cell is the average over the previous types
that are (`cell_of`).

**Test pitches.** 2026 regular-season pitches whose count and previous type are in the pitcher's grid and
whose type is in his menu (the game's own pitches are left out).

**Within-count gap** (`validate_sequencer.within_count_gap`). For a 0/1 flag $`m`$ (the pitch thrown
was the pick) and a value $`y`$ (CSW, or realized runs), in each count $`c`$ with both groups present:

```math
\Delta_c = \bar y_{c,\,m=1} - \bar y_{c,\,m=0}, \qquad
\Delta = \frac{\sum_c n_c \Delta_c}{\sum_c n_c} \times 100
```

where $`n_c`$ is the number of test pitches in count $`c`$. Comparing inside a count removes the fact that
both CSW and the top pick depend heavily on the count.

**Confidence intervals** (`validate_sequencer.boot`). Pitches from one pitcher are not independent, so
intervals come from a pitcher bootstrap: draw the test's $`P`$ pitchers with replacement, weight each
pitcher's pitches by the number of times he was drawn, recompute the statistic, repeat, and take the
2.5th and 97.5th percentiles. 2,000 draws.

**Primary.** $`\Delta_A - \Delta_B`$ on the same pitches. Pass if > 0 with the CI above 0. A minus C, D
and E are reported the same way, with the share of pitches where the pair picks the same type.

**Forecast check.** The 2026 log-loss of each model on the pitches actually thrown, for swing, whiff (per
pitch: $`\hat p_{\text{swing}}\,\hat p_{\text{whiff}\mid\text{swing}}`$ scored against whiff on all pitches)
and CSW. Brackets give the gain x1000 over C.

**Score: held-out log-loss gain.** On the $`n`$ test pitches,

```math
L = -\frac{1}{n} \sum_{i=1}^{n} \big[ y_i \log \hat p_i + (1 - y_i) \log (1 - \hat p_i) \big],
\qquad \text{gain} \times 1000 = 1000\,(L_{\text{without}} - L_{\text{with}})
```

A positive gain means the added features improved the forecast of pitches the model never saw. For scale,
adding the previous pitch's type, location and result to the controls gained 1.4 (whiff per swing) and
2.1 (swing) per 1000 in 2025.

**Game report (descriptive).** For each pitch of the chosen game, the grid of every picker (built with
lower minimums: >= 300 regular-season pitches, >= 60 to a side, >= 15 of a type), its pick, and what
happened. One game is far too small to rank models.

## Variables

| Variable | Meaning | Units | Source |
|---|---|---|---|
| `pitcher` | MLBAM id of the pitcher | id | Statcast column `pitcher` (`datamining/outputs`) |
| `stand` | Batter side, R or L. Sets the eye position and the sign $`s_b`$ | category | Statcast column `stand` (`datamining/outputs`) |
| `count` | `balls-strikes` before the pitch, e.g. `1-2` (12 values) | category | built from `balls`, `strikes` in `features.add_outcomes_and_controls` |
| `pitch_type` | Statcast pitch type code: FF four-seam, SI sinker, FC cutter, SL slider, ST sweeper, CU curve, KC knuckle curve, SV slurve, CH changeup, FS splitter, FO forkball, KN knuckleball. PO (pitchout), EP (eephus), FA (position players), CS (slow curve), SC (screwball) and UN (unknown) are dropped | category | Statcast column `pitch_type` (`datamining/outputs`) |
| `description` | Result of the pitch (ball, called_strike, swinging_strike, foul, hit_into_play, ...) | category | Statcast column `description` (`datamining/outputs`) |
| `csw` | Called strike or whiff: 1 if `whiff` or `description == called_strike` | 0/1 | `features.add_outcomes_and_controls` |
| `CSW` | Share of pitches that are a called strike or whiff (`csw` averaged) | % or pts | `csw` |
| `prev_pitch_type` | Type of the previous pitch in the same PA (`none` on the first pitch) | category | `features.add_previous` |
| `LightGBM` | Gradient-boosted trees. Settings everywhere unless stated: binary log-loss objective, learning rate 0.05, 63 leaves, >= 200 rows per leaf, 80% of features and 80% of rows per tree, 400 trees, seed 7 | - | `feasibility.PARAMS`, `feasibility.ROUNDS` |
| `M0` | Control model: LightGBM on CONTROLS = BASE + PREV + fastball differentials + `arm_angle_deg` + batter rates | - | `perceptual.CONTROLS` |
| `M1` | M0 plus all 17 perceptual features (P1-P6) | - | `perceptual.ALL_P` |
| `plate_speed_mph` | Speed of the fit at the front of the plate, $`\lVert\mathbf v(T)\rVert/1.46667`$ | mph | `speed_check.add_speed` |
| `flight_ms` | Flight time $`1000\,T`$ | ms | `speed_check.add_speed` |
| `prev_dv` | Release speed minus the previous pitch's release speed | mph | `speed_check.add_speed` |
| `prev_dtf_ms` | Flight time minus the previous pitch's flight time | ms | `speed_check.add_speed` |
| `fb_dtf_ms` | Flight time minus the primary fastball's mean flight time | ms | `speed_check.add_speed` |
| `tau_b` | P1. Hitter's commit time from bat tracking: $`0.175 \times \tilde\rho_b/\bar\rho`$, clipped to 0.125-0.225 s, where $`\rho`$ = swing length / bat speed (swings over 40 mph), $`\bar\rho`$ the league mean and $`\tilde\rho_b`$ the batter's mean shrunk to it with 50 prior swings. 0.175 s if he has none | s | `perceptual.batter_tau` |
| `surprise_count` | P5. Surprisal of the type given the count, $`-\log_2 P(\text{type} \mid \text{pitcher, side, count})`$, usage smoothed with 5 prior pitches | bits | `perceptual.surprisal` |
| `surprise_prev` | P5. Surprisal given the previous pitch type, $`-\log_2 P(\text{type} \mid \text{pitcher, side, previous type})`$; missing on a PA's first pitch | bits | `perceptual.surprisal` |
| `fb_seen_pa` | P6. Fastballs (FF, SI, FC) already thrown earlier in the PA | count | `perceptual.season` |
| `sep_last_fb` | P6. Angle from the eye at commit between this pitch and the most recent earlier fastball in the PA (same time since release); missing if none | deg | `perceptual.season` |
| `n_priorpa_thisgame` | Times the batter has already faced this pitcher in the game (times through the order minus 1). Also a P6 feature | count | Statcast column `n_priorpa_thisgame_player_at_bat` (`datamining/outputs`) |
| `pitch_number` | Pitch number within the plate appearance (1 for the first pitch). Also a P6 feature | count | Statcast column `pitch_number` (`datamining/outputs`) |
| `bat_whiff_rate` | Batter's whiff per swing in the same season, leaving out the current game, shrunk to the league rate with 100 prior swings | rate | `perceptual.batter_rates` |
| `bat_swing_rate` | Batter's swing rate in the same season, leaving out the current game, shrunk with 100 prior pitches | rate | `perceptual.batter_rates` |
| `cell` | Pitcher x batter side x count x previous pitch type. The grid holds, for each candidate type in the cell, predicted swing %, whiff % per pitch, CSW % and deception lift (and runs saved per 100 in the run-value version) | - | `sequencing.build_cells` |
| `menu` | A pitcher's candidate types to a side: >= 40 pitches and >= 4% usage in 2026; pitchers need >= 1,000 2026 pitches and >= 150 to the side | - | `sequencing.MIN_*` |
| `top pick` | The candidate type with the highest predicted CSW in the cell | - | `validate_sequencer.test_pitches` |
| `matched` | 1 if the type actually thrown equals the pick being tested | 0/1 | each validation script |
| `gap pts` | Within-count gap: in each count, CSW of matched pitches minus unmatched, then averaged over counts weighted by pitches per count | pts | `validate_sequencer.within_count_gap` |
| `95% CI` | 2.5th to 97.5th percentile of the statistic over a pitcher bootstrap: pitchers drawn with replacement, each pitcher's pitches weighted by how often he was drawn | - | `validate_sequencer.boot` |
| `pts` | Percentage points (a rate difference x100) | pts | - |
| `top-pick share` | Share of test pitches in the count that matched the top pick | % | `validate_sequencer.main` |
| `naive pick` | The pitcher's type with the highest observed 2025 CSW to that side in that count (>= 30 pitches), else his highest 2025 CSW type to that side overall | - | `validate_sequencer.naive_picks` |
| `seq_delta` | Predicted CSW of (actual previous type -> actual type) minus the mean predicted CSW of that type over all previous types in that count | probability (pts) | `validate_sequencer.test_pitches` |
| `picker A` | Perceptual sequencer: CONTROLS + SPEED + all P, highest CSW | - | `game_report.COLS_A` |
| `picker B` | Standard pitch model: CONTROLS + SPEED, no perceptual features, highest CSW | - | `game_report.COLS_B` |
| `picker C` | Stuff-only: pitch type, hands, speed, movement, spin, extension, release, arm angle, fastball differentials. No location, count or previous pitch | - | `game_report.COLS_C` |
| `picker D` | Best pitch for the count: the naive pick | - | `validate_sequencer.naive_picks` |
| `picker E` | Pitcher's habit: his most-used 2026 type to that side in that count | - | `game_report.habit_picks` |
| `picker` | A pitch-choice rule, defined under How the model works | - | the script's `LABELS` |
| `pitches` | Test pitches the statistic is computed on | count | computed in the script |
| `share thrown` | Share of test pitches where the pitcher threw the picker's choice | % | `game_report`, `runs_compare` |
| `CSW when thrown` | Actual CSW of test pitches that matched the pick | % | `game_report.season_compare` |
| `picks agree` | Share of test pitches where two pickers chose the same type | % | `game_report`, `runs_compare` |
| `log-loss` | $`-\frac1n\sum_i [y_i\log\hat p_i + (1-y_i)\log(1-\hat p_i)]`$ on held-out pitches; lower is better | nats per pitch | `sklearn.metrics.log_loss` |
| `gain x1000` | $`1000\,(L_{\text{without}} - L_{\text{with}})`$: how much adding the features lowers held-out log-loss. + helps | nats per 1000 pitches | computed in each script |
| `arm_angle_deg` | Pitcher arm angle at release | deg | Statcast column `arm_angle` (`datamining/outputs`) |
| `primary fastball` | For a pitcher x batter side: the most-used of FF, SI and FC (to that side, in the data used for the features) | - | `arsenal.group_table`, `perceptual.primary_fastball` |
| `fb_dv` | Release speed minus the primary fastball's mean release speed | mph | `arsenal.group_table`, `perceptual.season` |
| `fb_dmx` | `mov_x` minus the primary fastball's mean `mov_x` | ft | `arsenal.group_table`, `perceptual.season` |
| `fb_dmz` | `pfx_z_ft` minus the primary fastball's mean `pfx_z_ft` | ft | `arsenal.group_table`, `perceptual.season` |
| `fb_drx` | `rel_x` minus the primary fastball's mean `rel_x` | ft | `arsenal.group_table`, `perceptual.season` |
| `fb_drz` | `release_pos_z_ft` minus the primary fastball's mean release height | ft | `arsenal.group_table`, `perceptual.season` |
