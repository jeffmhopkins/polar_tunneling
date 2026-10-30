# Model comparison by runs (pre-registered in PREREG_runs.md, single run)

> How every number below is computed is in [How the model works](#how-the-model-works), and every variable is defined in [Variables](#variables).

All pickers built from 2025 only, average hitter. RA = Perceptual, by runs; RB = Standard, by runs; RC = Stuff-only, by runs; A = Perceptual, by CSW; D = Best pitch for the count; E = Pitcher's habit; FS = FanGraphs Stuff+ (2025); FP = FanGraphs Pitching+ (2025); FB = PitchingBot (2025).

Test pitches: 470,321 from 265 pitchers (2026 regular season).

## Each picker: thrown the pick minus not, within count

Runs per 100 pitches (negative is better) and CSW pts (positive is better).

| picker | pitches | share thrown | runs per 100 (95% CI) | CSW pts |
|:--|--:|--:|:--|--:|
| RA. Perceptual, by runs | 470,321 | 32.0% | -0.60 (-0.72 to -0.49) | +1.15 |
| RB. Standard, by runs | 470,321 | 28.9% | -0.58 (-0.68 to -0.47) | +1.25 |
| RC. Stuff-only, by runs | 470,321 | 28.7% | -0.47 (-0.60 to -0.35) | +1.19 |
| A. Perceptual, by CSW | 470,321 | 28.7% | -0.52 (-0.63 to -0.40) | +4.19 |
| D. Best pitch for the count | 417,749 | 31.7% | -0.37 (-0.49 to -0.24) | +2.67 |
| E. Pitcher's habit | 470,321 | 44.7% | -0.53 (-0.62 to -0.42) | +0.52 |
| FS. FanGraphs Stuff+ (2025) | 425,045 | 26.6% | -0.39 (-0.52 to -0.24) | +2.84 |
| FP. FanGraphs Pitching+ (2025) | 425,045 | 27.6% | -0.36 (-0.50 to -0.23) | +1.22 |
| FB. PitchingBot (2025) | 425,045 | 28.5% | -0.29 (-0.43 to -0.15) | +1.21 |

## RA minus each other picker (same pitches)

| comparison | picks agree | runs per 100 (95% CI) | actual delta_run_exp per 100 | CSW pts |
|:--|--:|:--|--:|--:|
| RA vs RB. Standard, by runs **(primary)** | 78% | -0.03 (-0.12 to +0.06) | -0.07 | -0.10 |
| RA vs RC. Stuff-only, by runs | 36% | -0.13 (-0.29 to +0.01) | -0.30 | -0.04 |
| RA vs A. Perceptual, by CSW | 56% | -0.09 (-0.21 to +0.04) | -0.14 | -3.04 |
| RA vs D. Best pitch for the count | 31% | -0.21 (-0.36 to -0.07) | -0.23 | -1.58 |
| RA vs E. Pitcher's habit | 40% | -0.08 (-0.23 to +0.07) | +0.07 | +0.64 |
| RA vs FS. FanGraphs Stuff+ (2025) | 29% | -0.19 (-0.35 to +0.00) | -0.43 | -1.81 |
| RA vs FP. FanGraphs Pitching+ (2025) | 31% | -0.21 (-0.38 to -0.04) | -0.40 | -0.19 |
| RA vs FB. PitchingBot (2025) | 32% | -0.29 (-0.43 to -0.12) | -0.36 | -0.18 |

## Perceptual CSW pick (A) minus the public picks, on CSW

| comparison | picks agree | CSW pts (95% CI) |
|:--|--:|:--|
| A vs FS. FanGraphs Stuff+ (2025) | 37% | +1.32 (+0.67 to +2.02) |
| A vs FP. FanGraphs Pitching+ (2025) | 31% | +2.94 (+2.11 to +3.77) |
| A vs FB. PitchingBot (2025) | 33% | +2.95 (+2.12 to +3.78) |

Primary (perceptual features help pick by runs): **False**.

## Game 823491

Pitches where the pitcher threw each runs picker's choice: RA 82 of 226, RB 73 of 226, RC 63 of 226.

## How the model works

This pre-registered comparison (`PREREG_runs.md`) asks whether the perceptual features change the best
pitch when pitches are chosen and scored by runs, and how the picks compare with FanGraphs' public pitch
grades.

**Pitch features.** Every pitch is rebuilt from Statcast's 9-parameter fit, viewed from the batter's eye
and described by the controls, speed controls and perceptual features of `perceptual_report.md` and
`speed_check_report.md` (each of those reports walks through the formulas). In short: the trajectory is
$`\mathbf p(t) = \mathbf r_0 + \mathbf v_r t + \tfrac12\mathbf a t^2`$ from release to the front of the plate
at $`t = T`$, the hitter's eye is at $`\mathbf E`$, and the perceptual features describe the ball at the
commit time $`t_c = \max(T - \tau_b, 0.05)`$.

**Expected run value** (`damage_sequencer.run_values`, `expected_rv`). From the 2025
`delta_run_exp` (runs, from the batting team's side, so negative is good for the pitcher):

- $`RV(c, o)`$ = mean `delta_run_exp` of pitches in count $`c`$ with non-contact outcome
  $`o \in \{\text{ball}, \text{cs}, \text{whiff}, \text{foul}\}`$. This carries walks, strikeouts and
  two-strike fouls.
- Balls in play: a least-squares fit over training balls in play,
  $`\text{delta\_run\_exp} = \alpha_c + \beta\,\text{xwOBA}`$, with one intercept per count. So
  $`RV_{bip}(c, x) = \alpha_c + \beta x`$.

Six LightGBM models give $`p_s = P(\text{swing})`$, $`p_w = P(\text{whiff} \mid \text{swing})`$,
$`p_f = P(\text{foul} \mid \text{contact})`$, $`p_{cs} = P(\text{called strike} \mid \text{take})`$,
$`P(\text{CSW})`$ and $`\hat x = E[\text{xwOBA} \mid \text{ball in play}]`$ (a regression, squared-error
loss). Then

```math
E[RV] = (1 - p_s)\big[p_{cs}\, RV(c, \text{cs}) + (1 - p_{cs})\, RV(c, \text{ball})\big]
      + p_s \Big[p_w\, RV(c, \text{whiff}) + (1 - p_w)\big(p_f\, RV(c, \text{foul}) + (1 - p_f)\, RV_{bip}(c, \hat x)\big)\Big]
```

Runs saved per 100 pitches $`= -100\,E[RV]`$.

**Realized value of a test pitch** (`realized_rv`): $`RV(c, o)`$ for a non-contact outcome, and
$`RV_{bip}(c, \text{xwOBA})`$ at the pitch's actual xwOBA for a ball in play (the training mean ball-in-play
value for that count if xwOBA is missing). Contact is valued by how hard and at what angle the ball was hit,
not by where it landed.

**Model** (`feasibility.PARAMS`). Each outcome $`y \in \{0, 1\}`$ gets a LightGBM classifier: an additive
sum of $`K = 400`$ regression trees on the features $`\mathbf x`$, passed through the logistic function,

```math
\hat p(\mathbf x) = \sigma\Big(\sum_{k=1}^{K} f_k(\mathbf x)\Big), \qquad \sigma(u) = \frac{1}{1 + e^{-u}}
```

where each tree $`f_k`$ is fitted to the gradient of the log-loss of the trees before it (learning rate
0.05, at most 63 leaves, at least 200 training rows per leaf, a random 80% of features and of rows per
tree, seed 7). Categorical columns (`pitch_type`, `stand`, `p_throws`, `prev_pitch_type`, `prev_desc`)
are split on directly. Missing values are allowed: a tree learns which side of each split they go to.

**Pickers.** All from 2025 only, average hitter, using the grid below. Each model set has the six models of
the run-value formula.

| Picker | Features or rule | Pick |
|---|---|---|
| RA. Perceptual, by runs | model H: CONTROLS + SPEED + all P + hitter class rates | lowest $`E[RV]`$ |
| RB. Standard, by runs | model H minus every perceptual feature | lowest $`E[RV]`$ |
| RC. Stuff-only, by runs | the stuff-only features of `game_report.md` | lowest $`E[RV]`$ |
| A. Perceptual, by CSW | RA's CSW model | highest $`\widehat{\text{CSW}}`$ |
| D. Best pitch for the count | naive pick, `validation_report.md` | - |
| E. Pitcher's habit | most-used 2026 type to that side in that count | - |
| FS, FP, FB | FanGraphs Stuff+, Pitching+, PitchingBot 2025 grades by pitch type (ST and SV read as SL) | his menu type with the highest grade, in every cell; none if fewer than two are graded |

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

The models predict (average hitter) for every row; a cell's value for a type is the mean over its candidate pitches:

```math
\widehat{\text{swing}} = \hat p_{\text{swing}}, \qquad
\widehat{\text{whiff}} = \hat p_{\text{swing}} \cdot \hat p_{\text{whiff}\mid\text{swing}}, \qquad
\widehat{\text{CSW}} = \hat p_{\text{csw}}
```

The top pick of a cell is the type with the highest $`\widehat{\text{CSW}}`$.

**Test pitches.** 2026 regular-season pitches in the pitcher's grid (count, previous type and type), the
game's own pitches left out. The value of each is its realized RV.

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
2.5th and 97.5th percentiles. 1,000 draws.

**Primary.** $`\Delta_{RA} - \Delta_{RB}`$ with $`y`$ = 100 x realized RV (runs per 100 pitches; negative is
better). Pass if < 0 with the CI below 0. Also reported for each picker: its own $`\Delta`$ on runs and on CSW,
RA minus every other picker (also with actual `delta_run_exp`), and A minus each public picker on CSW.

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
| `batter` | MLBAM id of the batter | id | Statcast column `batter` (`datamining/outputs`) |
| `class` | Pitch class: fb (FF, SI, FC), brk (SL, ST, CU, KC, SV, CS), off (everything else) | category | `validate_choice.pclass`, `hitter_sequencer.pclass` |
| `bat_cls_whiff` | Hitter's whiff per swing against the class of the pitch, shrunk to the league class rate with 100 prior swings | rate | `hitter_sequencer.shrink` |
| `bat_cls_swing` | Hitter's swing rate against the class, shrunk with 100 prior pitches | rate | `hitter_sequencer.shrink` |
| `bat_cls_chase` | Hitter's swing rate on out-of-zone pitches (`zone` > 9) of the class, shrunk with 100 prior out-of-zone pitches | rate | `hitter_sequencer.shrink` |
| `zone` | Statcast zone of the plate crossing: 1-9 inside the strike zone (3 x 3 grid), 11-14 the four outside quadrants | category | Statcast column `zone` (`datamining/outputs`) |
| `model H` | The sequencer model (CONTROLS + SPEED + all P) plus the three `bat_cls_*` features | - | `hitter_sequencer.COLS_H` |
| `adj` | Batter adjustment for a pitcher x side x batter x count x type: mean predicted CSW over the candidate pitches with this hitter's features minus with the average hitter's (previous pitch = primary fastball, none at 0-0) | pts | `hitter_sequencer.build` |
| `side pick` | Top pick for the average hitter | - | `hitter_sequencer.validate` |
| `batter pick` | Type with the highest side-only CSW + `adj` for this hitter | - | `hitter_sequencer.validate` |
| `log-loss` | $`-\frac1n\sum_i [y_i\log\hat p_i + (1-y_i)\log(1-\hat p_i)]`$ on held-out pitches; lower is better | nats per pitch | `sklearn.metrics.log_loss` |
| `gain x1000` | $`1000\,(L_{\text{without}} - L_{\text{with}})`$: how much adding the features lowers held-out log-loss. + helps | nats per 1000 pitches | computed in each script |
| `delta_run_exp` | Change in the batting team's run expectancy caused by the pitch (negative is good for the pitcher) | runs | Statcast column `delta_run_exp` (`datamining/outputs`) |
| `xwoba` | Expected wOBA of the plate appearance, on its last pitch: from exit speed and launch angle for a ball in play, the wOBA weight for a walk or HBP, 0 for a strikeout | wOBA scale | Statcast column `estimated_woba_using_speedangle` (`datamining/outputs`) |
| `outcome` | Pitch outcome class: cs (called strike), ball (any other take), whiff, foul, bip (ball in play) | category | `damage_sequencer.add_outcomes` |
| `RV(count, outcome)` | 2025 mean `delta_run_exp` for a count and a non-contact outcome | runs | `damage_sequencer.run_values` |
| `RV_bip` | Run value of a ball in play: $`\alpha_{\text{count}} + \beta\,\text{xwOBA}`$, fitted by least squares on 2025 balls in play | runs | `damage_sequencer.run_values` |
| `E[RV]` | Expected run value of a pitch from the six model probabilities (formula in the walkthrough). Negative is good for the pitcher | runs | `damage_sequencer.expected_rv` |
| `runs saved per 100` | $`-100\,E[RV]`$ | runs per 100 pitches | `damage_sequencer.build` |
| `realized RV` | Value of what happened: RV(count, outcome) for non-contact pitches, RV_bip at the pitch's actual xwOBA for balls in play | runs | `damage_sequencer.realized_rv` |
| `runs per 100` | Within-count gap of realized RV x100 (or actual `delta_run_exp` x100 where stated). Negative is fewer runs | runs per 100 pitches | `damage_sequencer`, `runs_compare` |
| `CSW pick` | Highest predicted CSW (plus hitter adjustment) | - | `damage_sequencer.validate` |
| `RV pick` | Lowest predicted E[RV] (plus hitter adjustment) | - | `damage_sequencer.validate` |
| `R^2` | $`1 - \sum(y-\hat y)^2/\sum(y-\bar y)^2`$ of xwOBA on 2026 balls in play | share | `damage_sequencer.validate` |
| `picker A` | Perceptual sequencer: CONTROLS + SPEED + all P, highest CSW | - | `game_report.COLS_A` |
| `picker D` | Best pitch for the count: the naive pick | - | `validate_sequencer.naive_picks` |
| `picker E` | Pitcher's habit: his most-used 2026 type to that side in that count | - | `game_report.habit_picks` |
| `RA` | Perceptual by runs: model H features, lowest E[RV] | - | `runs_compare.SETS` |
| `RB` | Standard by runs: model H features minus all perceptual features, lowest E[RV] | - | `runs_compare.SETS` |
| `RC` | Stuff-only by runs: C's features, lowest E[RV] | - | `runs_compare.SETS` |
| `FS` | FanGraphs Stuff+ (2025): his menu type with the highest grade, in every cell (`sp_s_<type>`) | - | `fangraphs_grades_2025.csv` |
| `FP` | FanGraphs Pitching+ (2025), `sp_p_<type>` | - | `fangraphs_grades_2025.csv` |
| `FB` | PitchingBot overall (2025), `pb_o_<type>` | - | `fangraphs_grades_2025.csv` |
| `picker` | A pitch-choice rule, defined under How the model works | - | the script's `LABELS` |
| `pitches` | Test pitches the statistic is computed on | count | computed in the script |
| `share thrown` | Share of test pitches where the pitcher threw the picker's choice | % | `game_report`, `runs_compare` |
| `picks agree` | Share of test pitches where two pickers chose the same type | % | `game_report`, `runs_compare` |
