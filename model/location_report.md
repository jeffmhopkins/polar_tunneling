# Location in the sequencer (pre-registered in PREREG_location.md, single run)

> How every number below is computed is in [How the model works](#how-the-model-works), and every variable is defined in [Variables](#variables).

Pickers built from 2025 only, average hitter, real previous pitch. LA = Perceptual, type + zone; LB = Standard, type + zone; LZ = League location chart.

Test pitches: 71,670 from 265 pitchers (15% sample of 2026).

## Each picker: thrown the pick (type and zone) minus not, within count

| picker | pitches | matched | runs per 100 (95% CI) | CSW pts |
|:--|--:|--:|:--|--:|
| LA. Perceptual, type + zone | 71,670 | 2.6% | -4.97 (-5.64 to -4.28) | +16.83 |
| LB. Standard, type + zone | 71,670 | 2.3% | -4.90 (-5.69 to -4.07) | +14.93 |
| LZ. League location chart | 71,670 | 1.8% | -4.39 (-5.38 to -3.33) | +16.89 |

## LA minus each other picker (same pitches)

| comparison | same type | same zone | same both | runs per 100 (95% CI) | CSW pts (95% CI) |
|:--|--:|--:|--:|:--|:--|
| LA vs LB. Standard, type + zone **(primary)** | 70% | 66% | 60% | -0.07 (-0.71 to +0.50) | +1.90 (-0.23 to +4.19) |
| LA vs LZ. League location chart | 41% | 27% | 17% | -0.58 (-1.54 to +0.45) | -0.06 (-3.31 to +3.04) |

Primary (perceptual features help choose type and location): **False**.

## Game 823491

218 pitches with location picks.

## How the model works

This pre-registered comparison (`PREREG_location.md`) has pickers choose a location as well as a type,
in each test pitch's real sequence context, as the fairest test of whether perceptual features matter for
decisions.

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

**Candidates** (`pool`). For each pitcher x side (menu rules of `sequencing.py`), a candidate is a
(type, Statcast zone) pair with >= 8 of his 2026 pitches of that type to that side in that zone, zones 1-9
inside the strike zone and 11-14 outside. Each candidate is represented by 4 of those real pitches (fixed
seed).

**Real context** (`score`). Each test pitch is scored in its own situation: its count, pitch number, times
through the order, the actual previous pitch (type, full trajectory, plate location, result) and the last
fastball in the PA. Every candidate is placed into that context with the average hitter, and the
speed differentials, surprisal and commit-point features are recomputed for it. A candidate's value is the
mean over its 4 pitches.

**Pickers.**

| Picker | Rule |
|---|---|
| LA. Perceptual, type + zone | candidate with the lowest $`E[RV]`$ from the RA models (model H features) |
| LB. Standard, type + zone | the same with the RB models (no perceptual features) |
| LZ. League location chart | candidate with the lowest 2025 league mean realized RV for type x zone x count x platoon (cells with >= 50 pitches), else type x zone x platoon. No model and no sequence |

**Test pitches.** A random 15% of 2026 regular-season pitches with a zone (seed 7), by pitchers with a menu,
of a menu type, with the previous pitch in the data. A pitch matches a picker when both its type and its
zone equal the pick.

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

**Primary.** $`\Delta_{LA} - \Delta_{LB}`$ with $`y`$ = 100 x realized RV. Pass if < 0 with the CI below 0.
Also LA minus LZ, the same comparisons on CSW, and how often LA and LB agree on type, zone and both.

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
| `RA` | Perceptual by runs: model H features, lowest E[RV] | - | `runs_compare.SETS` |
| `RB` | Standard by runs: model H features minus all perceptual features, lowest E[RV] | - | `runs_compare.SETS` |
| `LA` | Perceptual, type + zone: candidate (type, zone) with the lowest E[RV], RA features | - | `location_compare.SETS` |
| `LB` | Standard, type + zone: the same with RB features | - | `location_compare.SETS` |
| `LZ` | League location chart: candidate with the lowest 2025 league mean realized RV for type x zone x count x platoon (>= 50 pitches, else type x zone x platoon) | - | `location_compare.league_chart` |
| `picker` | A pitch-choice rule, defined under How the model works | - | the script's `LABELS` |
| `pitches` | Test pitches the statistic is computed on | count | computed in the script |
| `same type, same zone, same both` | Share of test pitches where LA and the other picker chose the same type, zone, or both | % | `location_compare.season` |
