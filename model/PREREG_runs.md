# Model comparison by runs: pre-registration

Written and committed before any picker in this step is trained or scored.
Run once. Train on 2025, test on 2026.

## Why

`PREREG_game.md` compared pickers by CSW only. The perceptual pick tied the
standard model's (+0.16 CSW pts, CI -0.06 to +0.38). The run-value sequencer
(`PREREG_damage.md`) ranks pitches by expected runs, which changes the pick
on 46% of pitches. This step asks whether the perceptual features matter
when pitches are chosen and scored by runs.

## Pickers

All from 2025 only, average hitter, built as in `game_report.py` (each
candidate type represented by the same real 2026 pitches, placed in the
count after the previous type). Expected run value per pitch as in
`damage_sequencer.py` (`run_values`, `expected_rv`, six models).

- **RA. Perceptual, by runs**: the run-value sequencer's features
  (`hitter_sequencer.COLS_H`); pick = lowest E[RV].
- **RB. Standard, by runs**: the same minus every perceptual feature
  (`perceptual.ALL_P`).
- **RC. Stuff-only, by runs**: `game_report.COLS_C`.
- **A. Perceptual, by CSW**: highest predicted CSW from RA's CSW model.
- **D. Best pitch for the count** and **E. Pitcher's habit**: as in
  `PREREG_game.md`.

## Test

Test pitches: 2026 regular season, type, side, count and previous type in
the pitcher's grid (minimums as in `sequencing.py`). Realized value as in
`PREREG_damage.md` (2025 run values, xwOBA on contact).

For each picker: within-count realized runs per 100 pitches when the pitch
thrown matched its pick minus when it didn't (negative is better), pitcher
bootstrap, 1,000 draws.

**Primary: RA's gap minus RB's gap.** The perceptual features help pick by
runs if < 0 with a 95% CI excluding 0.

Secondary (reported, not pass/fail): RA minus RC, A, D and E; the same
comparisons on CSW; share of pitches where each pair agrees; realized runs
using actual delta_run_exp.

## Game report

The Orioles game page (game 823491) gets RA, RB and RC picks and RA's
runs saved per 100 for every candidate. Descriptive only.

## Amendment: public models (before any picker was scored)

Jeff asked how the model compares with public models. Three more pickers,
from FanGraphs' published 2025 pitch-type grades
(`datamining/pull_fangraphs_grades.py`, `fangraphs_grades_2025.csv`):

- **FS. Stuff+**, **FP. Pitching+** and **FB. PitchingBot** (overall grade):
  pick = the type in his menu with the highest 2025 grade. Our ST and SV map
  to FanGraphs' SL, everything else by its own code. No pick when fewer than
  two of his menu types have a grade.

These grades ignore count and sequence, so the same type is picked in every
cell. They are scored like the others (runs and CSW), with RA minus each and
A minus each as secondary comparisons. The primary test is unchanged.

## Variables

Added on 2026-10-30 to define the terms used above. It is documentation only: the plan, data, tests and pass bars above are unchanged. The step-by-step math is in [runs_report.md](runs_report.md#how-the-model-works).

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
