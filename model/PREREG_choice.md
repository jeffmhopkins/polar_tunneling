# Tool vs pitcher choice: pre-registration

Written and committed before running. Single run on the same 2026 test
pitches and the same 2025-only sequencer as `PREREG_validation.md`.

## Question

Pitches that matched the sequencer's top pick had a CSW 4.48 pts higher than
others in the same count. Pitchers pick pitches knowing the batter and the
situation. How much of the 4.48 pts is left once that choice is accounted for?

## Covariates (no 2026 outcomes)

- **Batter:** from 2025, his CSW-against rate overall and against the class
  of the pitch thrown (fastball FF/SI/FC, breaking SL/ST/CU/KC/SV/CS,
  offspeed CH/FS/FO/other), shrunk to the league rate with a prior of 200
  pitches. Batters without 2025 data get the league rate and a flag.
- **Situation:** outs, runners on 1st/2nd/3rd, inning (1-3, 4-6, 7+),
  platoon.
- **Previous pitch:** its class and its result (ball, called strike, whiff,
  foul, other).
- **Fixed effects:** pitcher x batter side x count.

## Estimates

**A. Adjusted gap (primary).**
- A linear probability model of CSW on matched + the covariates + the fixed
  effects, estimated by within-group demeaning.
- Report the coefficient on matched (pts) and its share of the raw 4.48.
- 95% CI from a pitcher bootstrap (500 draws).
- The tool effect **survives** if the coefficient is > 0 with a CI excluding 0.

**B. Propensity check.**
- P(matched) is modeled from the covariates + pitcher + count + previous pitch
  type, with LightGBM and 5-fold cross-fitting. No outcomes are used.
- The estimate is an inverse-propensity-weighted gap (normalized weights,
  propensity trimmed to [0.05, 0.95]), with a pitcher-bootstrap CI (500
  draws, propensities held fixed).
- Also reported: the within-count gap in each propensity tercile. The lowest
  tercile is pitches where the top pick was unlikely given the situation, so
  a pitcher who threw it there broke his usual pattern.

Reading: if A and B both stay clearly above 0, a real share of the gain is
the tool's choice, not the pitcher's. Whatever remains is still an upper
bound, because unmeasured factors (catcher, scouting, game plan) are not
controlled.

## Variables

Added on 2026-10-30 to define the terms used above. It is documentation only: the plan, data, tests and pass bars above are unchanged. The step-by-step math is in [choice_report.md](choice_report.md#how-the-model-works).

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
| `bat_all` | Batter's 2025 CSW-against rate, shrunk to the league rate with 200 prior pitches | rate | `validate_choice.batter_rates` |
| `bat_cls` | Batter's 2025 CSW-against rate on the class of the pitch thrown, shrunk with 200 prior pitches | rate | `validate_choice.batter_rates` |
| `bat_new` | 1 if the batter has no 2025 data (rates set to the league rate) | 0/1 | `validate_choice.covariates` |
| `outs_when_up` | Outs before the pitch | 0-2 | Statcast column `outs_when_up` (`datamining/outputs`) |
| `inning` | Inning | count | Statcast column `inning` (`datamining/outputs`) |
| `on_1b` | Runner id on first (missing if empty); likewise `on_2b`, `on_3b` | id | Statcast column `on_1b` (`datamining/outputs`) |
| `platoon` | 1 if batter and pitcher are on the same side (`stand == p_throws`) | 0/1 | `features.add_outcomes_and_controls` |
| `prev_desc` | Result of the previous pitch, grouped: whiff, called, foul, ball, other, none | category | `feasibility.prep`, `perceptual.season` |
| `fixed effects` | A separate intercept per group (here pitcher x side x count). Estimated by subtracting each group's mean from every variable before the regression | - | `validate_choice.within_ols` |
| `propensity` | $`e = P(\text{matched} = 1 \mid \text{situation})`$ from a cross-fitted LightGBM, trimmed to 0.05-0.95 | probability | `validate_choice.propensity` |
| `IPW gap` | $`\frac{\sum d_i y_i/e_i}{\sum d_i/e_i} - \frac{\sum (1-d_i) y_i/(1-e_i)}{\sum (1-d_i)/(1-e_i)}`$, with $`d_i`$ = matched | pts | `validate_choice.ipw_gap` |
| `propensity tercile` | Tercile of the propensity $`e`$ | category | `validate_choice.main` |
| `matched share` | Share of the tercile's pitches that matched the top pick | % | `validate_choice.main` |
| `within-count gap pts` | Within-count CSW gap in the tercile | pts | `validate_sequencer.within_count_gap` |
