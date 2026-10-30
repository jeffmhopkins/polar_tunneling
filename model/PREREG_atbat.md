# Whole at-bat test: pre-registration

Written and committed before any 2026 at-bat outcome is compared with the
sequencer's picks. Run once.

## Question

Pitches that match the sequencer's top pick get more called strikes and
whiffs (`validation_report.md`, `choice_report.md`). Does following the tool
across an at-bat also produce more strikeouts and weaker contact, or does the
per-pitch CSW gain wash out (for example because the picks get hit hard when
they are put in play)?

## Setup

- Sequencer: the 2025-only grid (`sequencing_2025model.json`), the same one
  used in `PREREG_validation.md` and `PREREG_choice.md`.
- Plate appearances: 2026 regular season, pitcher and batter side in the grid,
  ending in a strikeout, walk, hit by pitch or ball in play. Sacrifice bunts,
  catcher interference, intentional walks and truncated PAs are dropped.
- A pitch is **scored** when its count x previous-type cell is in the grid.
  matched = 1 if the pitch type thrown is that cell's top pick, else 0 (a type
  the grid doesn't list counts as not matched). PAs need >= 1 scored pitch.
- **Adherence** of a PA = mean over its scored pitches of (matched minus the
  pitcher's 2026 match rate in that pitcher x side x count). This removes the
  fact that the top pick is thrown more often in some counts, so a PA is not
  credited with adherence just because it reached a two-strike count. The base
  rates use only which pitch was thrown, never results.
- Effects are reported per +25 points of adherence (about a quarter more of
  the PA's pitches following the tool than that pitcher usually does).

## Outcomes (2026)

- K: PA ends in a strikeout.
- xwOBA: Statcast's PA-level expected wOBA (K = 0, walk/HBP = their weights,
  ball in play = from exit speed and angle).
- Contact xwOBA: xwOBA on PAs that end in a ball in play.
- Also: walk + HBP rate, hard-hit rate (exit speed >= 95 mph) on balls in
  play, actual wOBA.

## Controls (no 2026 outcomes)

- Fixed effects: pitcher x batter side.
- Batter, from 2025, shrunk to the league rate: K rate and walk + HBP rate
  (prior 200 PA), xwOBA per PA (prior 200 PA), contact xwOBA (prior 100 balls
  in play). Batters without 2025 data get the league rate and a flag.
- Situation: outs, runners on 1st/2nd/3rd, inning (1-3, 4-6, 7+), platoon,
  times the batter has faced the pitcher earlier in the game.

These are the controls from `PREREG_choice.md`, with the batter rates changed
to match the PA outcomes.

## Estimates

A linear probability (or linear, for xwOBA) model of each outcome on
adherence + controls, estimated by within-group demeaning on the fixed
effects. 95% CIs from a pitcher bootstrap (500 draws).

### Primary (three tests)

1. **More strikeouts:** K coefficient > 0, CI excluding 0.
2. **Fewer runs:** xwOBA coefficient < 0, CI excluding 0.
3. **Weaker contact:** contact xwOBA coefficient < 0, CI excluding 0. If the
   CI includes 0 but its upper end is below +0.005 per +25 pts, report that
   the picks are not hit harder, without calling it a pass.

### Secondary (reported, not pass/fail)

- **First pitch only.** The same models with adherence replaced by "the 0-0
  pitch matched the top pick" (minus that pitcher x side's 0-0 match rate).
  The first pitch is chosen before anything happens in the PA, so the count
  path cannot feed back into the exposure.
- **Pitch-level contact check.** On balls in play, contact xwOBA of pitches
  that matched the top pick vs not, with pitcher x side x count fixed effects
  and the same controls. This asks directly whether top picks get hit harder
  when they are put in play.
- **Implied vs observed K.** For scale, a simple chain: a = the adherence
  coefficient on the PA's number of CSW pitches, b = the coefficient of K on
  the number of CSW pitches (same controls and fixed effects). a x b is the K
  change a CSW-only effect would suggest; compare it with the K coefficient.
  This is descriptive.
- Walk + HBP rate, hard-hit rate, actual wOBA, and a table of adjusted
  outcomes by adherence quartile.

## Caveats stated in advance

This is observational, like the earlier checks. Catcher, scouting and game
plans are not measured, so the estimates are upper bounds on what the tool
adds. Within a PA, later pitches depend on earlier results, which is why the
first-pitch version is reported alongside.

## Variables

Added on 2026-10-30 to define the terms used above. It is documentation only: the plan, data, tests and pass bars above are unchanged. The step-by-step math is in [atbat_report.md](atbat_report.md#how-the-model-works).

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
| `events` | Result of the plate appearance, on its last pitch (strikeout, walk, single, ...) | category | Statcast column `events` (`datamining/outputs`) |
| `xwoba` | Expected wOBA of the plate appearance, on its last pitch: from exit speed and launch angle for a ball in play, the wOBA weight for a walk or HBP, 0 for a strikeout | wOBA scale | Statcast column `estimated_woba_using_speedangle` (`datamining/outputs`) |
| `woba_value` | Actual wOBA value of the PA's result | wOBA scale | Statcast column `woba_value` (`datamining/outputs`) |
| `launch_speed_mph` | Exit velocity on a ball in play | mph | Statcast column `launch_speed` (`datamining/outputs`) |
| `outs_when_up` | Outs before the pitch | 0-2 | Statcast column `outs_when_up` (`datamining/outputs`) |
| `inning` | Inning | count | Statcast column `inning` (`datamining/outputs`) |
| `on_1b` | Runner id on first (missing if empty); likewise `on_2b`, `on_3b` | id | Statcast column `on_1b` (`datamining/outputs`) |
| `platoon` | 1 if batter and pitcher are on the same side (`stand == p_throws`) | 0/1 | `features.add_outcomes_and_controls` |
| `fixed effects` | A separate intercept per group (here pitcher x side x count). Estimated by subtracting each group's mean from every variable before the regression | - | `validate_choice.within_ols` |
| `adherence` | Mean over a PA's scored pitches of (matched minus the pitcher x side x count 2026 match rate) | share | `validate_atbat.scored_pitches`, `pa_table` |
| `K` | 1 if the PA ends in a strikeout | 0/1 | `validate_atbat.pa_results` |
| `free` | 1 if the PA ends in a walk or hit by pitch | 0/1 | `validate_atbat.pa_results` |
| `hard` | 1 if a ball in play left the bat at >= 95 mph | 0/1 | `validate_atbat.pa_results` |
| `cxwoba` | `xwoba` on PAs that end in a ball in play (contact xwOBA) | wOBA scale | `validate_atbat.pa_results` |
| `n_csw` | CSW pitches in the PA | count | `validate_atbat.pa_table` |
| `tto` | `n_priorpa_thisgame` capped at 3, as a category | category | `validate_atbat.design` |
| `PAs` | Plate appearances in the fit | count | `validate_atbat.Fit` |
| `per +25 pts adherence` | $`0.25\,\beta`$ from the fixed-effects regression of the outcome on adherence | pts or wOBA | `validate_atbat.Fit` |
| `Q1 (least) ... Q4 (most)` | Adherence quartiles; Q2-Q4 give the adjusted difference from Q1 | category | `validate_atbat.main` |
