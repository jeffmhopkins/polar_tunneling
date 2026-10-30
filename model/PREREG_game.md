# Model comparison and game report: pre-registration

Written and committed before any comparison model's picks are scored
against 2026 outcomes. The season comparison is run once.

## Models

All are trained on 2025 only and scored the same way: each candidate pitch
type is represented by the same real 2026 pitches of that pitcher (shapes and
locations, as in `sequencing.py`), placed in the count and after the
previous pitch type, for an average hitter. The top pick is the type with the
highest predicted CSW in the pitcher x side x count x previous-type cell.

- **A. Perceptual sequencer**: the validated sequencer model (controls +
  speed + perceptual features), i.e. `sequencing.py --train-years 2025`.
- **B. Standard pitch model**: the same LightGBM on the controls + speed
  only (stuff, location, count, previous pitch, fastball differentials, arm
  angle, batter rates). No perceptual features.
- **C. Stuff-only model** (Stuff+-style): pitch type, hands, velocity,
  movement, spin, extension, release point, arm angle and fastball
  differentials. No location, count or previous pitch, so its pick is the
  pitcher's best-stuff pitch to that side whatever the situation.
- **D. Best pitch for the count**: the naive pick from
  `PREREG_validation.md` (highest observed 2025 CSW to that side in that
  count, >= 30 pitches, else highest overall).
- **E. Pitcher's habit**: the type he threw most often to that side in that
  count in 2026 (descriptive; it uses 2026 usage, not outcomes).

## Season comparison (2026 regular season)

Test pitches as in `PREREG_validation.md`: the pitch's type, side, count and
previous type must be in that pitcher's grid. For each model, the
within-count CSW gap of pitches that matched its pick minus pitches that
didn't, with a 2,000-draw pitcher bootstrap.

Primary: **A's gap minus B's gap** on the same pitches. A beats the standard
model if > 0 with a 95% interval excluding 0. Also reported: A minus C,
A minus D, A minus E, and the share of pitches where each pair agrees.

Caveat stated in advance: this is observational, like the earlier
validation. Pitchers choose with knowledge we don't have.

## Game report (descriptive, not a test)

One 2026 postseason game (default: World Series Game 5, game 849799). The
models never saw postseason pitches. For every pitch: each model's pick and
predicted CSW for each candidate, the pitch actually thrown, and the result.
Game pitchers get the same grids with lower minimums (>= 300 regular-season
pitches, >= 60 to a side, >= 15 of a type). One game is far too small to rank
models; the season comparison is the evidence.

Amendment (before any run): Jeff asked for an Orioles game, so the default
game is now 823491, Orioles at Yankees on 2026-09-25 (their latest game in
the data). It is a regular-season game, so its own pitches are left out of
the 2026 shapes and usage used to build the grids. The model still never saw
any 2026 outcome.

## Variables

Added on 2026-10-30 to define the terms used above. It is documentation only: the plan, data, tests and pass bars above are unchanged. The step-by-step math is in [game_report.md](game_report.md#how-the-model-works).

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
