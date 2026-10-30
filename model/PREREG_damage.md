# Contact quality in the sequencer: pre-registration

Written and committed before any model in this step is trained or any 2026
outcome is compared with its picks. Run once. Train on 2025, test on 2026.

## Why

The sequencer ranks pitches by CSW (called strikes + whiffs). A pitch can
earn CSW and still get hit hard when it is put in play. This step ranks
pitches by expected run value instead, which counts balls, strikes, fouls
and how hard the ball is hit.

## Expected run value per pitch

Outcome classes: take -> called strike or ball (ball, blocked ball, hit by
pitch, automatic ball, pitchout); swing -> whiff (`features.WHIFF`), foul
(foul, foul bunt, bunt foul tip) or ball in play.

From 2025 (Statcast `delta_run_exp`, runs from the batting team's view):
- RV(count, outcome) = mean delta_run_exp for each count and non-contact
  outcome (ball, called strike, whiff, foul). This covers walks, strikeouts
  and two-strike fouls.
- Ball in play: a linear fit of delta_run_exp on count dummies + xwOBA
  (`estimated_woba_using_speedangle`) over 2025 balls in play.

Models (2025, the batter-specific model H features, fixed PARAMS, 400
rounds; timing from bat tracking as decided in PREREG_hitter.md):
P(swing), P(whiff | swing), P(foul | contact), P(called strike | take), and
E[xwOBA | ball in play] (regression).

E[RV] = P(take) [P(cs) RV(cs) + (1 - P(cs)) RV(ball)]
      + P(swing) [P(whiff) RV(whiff) + (1 - P(whiff)) (P(foul) RV(foul)
      + (1 - P(foul)) RV_bip(E[xwOBA]))]

The page shows runs saved per 100 pitches = -100 x E[RV].

## Picks

Same construction as the batter-specific sequencer (PREREG_hitter.md Part
2): side-only grid for an average hitter plus a per-hitter adjustment by
count and pitch type, all from the 2025 model. To keep run time in bounds,
the hitter adjustment uses 10 candidate pitches per type (the grid keeps
20).
- CSW pick: highest batter-specific CSW (as now).
- RV pick: lowest batter-specific E[RV].

Test pitches: as in PREREG_hitter.md Part 2 (2026, covered by the grid,
hitter with >= 200 pitches in 2025).

## Realized value of a test pitch

Contact quality is measured with xwOBA, not with where the ball landed:
- non-contact outcomes: RV(count, outcome) from the 2025 table
- ball in play: the 2025 linear fit at the pitch's actual xwOBA (missing
  xwOBA: the 2025 mean ball-in-play value for that count)

## Primary

Pitches where the RV pick and the CSW pick differ and the thrown type is one
of the two. Within-count mean realized value (weighted by pitches per
count) when the thrown type is the RV pick, minus when it is the CSW pick.
95% CI from a pitcher bootstrap (1,000 draws). **Pass** if < 0 (fewer runs)
with the CI excluding 0.

## Secondary (reported, not pass/fail)

- The same comparison with actual delta_run_exp.
- The same comparison for CSW (what the RV pick gives up) and for xwOBA on
  balls in play.
- Share of test pitches where the two picks differ.
- 2026 fit of E[xwOBA | ball in play]: R^2, against a model without the
  perceptual and hitter features (CONTROLS + SPEED only).

## Variables

Added on 2026-10-30 to define the terms used above. It is documentation only: the plan, data, tests and pass bars above are unchanged. The step-by-step math is in [damage_report.md](damage_report.md#how-the-model-works).

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
