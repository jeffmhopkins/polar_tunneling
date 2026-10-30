# Tool vs pitcher choice (pre-registered, single run)

> How every number below is computed is in [How the model works](#how-the-model-works), and every variable is defined in [Variables](#variables).

471,215 2026 test pitches, 265 pitchers. Raw within-count gap: +4.48 pts.

## A. Adjusted gap (primary)

With pitcher x side x count fixed effects only: +4.25 pts.

Adding batter, situation and previous-pitch controls: **+3.56 pts** (95% CI +3.13 to +4.00), **80%** of the raw gap. Survives: **True**.

## B. Propensity check

Note: the propensity model uses the batter's 2025 rate against the class of the *top pick* in place of the class of the pitch thrown (which is what is being chosen, so it cannot predict the choice).

Propensity of throwing the top pick: median 0.27, IQR 0.18-0.37.

Inverse-propensity-weighted gap: **+4.16 pts** (95% CI +3.67 to +4.69), 93% of the raw gap.

| propensity tercile   |   pitches | matched share   |   within-count gap pts |
|:---------------------|----------:|:----------------|-----------------------:|
| low (broke pattern)  |    157072 | 13%             |                   5.33 |
| middle               |    157071 | 27%             |                   3.97 |
| high (usual choice)  |    157072 | 46%             |                   3.76 |

## How the model works

This pre-registered check (`PREREG_choice.md`) asks how much of the +4.48-pt top-pick gap in
`validation_report.md` is left after accounting for how pitchers choose pitches.

**Pitch features.** Every pitch is rebuilt from Statcast's 9-parameter fit, viewed from the batter's eye
and described by the controls, speed controls and perceptual features of `perceptual_report.md` and
`speed_check_report.md` (each of those reports walks through the formulas). In short: the trajectory is
$`\mathbf p(t) = \mathbf r_0 + \mathbf v_r t + \tfrac12\mathbf a t^2`$ from release to the front of the plate
at $`t = T`$, the hitter's eye is at $`\mathbf E`$, and the perceptual features describe the ball at the
commit time $`t_c = \max(T - \tau_b, 0.05)`$.

**Sequencer model** (`sequencing.py --train-years 2025`, written to `sequencing_2025model.json`). Three
LightGBM classifiers (swing, whiff \| swing, CSW) on CONTROLS + SPEED + all perceptual features, trained on
2025 only. From 2026 the grid uses only pitch shapes, locations and usage, never results. The previous-pitch
result lookup and the average hitter's rates also come from 2025.

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

**Test pitches.** The same 471,215 2026 pitches as the validation. For pitch $`i`$, $`d_i = 1`$ if it
matched the top pick and $`y_i`$ = CSW.

**Covariates (no 2026 outcomes)** (`validate_choice.covariates`).

- Batter rates from 2025, shrunk to the league rate with $`k = 200`$ prior pitches:
  $`\text{bat\_all}_b = (C_b + 200\,\ell)/(N_b + 200)`$ with $`C_b`$ his CSW-against pitches out of $`N_b`$;
  `bat_cls` the same against the class of the pitch thrown (fb, brk, off) with the league class rate. No
  2025 data: league rate and `bat_new` = 1.
- Situation: outs, runners on 1st, 2nd, 3rd (0/1 each), inning group (1-3, 4-6, 7+), platoon.
- Previous pitch: its class and result (whiff, called, foul, ball, other, none).

Categorical covariates enter as dummy variables.

**A. Adjusted gap (linear probability model with fixed effects).** With a separate intercept
$`\alpha_g`$ for each pitcher x side x count group $`g`$,

```math
y_i = \alpha_{g(i)} + \beta\, d_i + \boldsymbol\gamma^\top \mathbf z_i + \varepsilon_i
```

estimated by subtracting each group's mean from $`y`$, $`d`$ and $`\mathbf z`$ and running least squares on the
demeaned data (the within estimator). The reported gap is $`100\,\beta`$ (pts). The "fixed effects only"
line drops $`\mathbf z`$.

**B. Propensity check.** A LightGBM classifier (same settings, 300 trees) predicts $`d_i`$ from the count,
previous type and result, outs, inning group, runners, platoon, `bat_all`, `bat_new`, pitcher, side and
the batter's 2025 CSW-against rate on the class of the top pick. It is cross-fitted in 5 random folds, so
each pitch's propensity $`e_i`$ comes from a model that did not see it, then trimmed to [0.05, 0.95]. The
weighted gap is

```math
\text{IPW gap} = 100\left(\frac{\sum_i d_i y_i / e_i}{\sum_i d_i / e_i} - \frac{\sum_i (1 - d_i) y_i / (1 - e_i)}{\sum_i (1 - d_i)/(1 - e_i)}\right)
```

The tercile table splits pitches by $`e_i`$ and gives the within-count gap in each. The low tercile is
where the top pick was unlikely given the situation, so throwing it there broke the pitcher's pattern.

**Confidence intervals** (`validate_sequencer.boot`). Pitches from one pitcher are not independent, so
intervals come from a pitcher bootstrap: draw the test's $`P`$ pitchers with replacement, weight each
pitcher's pitches by the number of times he was drawn, recompute the statistic, repeat, and take the
2.5th and 97.5th percentiles. 500 draws; for B the propensities are held fixed across draws.

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
