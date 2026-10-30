# Whole at-bat test (pre-registered, single run)

> How every number below is computed is in [How the model works](#how-the-model-works), and every variable is defined in [Variables](#variables).

2026 plate appearances: 124,663 from 265 pitchers, sequencer trained on 2025 only. Mean share of scored pitches that matched the top pick: 27.6%. SD of adherence (match share minus the pitcher's usual rate in those counts): 0.254.

Effects are per +25 pts of adherence, with pitcher x side fixed effects, batter 2025 rates and situation controls. 95% CIs from a pitcher bootstrap (500 draws).

## Primary

1. **More strikeouts:** K rate +1.15 pts (95% CI +0.94 to +1.34). Pass: **True**.
2. **Fewer runs:** xwOBA per PA -0.0043 (95% CI -0.0063 to -0.0024). Pass: **True**.
3. **Weaker contact:** contact xwOBA +0.0003 (95% CI -0.0021 to +0.0026) on 84,677 balls in play. Pass: **False**. Not a pass, but the upper end is below +0.005, so the picks are not hit harder.

## All outcomes

| outcome                  |    PAs |   mean | per +25 pts adherence               |
|:-------------------------|-------:|-------:|:------------------------------------|
| K rate                   | 124663 |  0.226 | +1.15 pts (95% CI +0.94 to +1.34)   |
| xwOBA per PA             | 124349 |  0.312 | -0.0043 (95% CI -0.0063 to -0.0024) |
| contact xwOBA            |  84677 |  0.362 | +0.0003 (95% CI -0.0021 to +0.0026) |
| walk + HBP rate          | 124663 |  0.093 | -0.05 pts (95% CI -0.19 to +0.09)   |
| hard-hit rate on contact |  84677 |  0.389 | +0.15 pts (95% CI -0.19 to +0.47)   |
| actual wOBA              | 124663 |  0.317 | -0.0064 (95% CI -0.0091 to -0.0037) |

## Secondary: first pitch only

124,550 PAs with a scored 0-0 pitch (28% matched). Exposure: the 0-0 pitch matched the top pick (minus the pitcher x side 0-0 match rate). Effect of matching vs not.

| outcome                  |    PAs | 0-0 pitch was the top pick          |
|:-------------------------|-------:|:------------------------------------|
| K rate                   | 124550 | +0.38 pts (95% CI -0.17 to +0.85)   |
| xwOBA per PA             | 124236 | -0.0021 (95% CI -0.0070 to +0.0028) |
| contact xwOBA            |  84600 | +0.0008 (95% CI -0.0056 to +0.0076) |
| walk + HBP rate          | 124550 | -0.32 pts (95% CI -0.69 to +0.08)   |
| hard-hit rate on contact |  84600 | +0.58 pts (95% CI -0.20 to +1.46)   |
| actual wOBA              | 124550 | -0.0043 (95% CI -0.0106 to +0.0023) |

## Secondary: pitch-level contact check

83,080 balls in play on scored pitches (26% were the top pick), pitcher x side x count fixed effects. Top pick minus other pitches:

- contact xwOBA -0.0016 (95% CI -0.0086 to +0.0052)
- hard-hit rate -0.19 pts (95% CI -1.27 to +1.04)

## Secondary: implied vs observed K

Per +25 pts adherence, a PA gets +0.041 more CSW pitches; each extra CSW pitch goes with +29.50 pts of K rate. A CSW-only chain implies +1.21 pts (95% CI +0.97 to +1.43) of K rate, against the observed +1.15 pts (95% CI +0.94 to +1.34). (The per-pitch adjusted CSW gain from `choice_report.md` is +3.56 pts.)

## By adherence quartile

Adjusted difference from the least adherent quartile (same controls and fixed effects). Mean top-pick share by quartile: Q1 (least) 3%, Q2 10%, Q3 33%, Q4 (most) 64%.

| outcome       |     Q2 |     Q3 |   Q4 (most) |
|:--------------|-------:|-------:|------------:|
| K rate        |  6.05  | 13.53  |       5.89  |
| xwOBA per PA  | -0.003 | -0.034 |      -0.017 |
| contact xwOBA |  0.006 | -0.007 |       0     |

## Exploratory (added after the run, not pre-registered): same-length at-bats

Short PAs have extreme adherence (a one-pitch PA either matched or didn't) and cannot end in a strikeout, so the quartile table above mostly sorts PAs by length. Here quartiles are formed within PAs with the same number of scored pitches, and the fixed effects are pitcher x side x number of pitches (capped at 8). PA length is itself partly a result of the pitches, so this is a check on the shape of the effect, not a replacement for the primary test.

| outcome       |     Q2 |     Q3 |   Q4 (most) |
|:--------------|-------:|-------:|------------:|
| K rate        | -0.81  |  1.85  |       2.4   |
| xwOBA per PA  |  0.009 | -0.01  |      -0.007 |
| contact xwOBA |  0.006 | -0.005 |       0.003 |

Linear effect per +25 pts adherence within same-length PAs: K rate +0.86 pts (95% CI +0.68 to +1.04); xwOBA per PA -0.0031 (95% CI -0.0050 to -0.0012); contact xwOBA +0.0007 (95% CI -0.0017 to +0.0030).

## How the model works

This pre-registered test (`PREREG_atbat.md`) asks whether following the 2025-only sequencer across a
whole 2026 plate appearance goes with more strikeouts and fewer runs.

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

**Plate appearances.** 2026 regular-season PAs whose pitcher and batter side are in the grid and with at
least one scored pitch (a pitch is scored when its count x previous-type cell exists). The result is read
from the PA's last pitch. Sacrifice bunts, catcher interference, intentional walks and truncated PAs are
dropped.

**Adherence** (`scored_pitches`, `pa_table`). For scored pitch $`i`$, $`m_i = 1`$ if the type thrown is the
top pick. $`\bar m_{p,s,c}`$ is pitcher $`p`$'s 2026 match rate to side $`s`$ in count $`c`$ (which pitch was
thrown only, no results). For a PA with scored pitches $`I`$:

```math
\text{adh} = \frac{1}{|I|}\sum_{i \in I}\big(m_i - \bar m_{p,s,c(i)}\big)
```

Subtracting the base rate keeps a PA from earning adherence just by reaching counts where the top pick is
the usual pitch.

**Outcomes.** $`K`$ (strikeout), xwOBA of the PA (Statcast's expected wOBA: 0 for a strikeout, the wOBA
weight for a walk or HBP, from exit speed and angle for a ball in play), contact xwOBA (balls in play only),
walk + HBP, hard-hit (exit speed >= 95 mph, balls in play), and actual wOBA.

**Controls** (`design`). Batter 2025 rates shrunk to the league rate: $`K`$ and walk + HBP rate and xwOBA
per PA with 200 prior PAs, contact xwOBA with 100 prior balls in play, $`(\sum x + k\ell)/(n + k)`$; a flag
for batters without 2025 data; platoon; runners on 1st, 2nd, 3rd; and dummies for outs, inning group (1-3,
4-6, 7+) and times through the order (`n_priorpa_thisgame` capped at 3).

**Model.** For each outcome $`y`$, with a fixed effect for each pitcher x batter side $`g`$:

```math
y_j = \alpha_{g(j)} + \beta\, \text{adh}_j + \boldsymbol\gamma^\top \mathbf z_j + \varepsilon_j
```

estimated by within-group demeaning and least squares (a linear probability model for 0/1 outcomes). The
report gives $`0.25\,\beta`$: the change for +25 pts of adherence (x100 for rates, in pts).

**Pass rules.** K: $`\beta > 0`$ with the CI above 0. xwOBA and contact xwOBA: $`\beta < 0`$ with the CI
below 0.

**Confidence intervals** (`validate_sequencer.boot`). Pitches from one pitcher are not independent, so
intervals come from a pitcher bootstrap: draw the test's $`P`$ pitchers with replacement, weight each
pitcher's pitches by the number of times he was drawn, recompute the statistic, repeat, and take the
2.5th and 97.5th percentiles. 500 draws.

**Secondary.**

- First pitch only: adherence is replaced by $`m_i - \bar m_{p,s,\text{0-0}}`$ for the PA's 0-0 pitch, and
  the coefficient is reported unscaled (matched vs not).
- Pitch-level contact: on balls in play, contact xwOBA and hard-hit on $`m_i`$ with pitcher x side x count
  fixed effects and the same controls.
- Implied vs observed K: $`a`$ = the adherence coefficient on the PA's number of CSW pitches, $`b`$ = the
  coefficient of $`K`$ on the number of CSW pitches (same controls and fixed effects). $`0.25\,a\,b`$ is
  the K change a CSW-only chain implies, compared with $`0.25\,\beta_K`$; its interval comes from the
  bootstrap draws of $`a \cdot b`$.
- Quartiles: PAs are split into adherence quartiles, and the outcomes are regressed on quartile dummies
  (Q1 the reference) with the same controls and fixed effects.
- Exploratory (added after the run): the same within PAs of equal length, with quartiles formed among PAs
  with the same number of scored pitches and fixed effects for pitcher x side x PA length (capped at 8).

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
