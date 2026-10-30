# Sequencer validation on real 2026 pitches (pre-registered, single run)

> How every number below is computed is in [How the model works](#how-the-model-works), and every variable is defined in [Variables](#variables).

Sequencer trained on 2025 only. Test pitches: 471,215 from 265 pitchers. Share that matched the top pick: 28.6%.

## 1. Top pick vs not (primary)

Within-count CSW gap, matched minus unmatched: **+4.48 pts** (95% CI +3.99 to +4.98). Pass: **True**.

| count   |   pitches |   gap pts | top-pick share   |
|:--------|----------:|----------:|:-----------------|
| 0-0     |    123056 |      3.49 | 28%              |
| 0-1     |     60926 |      5.1  | 28%              |
| 0-2     |     31658 |      3.06 | 35%              |
| 1-0     |     45600 |      5.03 | 26%              |
| 1-1     |     47100 |      5.47 | 27%              |
| 1-2     |     45520 |      3.1  | 33%              |
| 2-0     |     15328 |      6.42 | 22%              |
| 2-1     |     24318 |      7.01 | 25%              |
| 2-2     |     38860 |      4.22 | 31%              |
| 3-0     |      4813 |      2.23 | 39%              |
| 3-1     |     10206 |     10.45 | 14%              |
| 3-2     |     23830 |      3.97 | 30%              |

## 2. Sequencer vs naive pick

On 418,453 pitches with a naive pick: sequencer gap +4.51 pts, naive gap +2.67 pts. Difference **+1.84 pts** (95% CI +1.24 to +2.39). Pass: **True**. The two picks agree in 36% of cells.

## 3. After-what effect

3,224 pitcher x type x side x count groups (124,022 pitches). Within group, CSW after a favored previous pitch minus after a less favored one: **+2.57 pts** (95% CI +2.04 to +3.11). Pass: **True**. For scale, the model predicted a gap of +1.39 pts between these halves.

## How the model works

This is the pre-registered check (`PREREG_validation.md`) of whether the sequencer's top picks did better
than other pitches in real 2026 at-bats.

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

**Test pitches.** 2026 regular-season pitches whose pitcher, batter side, count and previous type (none at
0-0) form a cell of the grid and whose type is in that cell (471,215 pitches, 265 pitchers). Each gets
$`m = 1`$ if its type is the cell's top pick.

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

**Test 1. Top pick vs not.** $`\Delta`$ with $`y`$ = CSW and $`m`$ = matched the top pick. Pass if
$`\Delta > 0`$ and the CI excludes 0. The table gives $`n_c`$, $`\Delta_c`$ and the top-pick share per count.

**Test 2. Against a naive pick.** The naive pick for a pitcher x side x count is the type with his highest
observed 2025 CSW there (types with >= 30 pitches in that cell), or else his highest 2025 CSW type to that
side (>= 30 pitches). On pitches where both picks exist, the statistic is
$`\Delta_{\text{sequencer}} - \Delta_{\text{naive}}`$, with the bootstrap applied to the difference.

**Test 3. After what.** For a test pitch of type $`k`$ in count $`c`$ after previous type $`q`$, with
$`\widehat{\text{CSW}}(c, q, k)`$ the grid value,

```math
\text{seq\_delta} = \widehat{\text{CSW}}(c, q, k) - \frac{1}{|Q|} \sum_{q' \in Q} \widehat{\text{CSW}}(c, q', k)
```

over the previous types $`Q`$ in that count. Within each pitcher x type x side x count group, pitches with
seq_delta above the group median are compared with the rest (groups need >= 10 pitches on each side). The
group gaps are averaged with weights equal to group size. The "model predicted" figure is the same
contrast computed on seq_delta itself.

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
