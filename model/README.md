<div align="center">

# model/

Every tunneling test, in the order it was run: the features, the pre-registration, the script and its report.

[![Train: 2025](https://img.shields.io/badge/train-2025-blue)](../datamining/outputs) [![Test: 2026, once](https://img.shields.io/badge/test-2026%2C%20once-blue)](../datamining/outputs) [![Model: LightGBM](https://img.shields.io/badge/model-LightGBM-blue)](requirements.txt)

</div>

> [!IMPORTANT]
> **How each test runs.** The question, data, controls and pass/fail bar go in a `PREREG_*.md` file, which
> is committed before any outcome is computed. The script then trains on 2025, scores 2026 once and writes
> its `*_report.md`. Every result is kept, including the failures. Anything not in the pre-registration is
> labelled exploratory.

| Study | Pre-registration | Script | Report | Verdict |
|---|---|---|---|---|
| [Sequence tunnel score](#sequence-tunnel-score) | none (go/no-go) | `feasibility.py`, `robustness.py` | [feasibility](feasibility_report.md), [robustness](robustness_report.md) | Fail |
| [Arsenal vs fastball](#arsenal-check-2025-vs-primary-fastball) | none | `arsenal.py`, `arsenal_holdout.py` | [2025](arsenal_report.md), [2026](arsenal_report_2026.md), [holdout](arsenal_holdout_report.md) | Fail |
| [Perceptual model](#perceptual-model-pre-registered-preregmd) | [PREREG.md](PREREG.md) | `perceptual.py` | [report](perceptual_report.md) | Pass (pitch level) |
| [Speed check](#speed-check-pre-registered-prereg_speedmd) | [PREREG_speed.md](PREREG_speed.md) | `speed_check.py` | [report](speed_check_report.md) | Pass |
| [Sequencer validation](#sequencer-validation-pre-registered-prereg_validationmd) | [PREREG_validation.md](PREREG_validation.md) | `validate_sequencer.py` | [report](validation_report.md) | Pass |
| [Tool vs pitcher choice](#tool-vs-pitcher-choice-pre-registered-prereg_choicemd) | [PREREG_choice.md](PREREG_choice.md) | `validate_choice.py` | [report](choice_report.md) | +3.6 pts survives |
| [Per-hitter timing](#per-hitter-timing-pre-registered-prereg_hittermd-part-1) | [PREREG_hitter.md](PREREG_hitter.md) | `hitter_timing.py` | [report](hitter_timing_report.md) | Fail |
| [Batter-specific sequencer](#batter-specific-sequencer-pre-registered-prereg_hittermd-part-2) | [PREREG_hitter.md](PREREG_hitter.md) | `hitter_sequencer.py` | [report](hitter_sequencer_report.md) | Pass |
| [Whole at-bat](#whole-at-bat-test-pre-registered-prereg_atbatmd) | [PREREG_atbat.md](PREREG_atbat.md) | `validate_atbat.py` | [report](atbat_report.md) | Pass (K, runs); fail (contact) |
| [Run-value sequencer](#contact-quality-run-value-sequencer-pre-registered-prereg_damagemd) | [PREREG_damage.md](PREREG_damage.md) | `damage_sequencer.py` | [report](damage_report.md) | Pass |
| [Model comparison, game report](#model-comparison-and-game-report-pre-registered-prereg_gamemd) | [PREREG_game.md](PREREG_game.md) | `game_report.py` | [report](game_report.md) | Fail (perceptual vs standard) |
| [Comparison by runs, public models](#model-comparison-by-runs-and-public-models-pre-registered-prereg_runsmd) | [PREREG_runs.md](PREREG_runs.md) | `runs_compare.py` | [report](runs_report.md) | Fail (perceptual vs standard); beats Pitching+, PitchingBot |
| [Location](#location-in-the-sequencer-pre-registered-prereg_locationmd) | [PREREG_location.md](PREREG_location.md) | `location_compare.py` | [report](location_report.md) | Fail |
| [Group-reference read](#group-reference-read-pre-registered-prereg_groupmd) | [PREREG_group.md](PREREG_group.md) | `group_read.py` | [report](group_report.md) | Pass (whiff, small); fail (swing) |

`pregame_report.py` builds the pre-game plan from the standard run-value model, and `sequencing.py`
builds the original CSW sequencer data (`sequencing.json`; `sequencing_2025model.json` is the 2025-only
version the validations use).

## Notation and shared math

Every study builds on the same steps. Each report ends with **How the model works**, which walks through its
own math from these steps to the numbers in its tables, and **Variables**, which defines every variable it
uses (meaning, units, source column). The short version:

**Coordinates.** Statcast's catcher's-view frame, in feet: $`x`$ across the plate (+ toward the
first-base side), $`y`$ from the back point of home plate toward the mound, $`z`$ up. The front edge of
the plate is at $`y_p = 17/12`$ ft.

**Rebuild the trajectory** (`polar_transformations/reconstruct_trajectory.py`, `features.add_state`).
Statcast fits every pitch with a constant acceleration $`\mathbf a = (a_x, a_y, a_z)`$ and publishes the
fit's velocity $`\mathbf v_{50} = (v_{x0}, v_{y0}, v_{z0})`$ at $`y = 50`$ ft, the release depth $`y_r`$ and
the plate crossing $`(x_p, z_p)`$ at $`y_p`$. The time, counted from $`y = 50`$, at which the fit reaches
depth $`y`$ is the root of $`50 + v_{y0} t + \tfrac12 a_y t^2 = y`$ that the ball reaches first:

```math
t(y) = \frac{-v_{y0} - \sqrt{v_{y0}^2 - 2 a_y (50 - y)}}{a_y}
```

With $`t_p = t(y_p)`$ (positive) and $`t_r = t(y_r)`$ (negative, release is before $`y = 50`$), the fit's
$`x`$ and $`z`$ at $`y = 50`$ follow from the plate crossing, and the state at release from running the fit
back to $`t_r`$:

```math
x_{50} = x_p - v_{x0} t_p - \tfrac12 a_x t_p^2, \qquad z_{50} = z_p - v_{z0} t_p - \tfrac12 a_z t_p^2
```

```math
\mathbf r_0 = \big(x_{50} + v_{x0} t_r + \tfrac12 a_x t_r^2,\ \ y_r,\ \ z_{50} + v_{z0} t_r + \tfrac12 a_z t_r^2\big), \qquad
\mathbf v_r = \mathbf v_{50} + \mathbf a\, t_r, \qquad T = t_p - t_r
```

$`T`$ is the flight time from release to the front of the plate. From release, the ball's position and
velocity at time $`t \in [0, T]`$ are

```math
\mathbf p(t) = \mathbf r_0 + \mathbf v_r t + \tfrac12 \mathbf a t^2, \qquad \mathbf v(t) = \mathbf v_r + \mathbf a t
```

Pitches with $`T \ge 0.6`$ s (lobs and bad fits) are dropped, as are pitch types PO, EP, FA, CS, SC and UN
and any row missing a fit column.

**View it from the batter's eye** (`features.EYE_*`). The eye sits 35.8 in off the plate's center line
on the batter's side, 28.3 in behind the back point of the plate and 65 in high:

```math
\mathbf E = \tfrac{1}{12}\,\big(-35.8\, s_b,\ -28.3,\ 65.0\big)\ \text{ft}, \qquad s_b = +1 \text{ (right-handed batter)},\ -1 \text{ (left-handed)}
```

The angle between two ball positions $`\mathbf a`$ and $`\mathbf b`$ as the hitter sees them is

```math
\theta(\mathbf a, \mathbf b) = \frac{180}{\pi} \arccos \frac{(\mathbf a - \mathbf E)\cdot(\mathbf b - \mathbf E)}{\lVert \mathbf a - \mathbf E\rVert\, \lVert \mathbf b - \mathbf E\rVert}\ \ \text{deg}
```

**Commit point.** The hitter has to decide $`\tau`$ seconds before the ball reaches the front of the plate
($`\tau = 0.175`$ s, `features.TAU`). For a pitch with flight time $`T`$ that moment is
$`t_c = \max(T - \tau, 0)`$ after release. Two pitches are compared at the same time since their own
release, so the hitter's clock, not the pitch's share of its flight, sets the comparison. The tracking
limit is 0.100 s before arrival (`features.TRACK`).

**Outcomes** (`features.add_outcomes_and_controls`):

```math
\text{swing} = \mathbb 1[\text{description} \in \text{SWING}], \quad
\text{whiff} = \mathbb 1[\text{description} \in \text{WHIFF}], \quad
\text{csw} = \text{whiff} \lor \mathbb 1[\text{description} = \text{called\_strike}]
```

Whiff is modelled on swings only, so its model estimates $`P(\text{whiff} \mid \text{swing})`$; swing and
CSW are modelled on all pitches.

**Controls (BASE)** (`feasibility.BASE`, `features.add_outcomes_and_controls`): `pitch_type`, `stand`,
`p_throws`, `release_speed_mph`, `mov_x` $`= s_a \cdot`$ `pfx_x_ft`, `pfx_z_ft`, `release_spin_rate_rpm`,
`release_extension_ft`, `release_pos_z_ft`, `rel_x` $`= s_a \cdot`$ `release_pos_x_ft`, `loc_x` $`= s_b x_p`$,
`loc_z` $`= (z_p - z_{bot})/(z_{top} - z_{bot})`$, `balls`, `strikes`, `platoon`. Here $`s_a = +1`$ for a
right-handed pitcher and $`-1`$ for a lefty, so both hands share one sign convention.

**Previous-pitch controls (PREV)**: `prev_pitch_type`, `prev_desc` (whiff, called, foul, ball, other),
`prev_loc_x`, `prev_loc_z`.

**Model** (`feasibility.PARAMS`). Each outcome $`y \in \{0, 1\}`$ gets a LightGBM classifier: an additive
sum of $`K = 400`$ regression trees on the features $`\mathbf x`$, passed through the logistic function,

```math
\hat p(\mathbf x) = \sigma\Big(\sum_{k=1}^{K} f_k(\mathbf x)\Big), \qquad \sigma(u) = \frac{1}{1 + e^{-u}}
```

where each tree $`f_k`$ is fitted to the gradient of the log-loss of the trees before it (learning rate
0.05, at most 63 leaves, at least 200 training rows per leaf, a random 80% of features and of rows per
tree, seed 7). Categorical columns (`pitch_type`, `stand`, `p_throws`, `prev_pitch_type`, `prev_desc`)
are split on directly. Missing values are allowed: a tree learns which side of each split they go to.

**Score: held-out log-loss gain.** On the $`n`$ test pitches,

```math
L = -\frac{1}{n} \sum_{i=1}^{n} \big[ y_i \log \hat p_i + (1 - y_i) \log (1 - \hat p_i) \big],
\qquad \text{gain} \times 1000 = 1000\,(L_{\text{without}} - L_{\text{with}})
```

A positive gain means the added features improved the forecast of pitches the model never saw. For scale,
adding the previous pitch's type, location and result to the controls gained 1.4 (whiff per swing) and
2.1 (swing) per 1000 in 2025.

**Permutation null.** To check that a gain is more than the extra flexibility of more columns, the
tested features are shuffled among pitches of the same pitcher x pitch type, and the model is refit and
scored the same way. Shuffling keeps each pitcher's pitch types and the features' spread, and breaks only
the link between a feature value and the pitch it belongs to. A real gain has to beat every null draw.

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
2.5th and 97.5th percentiles.

The perceptual features (P1-P6), the sequencer grid and the expected run value each get their full
derivation in [perceptual_report.md](perceptual_report.md#how-the-model-works),
[validation_report.md](validation_report.md#how-the-model-works) and
[damage_report.md](damage_report.md#how-the-model-works).

## Sequence tunnel score

The first go/no-go checks. Each pitch is compared with the previous pitch in the same plate appearance.

- `features.py` rebuilds each pitch from Statcast's 9-parameter fit, views it
  from the batter's eye, and computes the separations against the previous pitch:
  angular at the commit point (`d_dec_deg`), and physical at the tracking limit
  (`d_track_in`) and at the plate (`d_plate_in`).
- `feasibility.py` runs the checks and writes `feasibility_report.md`.
- `robustness.py` repeats the lift test at 125/175/225 ms, on different-type
  pairs only, with more null draws, and writes `robustness_report.md`.

### Findings (2025)

1. **Free variation is limited.** Pitch types plus the two plate locations
   predict 71% of the variance in commit-point separation (held-out R²). Most of
   what the tunnel feature measures is already pinned down by pitch types and
   where the two pitches ended up.
2. **The raw relationship runs the wrong way for tunneling.** Whiff per swing
   *rises* with early separation, from 21% in the closest decile to 28% in the
   farthest. Early-close + late-far pairs (the tunneling ideal) whiff 24.6%,
   while early-far + late-far pairs whiff 29.7%. In the raw data, separation
   mostly tracks pitch mix and chase location.
3. **No incremental predictive value.** Added to a model that already knows
   stuff, location, count and the previous pitch's type, location and result,
   the tunnel features change held-out log-loss by -0.6 to +0.5 per 1000. That
   holds at every commit point tested and for whiff/swing, swing and CSW. Real
   features beat shuffled ones for swing decisions, so a little information is
   there, but the absolute gain is close to zero.
4. **Pitcher-level values are stable but don't predict results.** A pitcher x
   pitch type's mean tunnel value is very reliable (split-half 0.93), because it
   is a function of his arsenal. But it does not predict his residual whiffs in
   the other half of games (r = 0.015).

> [!WARNING]
> The sequence tunnel score fails the go/no-go as designed.

## Arsenal check (2025, vs primary fastball)

`arsenal.py` tests the explorer's comparison across pitchers. Each pitcher x
pitch type x side's average path is compared with his primary fastball, using
features from Mar-Jul, and the test runs on Aug-Oct with fastball differentials
(speed, movement, release) in the controls. It writes `arsenal_report.md`.

- **Whiff/swing:** +0.41 per 1000 log-loss, beating all 10 null draws (null
  max 0.32). That is small but above noise.
- **Swing and CSW:** no gain.
- **Group level:** the direction matches tunneling. Pitch types closest to the
  fastball at the commit point whiff 1.5 pct points more than predicted (top
  quartile), against about 0 for the farthest quartile. But the rank
  correlation is only -0.08 over 364 groups (p of about 0.12).

This is a weak, suggestive signal for whiffs. It needs a true out-of-season
test on 2026 before it can carry a score.

## 2026 holdout

- **Within 2026** (`arsenal.py --year 2026`): the whiff signal does not
  replicate. The gain is -0.04 per 1000, the rank correlation is 0.01, and
  the quartiles are not ordered.
- **2025 -> 2026** (`arsenal_holdout.py`): the 2025 group features are stable
  year to year (commit-point angle r = 0.85). In 2026, the closest quartile
  whiffs 1.2 pct points more than the farthest, relative to the model (95%
  bootstrap 0.2 to 2.2), and the rank correlation is -0.07 over 1,065 groups.
  But closeness *at the plate* predicts more whiffs just as well (-0.11), and
  that is not the tunneling pattern. The absolute log-loss gain is 0.1 per
  1000. Note that the shuffled nulls here are not a fair bar: a model trained
  on 2025 partly uses the group features as identifiers, so shuffling them
  hurts more than having no features at all.

Conclusion: no tunneling variant tested here (sequence pairs, arsenal vs
fastball, within or across seasons) adds enough to justify an outcome-based
Tunnel+. The arsenal angle gaps are stable descriptors of arsenal shape, so
they are still useful as a description in the explorer.

## Perceptual model (pre-registered, `PREREG.md`)

`perceptual.py` trains on 2025 and scores 2026 once. The pre-registration was
committed before any outcome was computed, and the result is in
`perceptual_report.md`.

- **Primary: pass.** Over the controls (stuff, location, count, previous
  pitch, fastball differentials, arm angle, batter rates), the perceptual
  features improve 2026 log-loss by:
  - 1.83 per 1000 for whiff | swing (nulls -0.07 to 0.08)
  - 5.86 per 1000 for swing (nulls -0.47 to -0.07)

  The largest blocks are optical motion (mainly for swing decisions), at-bat
  build-up and fastball-prior miss.
- **Secondary: fail.** A pitcher x pitch type's mean perceptual value is stable
  (split-half 0.94) but does not predict his residual whiffs in the other half
  of games (r = -0.02).

Reading: the gain comes from pitch-to-pitch variation within a pitcher. The
null keeps each pitcher x type's feature distribution and still scores about
0, so the signal is in which pitch, when and after what. That makes it useful
for pitch-level and sequencing models. It does not support ranking pitchers
by a stable Tunnel+ skill. Part of the optical-motion gain may be ball speed
and trajectory information that the plate-based controls don't fully capture.
That is an untested caveat.

## Speed check (pre-registered, `PREREG_speed.md`)

`speed_check.py` adds plate speed, flight time, and speed and flight-time
differentials vs the previous pitch and the fastball to the controls. It runs
2025 -> 2026 once and writes `speed_check_report.md`.

- **Primary: pass.** The perceptual gain survives almost entirely:
  - whiff | swing: 2.14 per 1000 (nulls up to 0.54)
  - swing: 5.67 per 1000 (nulls all negative)

  That keeps 117% and 97% of the original gain, so speed is not the
  explanation.
- **Exploratory:** adding the ball's raw position and velocity at the commit
  point (6 numbers) to the controls gives about the same gain as all the
  perceptual features. On top of those kinematics:
  - optical motion adds about 0 for whiffs and 0.7 for swings
  - the other perceptual blocks together still add 0.9 (whiff) and 2.0 (swing)

Reading: the main signal is where the ball is, and how it is moving, at the
moment the hitter must decide. The plate-based controls miss this. It is not
speed. Expressing it as eye-frame angular motion adds little over raw
kinematics, but the expectation, ambiguity, fastball-prior miss and at-bat
build-up features add something beyond it (exploratory, not pre-registered).

## Sequencer validation (pre-registered, `PREREG_validation.md`)

`validate_sequencer.py` scores real 2026 pitches against a sequencer built
from a 2025-only model (`sequencing.py --train-years 2025`). It writes
`validation_report.md`. All three tests pass:

1. **Top pick vs not.** Pitches that matched the top pick had a CSW 4.5 pts
   higher than other pitches in the same count (95% CI +4.0 to +5.0). The gap
   is positive in every count.
2. **Beats a naive pick.** The naive pick is the pitcher's best 2025 CSW type
   in that count. The sequencer's gap beats the naive pick's by 1.8 pts
   (CI +1.2 to +2.4).
3. **After what.** With the same pitcher, pitch type, side and count, the pitch
   did 2.6 pts better on CSW after a previous pitch the tool favors (CI +2.0 to
   +3.1). The model predicted 1.4 pts.

Caveat: this is observational. Pitchers choose pitches with knowledge of the
batter and the situation, which these comparisons don't control. The
after-what gap being larger than predicted suggests some of that selection is
included.

## Tool vs pitcher choice (pre-registered, `PREREG_choice.md`)

`validate_choice.py` writes `choice_report.md`. It controls the 4.48-pt
top-pick gap for:

- pitcher x side x count
- the batter's 2025 CSW-against rates (overall and against the pitch's class)
- outs, runners, inning and platoon
- the previous pitch's class and result

Results:

- **Adjusted gap: +3.56 pts** (95% CI +3.13 to +4.00), 80% of the raw gap.
- **Inverse-propensity-weighted gap: +4.16 pts** (CI +3.67 to +4.69).
- When the top pick was unlikely given the situation (lowest propensity
  tercile), throwing it anyway still gained 5.3 pts. In the usual-choice
  tercile the gain was 3.8 pts.

So most of the gain is not explained by pitchers already choosing well for
the batter and the situation. What remains is an upper bound: catcher,
scouting reports and game plans are not measured. Part of it is also that
the top pick tends to be the pitcher's best-CSW type, which the naive pick
captures (validation test 2: the tool beats that by 1.8 pts).

## Per-hitter timing (pre-registered, `PREREG_hitter.md` Part 1)

`hitter_timing.py` writes `hitter_timing_report.md` and
`batter_tau_2025.csv`. Each hitter's commit point was fitted on 2025 from how
well the model predicted his swings and whiffs at 125-225 ms before arrival.

- **Did not pass.** Using the fitted timing made 2026 predictions worse than
  the current bat-tracking timing (whiff per swing -0.64, swing -0.87 x1000),
  about as bad as giving hitters someone else's fitted timing.
- The fitted timing doesn't repeat: odd vs even 2025 games correlate at
  r = 0.04. The spread between hitters is mostly noise.
- League-wide, 150 ms fit 2025 slightly better than 175 ms, but that did not
  carry over to 2026 in a useful way (whiff +0.28, swing -0.34).

So the sequencer keeps the bat-tracking timing.

## Batter-specific sequencer (pre-registered, `PREREG_hitter.md` Part 2)

`hitter_sequencer.py` writes `hitter_sequencer_report.md`. The model adds
each hitter's whiff, swing and chase rates against fastballs, breaking balls
and offspeed. The page's score is the average-hitter grid plus a per-hitter
adjustment by count and pitch type. Trained on 2025, tested once on 2026
(hitters with >= 200 pitches seen in 2025, rates from 2025 only):

- The hitter-specific pick differs from the average-hitter pick on 17.6% of
  pitches.
- **Primary: passes.** Where they differ, throwing the hitter-specific pick
  got **+2.90 CSW pts** more than throwing the average-hitter pick (95% CI
  +2.06 to +3.77, 37,667 pitches).
- Over all pitches, the matched gap rises from +4.02 to +4.71 pts (+0.69,
  CI +0.48 to +0.90).
- The class features improve 2026 log-loss (swing +0.98, whiff +0.58,
  CSW +0.78 x1000).

Pitchers scout hitters too, so this is an upper bound on what the tool adds
beyond that.

`hitter_sequencer.py --production` retrains on both seasons and writes
`sequencing.json` and `batter_adj/` (hitters with >= 2,500 pitches seen),
which `report_generation/build_sequencer.py` turns into the page.

## Whole at-bat test (pre-registered, `PREREG_atbat.md`)

`validate_atbat.py` writes `atbat_report.md`. It asks whether following the
2025-only sequencer across a 2026 at-bat leads to more strikeouts and weaker
contact. Adherence = the share of a PA's pitches that matched the top pick,
minus that pitcher's usual rate in the same counts. Controls are the
pitcher-choice ones (pitcher x side fixed effects, the batter's 2025 rates,
outs, runners, inning, platoon, times through the order). 124,663 PAs, 265
pitchers. Effects are per +25 pts of adherence:

- **More strikeouts: pass.** K rate +1.15 pts (95% CI +0.94 to +1.34), on a
  22.6% base.
- **Fewer runs: pass.** xwOBA per PA -0.0043 (CI -0.0063 to -0.0024).
  Actual wOBA -0.0064.
- **Weaker contact: fail.** Contact xwOBA +0.0003 (CI -0.0021 to +0.0026),
  hard-hit rate +0.15 pts. The picks are not hit harder, but not weaker either.
  At the pitch level, balls in play off top picks have the same xwOBA as
  other pitches (-0.002, CI -0.009 to +0.005).
- Walks don't change (-0.05 pts).
- The K gain is what the per-pitch CSW gain predicts (+1.21 pts implied),
  so CSW carries through to the at-bat result without washing out, and
  nothing extra appears beyond it.
- Following the tool on the first pitch alone gives a smaller, not
  significant effect (K +0.38 pts, CI -0.17 to +0.85). The gain builds over
  the at-bat.
- Exploratory (not pre-registered): within same-length at-bats the effect is
  a little smaller (K +0.86 pts, xwOBA -0.0031) and rises across quartiles.

So the tool's value over an at-bat comes through strikeouts. It has no
contact-quality signal, which is what a contact term in the model would add.

## Contact quality: run-value sequencer (pre-registered, `PREREG_damage.md`)

`damage_sequencer.py` writes `damage_report.md`. It ranks pitches by
expected run value instead of CSW. Six models (swing, whiff | swing,
foul | contact, called strike | take, CSW, xwOBA | ball in play) feed 2025
run values per count and outcome, and a fit of run value on xwOBA for balls
in play. Trained on 2025 with the batter-specific features, tested once on
2026:

- The run-value pick differs from the CSW pick on 45.9% of pitches.
- **Primary: passes.** Where they differ, the run-value pick allowed
  **-0.27 runs per 100 pitches** (95% CI -0.48 to -0.08), measured with
  xwOBA on contact. With actual run values: -0.38 (CI -0.67 to -0.09).
- It gives up 5.2 CSW points but allows 0.026 less xwOBA on balls in play
  (CI -0.038 to -0.014).
- The perceptual and hitter features add little to predicting xwOBA on
  contact (2026 R^2 0.032 vs 0.030), so most of the gain comes from weighing
  outcomes by their run value, not from predicting contact quality better.

`damage_sequencer.py --production` writes the page data (`sequencing.json`,
`batter_adj/`) with runs saved per 100 next to CSW, plus team tags for the
pre-game plan.

## Model comparison and game report (pre-registered, `PREREG_game.md`)

`game_report.py` compares five pitch pickers, all built from 2025 only, and
writes `game_report.md` plus `report_generation/game_<pk>.json` for one
game's page (`report_generation/build_game_report.py`). Default game:
Orioles 10 at Yankees 2, 2026-09-25.

- **Primary: did not pass.** The perceptual sequencer's picks gain +4.49 CSW
  pts over unmatched pitches, the standard model's (no perceptual features)
  +4.33. The difference is +0.16 (95% CI -0.06 to +0.38). The two agree on
  82% of pitches.
- Stuff-only (Stuff+-style) picks: +4.03; A minus C +0.46 (CI -0.12 to +1.04).
- Both clearly beat simple rules: best pitch for the count +2.67 (A minus D
  +1.85, CI +1.26 to +2.41) and the pitcher's habit +0.52 (A minus E +3.97).
- As forecasts of the pitch thrown, the perceptual model still has the best
  2026 log-loss (CSW 0.4727 vs 0.4748 standard, 0.5913 stuff-only).

Reading: most of the sequencer's value comes from modelling stuff, location
and count. The perceptual features improve predictions but rarely change
which pitch is best.

## Model comparison by runs, and public models (pre-registered, `PREREG_runs.md`)

`runs_compare.py` writes `runs_report.md` and the runs picks for the game
page. Pickers built from 2025 only: perceptual, standard and Stuff-only
models choosing by expected runs (RA, RB, RC), the perceptual CSW pick, the
two rules, and FanGraphs' published 2025 Stuff+, Pitching+ and PitchingBot
grades (best-graded type in his menu; `datamining/pull_fangraphs_grades.py`).

- **Primary: did not pass.** By runs, the perceptual pick (-0.60 runs per
  100 pitches vs not) and the standard model's pick (-0.58) are the same:
  RA minus RB -0.03 (95% CI -0.12 to +0.06). They agree on 78%.
- Against the public grades, our runs pick allows fewer runs than
  Pitching+ (-0.21, CI -0.38 to -0.04) and PitchingBot (-0.29, CI -0.43 to
  -0.12), and about the same as Stuff+ (-0.19, CI -0.35 to +0.00). Our CSW
  pick gets more CSW than all three (+1.3 to +2.9 pts, CIs above 0).
- The pitcher's habit is close on runs (RA minus habit -0.08, CI -0.23 to
  +0.07) while adding few strikes.

Reading: modelling the count and the pitch's real shape and location is what
beats the public grades, which ignore both. The perceptual features still
don't change the pick enough to matter, whether pitches are chosen by
strikes or by runs.

## Location in the sequencer (pre-registered, `PREREG_location.md`)

`location_compare.py` writes `location_report.md` and the location picks
for the game page. Each test pitch is scored in its real context (count, the
actual previous pitch and its location, the last fastball seen) against
every (type, Statcast zone) option in the pitcher's 2026 menu, using the
2025 run-value models. 71,670 test pitches (15% sample of 2026).

- **Primary: did not pass.** Perceptual (LA) minus standard (LB) type + zone
  picks: -0.07 runs per 100 pitches (95% CI -0.71 to +0.50); +1.90 CSW pts
  (CI -0.23 to +4.19). They agree on type and zone 60% of the time.
- A 2025 league location chart (LZ) is close too: LA minus LZ -0.58 runs per
  100 (CI -1.54 to +0.45).
- Pitches that land in the picked zone do far better than others (about -5
  runs per 100), but that is mostly where pitches end up, which a league chart
  captures as well, and it mixes in command.

Reading: choosing location in the real sequence context does not reveal a
tunneling effect on decisions either. Across type, runs and location, the
perceptual features improve forecasts but not picks.

## Group-reference read (pre-registered, `PREREG_group.md`)

`group_read.py` writes `group_report.md`. P2 measures the post-commit miss
from a fastball read. This test replaces the fastball with the group of the
pitcher's types that look like the pitch at commit: each type is weighted by
the P4 observer's posterior, each extrapolates from the commit point with its
own mean acceleration, and the read is the weighted mean crossing. Four
features: the miss in x, z and time, and the weighted spread of the group's
crossings.

- **Primary: passed.** Whiff | swing, M1 + G over M1: +0.52 per 1000 (nulls
  0.03 to 0.27).
- Head to head over M0: the group read +0.91, the fastball read (P2) +0.36.
- Swing: +0.18, under the 0.3 bar.

Reading: if tunneling is measured post-commit, the look-alike group is a
better reference than the fastball. It is still a small forecast gain, and it
has not been run through the sequencer, where earlier perceptual blocks did
not change picks.

## Variables

Every variable used above. The reports define the ones they use the same way.

| Variable | Meaning | Units | Source |
|---|---|---|---|
| `pitcher` | MLBAM id of the pitcher | id | Statcast column `pitcher` (`datamining/outputs`) |
| `batter` | MLBAM id of the batter | id | Statcast column `batter` (`datamining/outputs`) |
| `game_pk` | MLBAM game id. Its parity (odd or even) splits a season into two halves of games | id | Statcast column `game_pk` (`datamining/outputs`) |
| `at_bat_number` | Plate appearance number within the game; with `game_pk` it identifies a PA | count | Statcast column `at_bat_number` (`datamining/outputs`) |
| `pitch_number` | Pitch number within the plate appearance (1 for the first pitch). Also a P6 feature | count | Statcast column `pitch_number` (`datamining/outputs`) |
| `game_type` | R = regular season. Only R rows are used, except the one game a game page is built for | category | Statcast column `game_type` (`datamining/outputs`) |
| `release_pos_x_ft` | Release point, horizontal (catcher's view, + toward the first-base side) | ft | Statcast column `release_pos_x` (`datamining/outputs`) |
| `release_pos_y_ft` | Release point, distance from the back of home plate ($`y_r`$) | ft | Statcast column `release_pos_y` (`datamining/outputs`) |
| `release_pos_z_ft` | Release height | ft | Statcast column `release_pos_z` (`datamining/outputs`) |
| `vx0_ft_s` | Velocity of Statcast's fit at y = 50 ft, x component ($`v_{x0}`$) | ft/s | Statcast column `vx0` (`datamining/outputs`) |
| `vy0_ft_s` | Velocity of the fit at y = 50 ft, y component ($`v_{y0}`$, negative: toward the plate) | ft/s | Statcast column `vy0` (`datamining/outputs`) |
| `vz0_ft_s` | Velocity of the fit at y = 50 ft, z component ($`v_{z0}`$) | ft/s | Statcast column `vz0` (`datamining/outputs`) |
| `ax_ft_s2` | Constant acceleration of the fit, x ($`a_x`$) | ft/s² | Statcast column `ax` (`datamining/outputs`) |
| `ay_ft_s2` | Constant acceleration of the fit, y ($`a_y`$, positive: drag) | ft/s² | Statcast column `ay` (`datamining/outputs`) |
| `az_ft_s2` | Constant acceleration of the fit, z ($`a_z`$, gravity plus lift) | ft/s² | Statcast column `az` (`datamining/outputs`) |
| `plate_x_ft` | Horizontal position where the pitch crosses the front of the plate ($`x_p`$, catcher's view) | ft | Statcast column `plate_x` (`datamining/outputs`) |
| `plate_z_ft` | Height where the pitch crosses the front of the plate ($`z_p`$) | ft | Statcast column `plate_z` (`datamining/outputs`) |
| `y_p` | Depth of the front edge of the plate, 17/12 ft, where `plate_x_ft`/`plate_z_ft` are measured | ft | `reconstruct_trajectory.PLATE_FRONT_Y` |
| `t(y)` | Time, counted from y = 50 ft, at which the fit reaches depth y (negative before y = 50) | s | `reconstruct_trajectory._time_at_y` |
| `r_0` | Ball position at release (`rx`, `ry`, `rz`), rebuilt from the fit | ft | `reconstruct_trajectory.release_state_from_fit` |
| `v_r` | Ball velocity at release (`rvx`, `rvy`, `rvz`) | ft/s | `reconstruct_trajectory.release_state_from_fit` |
| `tf` | Flight time $`T`$ from release to the front of the plate. Pitches with $`T \ge 0.6`$ s are dropped | s | `features.add_state` |
| `p(t)` | Ball position $`t`$ seconds after release, $`\mathbf r_0 + \mathbf v_r t + \tfrac12 \mathbf a t^2`$ | ft | `features.position` |
| `E` | Batter's eye: $`(s_b \cdot (-35.8),\ -28.3,\ 65.0)`$ in, i.e. 35.8 in off the plate's center line on the batter's side, 28.3 in behind the back of the plate, 65 in high | ft | `features.EYE_*`, `to_polar.ORIGIN_*` |
| `theta(a,b)` | Angle between two ball positions as seen from the eye: $`\arccos\big((\mathbf a-\mathbf E)\cdot(\mathbf b-\mathbf E)/(\lVert\mathbf a-\mathbf E\rVert\,\lVert\mathbf b-\mathbf E\rVert)\big)`$ | deg | `features.angle_deg` |
| `pitch_type` | Statcast pitch type code: FF four-seam, SI sinker, FC cutter, SL slider, ST sweeper, CU curve, KC knuckle curve, SV slurve, CH changeup, FS splitter, FO forkball, KN knuckleball. PO (pitchout), EP (eephus), FA (position players), CS (slow curve), SC (screwball) and UN (unknown) are dropped | category | Statcast column `pitch_type` (`datamining/outputs`) |
| `stand` | Batter side, R or L. Sets the eye position and the sign $`s_b`$ | category | Statcast column `stand` (`datamining/outputs`) |
| `p_throws` | Pitcher hand, R or L. Sets the sign $`s_a`$ | category | Statcast column `p_throws` (`datamining/outputs`) |
| `release_speed_mph` | Release speed | mph | Statcast column `release_speed` (`datamining/outputs`) |
| `pfx_x_ft` | Horizontal movement relative to a spinless pitch, catcher's view | ft | Statcast column `pfx_x` (`datamining/outputs`) |
| `pfx_z_ft` | Vertical movement relative to a spinless pitch (induced vertical break) | ft | Statcast column `pfx_z` (`datamining/outputs`) |
| `release_spin_rate_rpm` | Spin rate at release | rpm | Statcast column `release_spin_rate` (`datamining/outputs`) |
| `release_extension_ft` | Release extension toward the plate | ft | Statcast column `release_extension` (`datamining/outputs`) |
| `sz_top_ft` | Top of the batter's strike zone for this pitch | ft | Statcast column `sz_top` (`datamining/outputs`) |
| `sz_bot_ft` | Bottom of the batter's strike zone for this pitch | ft | Statcast column `sz_bot` (`datamining/outputs`) |
| `balls` | Balls before the pitch | 0-3 | Statcast column `balls` (`datamining/outputs`) |
| `strikes` | Strikes before the pitch | 0-2 | Statcast column `strikes` (`datamining/outputs`) |
| `description` | Result of the pitch (ball, called_strike, swinging_strike, foul, hit_into_play, ...) | category | Statcast column `description` (`datamining/outputs`) |
| `s_b` | Batter-side sign: +1 for a right-handed batter, -1 for a left-handed one. Multiplying an x value by it makes + mean away from the batter | ±1 | `stand` |
| `s_a` | Pitcher-hand sign: +1 for a right-hander, -1 for a left-hander. Multiplying an x value by it puts both hands on one convention | ±1 | `p_throws` |
| `mov_x` | Horizontal movement mirrored for lefties, $`s_a\,`$`pfx_x_ft`. Arm-side run is negative for both hands (a right-hander's four-seamer averages -0.65 ft) | ft | `features.add_outcomes_and_controls` |
| `rel_x` | Horizontal release point mirrored for lefties, $`s_a\,`$`release_pos_x_ft`. Negative is toward the arm side | ft | `features.add_outcomes_and_controls` |
| `loc_x` | Plate location across the plate, $`s_b\,x_p`$. + is away from the batter | ft | `features.add_outcomes_and_controls` |
| `loc_z` | Plate height scaled to the batter's zone, $`(z_p - z_{bot})/(z_{top} - z_{bot})`$: 0 at the bottom of the zone, 1 at the top | zone heights | `features.add_outcomes_and_controls` |
| `platoon` | 1 if batter and pitcher are on the same side (`stand == p_throws`) | 0/1 | `features.add_outcomes_and_controls` |
| `prev_pitch_type` | Type of the previous pitch in the same PA (`none` on the first pitch) | category | `features.add_previous` |
| `prev_desc` | Result of the previous pitch, grouped: whiff, called, foul, ball, other, none | category | `feasibility.prep`, `perceptual.season` |
| `prev_loc_x` | Previous pitch's `loc_x`, using the current batter side | ft | `feasibility.prep`, `perceptual.season` |
| `prev_loc_z` | Previous pitch's height scaled to the current pitch's zone | zone heights | `feasibility.prep`, `perceptual.season` |
| `swing` | 1 if the batter swung: swinging_strike(_blocked), foul_tip, foul, foul_bunt, hit_into_play, missed_bunt, bunt_foul_tip (`features.SWING`) | 0/1 | `features.add_outcomes_and_controls` |
| `whiff` | 1 if the batter swung and missed: swinging_strike(_blocked), foul_tip, missed_bunt (`features.WHIFF`). Modelled on swings only, so the model is P(whiff \| swing) | 0/1 | `features.add_outcomes_and_controls` |
| `csw` | Called strike or whiff: 1 if `whiff` or `description == called_strike` | 0/1 | `features.add_outcomes_and_controls` |
| `CSW` | Share of pitches that are a called strike or whiff (`csw` averaged) | % or pts | `csw` |
| `primary fastball` | For a pitcher x batter side: the most-used of FF, SI and FC (to that side, in the data used for the features) | - | `arsenal.group_table`, `perceptual.primary_fastball` |
| `fb_dv` | Release speed minus the primary fastball's mean release speed | mph | `arsenal.group_table`, `perceptual.season` |
| `fb_dmx` | `mov_x` minus the primary fastball's mean `mov_x` | ft | `arsenal.group_table`, `perceptual.season` |
| `fb_dmz` | `pfx_z_ft` minus the primary fastball's mean `pfx_z_ft` | ft | `arsenal.group_table`, `perceptual.season` |
| `fb_drx` | `rel_x` minus the primary fastball's mean `rel_x` | ft | `arsenal.group_table`, `perceptual.season` |
| `fb_drz` | `release_pos_z_ft` minus the primary fastball's mean release height | ft | `arsenal.group_table`, `perceptual.season` |
| `arm_angle_deg` | Pitcher arm angle at release | deg | Statcast column `arm_angle` (`datamining/outputs`) |
| `bat_whiff_rate` | Batter's whiff per swing in the same season, leaving out the current game, shrunk to the league rate with 100 prior swings | rate | `perceptual.batter_rates` |
| `bat_swing_rate` | Batter's swing rate in the same season, leaving out the current game, shrunk with 100 prior pitches | rate | `perceptual.batter_rates` |
| `tau_b` | P1. Hitter's commit time from bat tracking: $`0.175 \times \tilde\rho_b/\bar\rho`$, clipped to 0.125-0.225 s, where $`\rho`$ = swing length / bat speed (swings over 40 mph), $`\bar\rho`$ the league mean and $`\tilde\rho_b`$ the batter's mean shrunk to it with 50 prior swings. 0.175 s if he has none | s | `perceptual.batter_tau` |
| `bat_speed_mph` | Bat speed on a tracked swing | mph | Statcast column `bat_speed` (`datamining/outputs`) |
| `swing_length_ft` | Swing length on a tracked swing | ft | Statcast column `swing_length` (`datamining/outputs`) |
| `t_c` | Commit time after release, $`T - \tau`$ (floored at 0 in the geometric features, at 0.05 s in the perceptual ones) | s | `features.add_features`, `perceptual.commit_features` |
| `v(t)` | Ball velocity $`t`$ seconds after release, $`\mathbf v_r + \mathbf a t`$ | ft/s | `perceptual.velocity` |
| `miss_x_in` | P2. Horizontal error of a fastball read: actual plate crossing minus where the ball would cross if, from the commit point, it kept the primary fastball's acceleration. Batter-oriented ($`s_b`$), + is away | in | `perceptual.commit_features` |
| `miss_z_in` | P2. Vertical error of the same fastball read; + means the pitch finished higher than the read | in | `perceptual.commit_features` |
| `miss_t_ms` | P2. Arrival-time error: actual time left at commit ($`\tau_b`$) minus the fastball read's time to the plate. + means the pitch arrives later than a fastball would | ms | `perceptual.commit_features` |
| `ang_speed` | P3. Angular speed of the ball as seen from the eye at commit, $`\lVert\mathbf v_\perp\rVert/r`$ | deg/s | `perceptual.angular_speed` |
| `ang_speed_vs_fb` | P3. `ang_speed` minus the angular speed of the primary fastball's mean path at the same time since release | deg/s | `perceptual.commit_features` |
| `sep_rate_vs_fb` | P3. Rate of change of the angle between the pitch and the fastball's mean path at commit, central difference with h = 5 ms | deg/s | `perceptual.commit_features` |
| `post_fb_commit` | P4. Posterior probability, from the ball's direction 175 ms before arrival, that the pitch is the primary fastball (Gaussian observer over the pitcher x side's types) | probability | `perceptual.posteriors` |
| `post_true_commit` | P4. Posterior probability, from the same view, of the pitch's true type | probability | `perceptual.posteriors` |
| `post_fb_release` | P4. As `post_fb_commit`, from the direction of the release point | probability | `perceptual.posteriors` |
| `post_true_release` | P4. As `post_true_commit`, from the direction of the release point | probability | `perceptual.posteriors` |
| `w_k` | G. Posterior weight of type $`k`$ from the ball's direction 175 ms before arrival (the P4 observer's full posterior) | probability | `group_read.group_features` |
| `gmiss_x_in` | G. Horizontal miss of the posterior-weighted group read at the plate. Batter-oriented, + away | in | `group_read.group_features` |
| `gmiss_z_in` | G. Vertical miss of the group read; + means higher than the read | in | `group_read.group_features` |
| `gmiss_t_ms` | G. Arrival-time miss of the group read; + means later | ms | `group_read.group_features` |
| `gspread_in` | G. Weighted spread of the look-alike types' plate crossings around the group read | in | `group_read.group_features` |
| `surprise_count` | P5. Surprisal of the type given the count, $`-\log_2 P(\text{type} \mid \text{pitcher, side, count})`$, usage smoothed with 5 prior pitches | bits | `perceptual.surprisal` |
| `count` | `balls-strikes` before the pitch, e.g. `1-2` (12 values) | category | built from `balls`, `strikes` in `features.add_outcomes_and_controls` |
| `surprise_prev` | P5. Surprisal given the previous pitch type, $`-\log_2 P(\text{type} \mid \text{pitcher, side, previous type})`$; missing on a PA's first pitch | bits | `perceptual.surprisal` |
| `fb_seen_pa` | P6. Fastballs (FF, SI, FC) already thrown earlier in the PA | count | `perceptual.season` |
| `sep_last_fb` | P6. Angle from the eye at commit between this pitch and the most recent earlier fastball in the PA (same time since release); missing if none | deg | `perceptual.season` |
| `n_priorpa_thisgame` | Times the batter has already faced this pitcher in the game (times through the order minus 1). Also a P6 feature | count | Statcast column `n_priorpa_thisgame_player_at_bat` (`datamining/outputs`) |
| `plate_speed_mph` | Speed of the fit at the front of the plate, $`\lVert\mathbf v(T)\rVert/1.46667`$ | mph | `speed_check.add_speed` |
| `flight_ms` | Flight time $`1000\,T`$ | ms | `speed_check.add_speed` |
| `prev_dv` | Release speed minus the previous pitch's release speed | mph | `speed_check.add_speed` |
| `prev_dtf_ms` | Flight time minus the previous pitch's flight time | ms | `speed_check.add_speed` |
| `fb_dtf_ms` | Flight time minus the primary fastball's mean flight time | ms | `speed_check.add_speed` |
| `month` | Calendar month of the game; Mar-Jul vs Aug-Oct splits a season in time | 1-12 | `game_date` |
| `TAU` | Commit point: how long before the ball reaches the front of the plate the hitter must decide. Fixed at 0.175 s unless stated | s | `features.TAU` |
| `TRACK` | Tracking limit: 0.100 s before arrival, the last moment the hitter can still follow the ball | s | `features.TRACK` |
| `d_dec_deg` | Angle from the eye between this pitch at its commit point $`t_c = T-\tau`$ and the previous pitch at the same time since its release (capped at its own $`T`$) | deg | `features.add_features` |
| `d_track_in` | Distance in the x-z plane between this pitch and the previous one at the same depth: the depth this pitch reaches 0.100 s before arrival | in | `features.add_features` |
| `d_plate_in` | Distance between the two pitches' plate crossings, $`12\sqrt{\Delta x_p^2+\Delta z_p^2}`$ | in | `features.add_features` |
| `same_type` | 1 if the pitch and the previous pitch have the same `pitch_type` | 0/1 | `feasibility.prep` |
| `tv` | Tunnel value of a swing: predicted whiff probability with the pitch's real tunnel features minus with the features set to its pitch type's median | probability (shown in pct points) | `feasibility.check_pitchers` |
| `resid` | Residual: actual outcome (0/1) minus a model's predicted probability from a model without the features being tested | probability (shown in pct points) | `feasibility`, `arsenal`, `perceptual` |
| `a_dec_deg` | Angle from the eye between a pitch type's average path and the primary fastball's average path, at the fastball's commit point ($`T_{fb}-\tau`$, same time since release) | deg | `arsenal.group_table` |
| `a_track_in` | x-z distance between the type's average path and the fastball's at the depth the type reaches 0.100 s before arrival | in | `arsenal.group_table` |
| `a_plate_in` | Distance between the type's and the fastball's average plate crossings | in | `arsenal.group_table` |
| `M0` | Control model: LightGBM on CONTROLS = BASE + PREV + fastball differentials + `arm_angle_deg` + batter rates | - | `perceptual.CONTROLS` |
| `M1` | M0 plus all 17 perceptual features (P1-P6) | - | `perceptual.ALL_P` |
| `M0s` | M0 plus the five speed controls | - | `speed_check.SPEED` |
| `M1s` | M0s plus all perceptual features | - | `speed_check.main` |
| `k_x` | Ball position at commit, batter-oriented: $`(s_b p_x, p_y, p_z)`$ as `k_x, k_y, k_z` | ft | `speed_check.add_speed` |
| `k_vx` | Ball velocity at commit, batter-oriented: $`(s_b v_x, v_y, v_z)`$ as `k_vx, k_vy, k_vz` | ft/s | `speed_check.add_speed` |
| `pv` | P value of a swing: M1's predicted whiff probability minus M1 with every P feature set to its pitch type's median | probability | `perceptual.main` |
| `LightGBM` | Gradient-boosted trees. Settings everywhere unless stated: binary log-loss objective, learning rate 0.05, 63 leaves, >= 200 rows per leaf, 80% of features and 80% of rows per tree, 400 trees, seed 7 | - | `feasibility.PARAMS`, `feasibility.ROUNDS` |
| `log-loss` | $`-\frac1n\sum_i [y_i\log\hat p_i + (1-y_i)\log(1-\hat p_i)]`$ on held-out pitches; lower is better | nats per pitch | `sklearn.metrics.log_loss` |
| `gain x1000` | $`1000\,(L_{\text{without}} - L_{\text{with}})`$: how much adding the features lowers held-out log-loss. + helps | nats per 1000 pitches | computed in each script |
| `null` | The same model refit after shuffling the tested features among pitches of the same pitcher x pitch type (or group), which keeps each feature's spread but breaks its link to the pitch | - | `feasibility.shuffled`, `perceptual.shuffled` |
| `null gains` | Gain x1000 of each shuffled-feature refit; the real gain has to beat all of them | nats per 1000 pitches | computed in each script |
| `share of nulls beaten` | Share of null draws whose gain is below the real gain | % | computed in each script |
| `R²` | $`1 - \sum(y-\hat y)^2/\sum(y-\bar y)^2`$ on held-out rows | share | `sklearn.metrics.r2_score` |
| `split-half r` | Pearson correlation of a group's value in odd-`game_pk` games with its value in even-`game_pk` games | r | computed in each script |
| `Spearman-Brown` | $`2r/(1+r)`$: the split-half r stepped up to the reliability of the full season | r | `feasibility.check_pitchers` |
| `r` | Pearson correlation | -1 to 1 | computed in each script |
| `pts` | Percentage points (a rate difference x100) | pts | - |
| `P1-P6` | The six perceptual feature blocks: timing, fastball-prior miss, optical motion, ambiguity, expectation, at-bat build-up | - | `perceptual.P` |
| `Tunnel+` | A per-pitcher tunnel score scaled to mean 100, SD 10, as proposed in the original outline. Not built, because the pitcher-level tests failed | index | `docs/original_outline.md` |
| `Stuff+` | FanGraphs' pitch-quality grade (100 = average) from velocity, movement, spin and release; Pitching+ also grades location, and PitchingBot is a separate pitch model. Used here as public comparison pickers | index | `datamining/outputs/fangraphs_grades_2025.csv` |
| `cell` | Pitcher x batter side x count x previous pitch type. The grid holds, for each candidate type in the cell, predicted swing %, whiff % per pitch, CSW % and deception lift (and runs saved per 100 in the run-value version) | - | `sequencing.build_cells` |
| `menu` | A pitcher's candidate types to a side: >= 40 pitches and >= 4% usage in 2026; pitchers need >= 1,000 2026 pitches and >= 150 to the side | - | `sequencing.MIN_*` |
| `top pick` | The candidate type with the highest predicted CSW in the cell | - | `validate_sequencer.test_pitches` |
| `matched` | 1 if the type actually thrown equals the pick being tested | 0/1 | each validation script |
| `gap pts` | Within-count gap: in each count, CSW of matched pitches minus unmatched, then averaged over counts weighted by pitches per count | pts | `validate_sequencer.within_count_gap` |
| `95% CI` | 2.5th to 97.5th percentile of the statistic over a pitcher bootstrap: pitchers drawn with replacement, each pitcher's pitches weighted by how often he was drawn | - | `validate_sequencer.boot` |
| `naive pick` | The pitcher's type with the highest observed 2025 CSW to that side in that count (>= 30 pitches), else his highest 2025 CSW type to that side overall | - | `validate_sequencer.naive_picks` |
| `seq_delta` | Predicted CSW of (actual previous type -> actual type) minus the mean predicted CSW of that type over all previous types in that count | probability (pts) | `validate_sequencer.test_pitches` |
| `lift` | Deception lift: predicted CSW minus predicted CSW with the look features (P2, P3, P4) set to the pitch type's 2026 median | pts | `sequencing.main` |
| `class` | Pitch class: fb (FF, SI, FC), brk (SL, ST, CU, KC, SV, CS), off (everything else) | category | `validate_choice.pclass`, `hitter_sequencer.pclass` |
| `bat_all` | Batter's 2025 CSW-against rate, shrunk to the league rate with 200 prior pitches | rate | `validate_choice.batter_rates` |
| `bat_cls` | Batter's 2025 CSW-against rate on the class of the pitch thrown, shrunk with 200 prior pitches | rate | `validate_choice.batter_rates` |
| `fixed effects` | A separate intercept per group (here pitcher x side x count). Estimated by subtracting each group's mean from every variable before the regression | - | `validate_choice.within_ols` |
| `propensity` | $`e = P(\text{matched} = 1 \mid \text{situation})`$ from a cross-fitted LightGBM, trimmed to 0.05-0.95 | probability | `validate_choice.propensity` |
| `IPW gap` | $`\frac{\sum d_i y_i/e_i}{\sum d_i/e_i} - \frac{\sum (1-d_i) y_i/(1-e_i)}{\sum (1-d_i)/(1-e_i)}`$, with $`d_i`$ = matched | pts | `validate_choice.ipw_gap` |
| `tau` | A commit time on the grid {125, 150, 175, 200, 225} ms | ms | `hitter_timing.GRID` |
| `tau*` | League-best grid commit time: the grid value with the highest total out-of-fold log-likelihood on 2025 | ms | `hitter_timing.main` |
| `tau_fit` | A hitter's fitted commit time: posterior mean over the grid | s | `hitter_timing.posterior_tau`, `batter_tau_2025.csv` |
| `bat_cls_whiff` | Hitter's whiff per swing against the class of the pitch, shrunk to the league class rate with 100 prior swings | rate | `hitter_sequencer.shrink` |
| `bat_cls_swing` | Hitter's swing rate against the class, shrunk with 100 prior pitches | rate | `hitter_sequencer.shrink` |
| `bat_cls_chase` | Hitter's swing rate on out-of-zone pitches (`zone` > 9) of the class, shrunk with 100 prior out-of-zone pitches | rate | `hitter_sequencer.shrink` |
| `zone` | Statcast zone of the plate crossing: 1-9 inside the strike zone (3 x 3 grid), 11-14 the four outside quadrants | category | Statcast column `zone` (`datamining/outputs`) |
| `model H` | The sequencer model (CONTROLS + SPEED + all P) plus the three `bat_cls_*` features | - | `hitter_sequencer.COLS_H` |
| `adj` | Batter adjustment for a pitcher x side x batter x count x type: mean predicted CSW over the candidate pitches with this hitter's features minus with the average hitter's (previous pitch = primary fastball, none at 0-0) | pts | `hitter_sequencer.build` |
| `side pick` | Top pick for the average hitter | - | `hitter_sequencer.validate` |
| `batter pick` | Type with the highest side-only CSW + `adj` for this hitter | - | `hitter_sequencer.validate` |
| `adherence` | Mean over a PA's scored pitches of (matched minus the pitcher x side x count 2026 match rate) | share | `validate_atbat.scored_pitches`, `pa_table` |
| `K` | 1 if the PA ends in a strikeout | 0/1 | `validate_atbat.pa_results` |
| `cxwoba` | `xwoba` on PAs that end in a ball in play (contact xwOBA) | wOBA scale | `validate_atbat.pa_results` |
| `hard` | 1 if a ball in play left the bat at >= 95 mph | 0/1 | `validate_atbat.pa_results` |
| `woba_value` | Actual wOBA value of the PA's result | wOBA scale | Statcast column `woba_value` (`datamining/outputs`) |
| `launch_speed_mph` | Exit velocity on a ball in play | mph | Statcast column `launch_speed` (`datamining/outputs`) |
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
| `picker A` | Perceptual sequencer: CONTROLS + SPEED + all P, highest CSW | - | `game_report.COLS_A` |
| `picker B` | Standard pitch model: CONTROLS + SPEED, no perceptual features, highest CSW | - | `game_report.COLS_B` |
| `picker C` | Stuff-only: pitch type, hands, speed, movement, spin, extension, release, arm angle, fastball differentials. No location, count or previous pitch | - | `game_report.COLS_C` |
| `picker D` | Best pitch for the count: the naive pick | - | `validate_sequencer.naive_picks` |
| `picker E` | Pitcher's habit: his most-used 2026 type to that side in that count | - | `game_report.habit_picks` |
| `RA` | Perceptual by runs: model H features, lowest E[RV] | - | `runs_compare.SETS` |
| `RB` | Standard by runs: model H features minus all perceptual features, lowest E[RV] | - | `runs_compare.SETS` |
| `RC` | Stuff-only by runs: C's features, lowest E[RV] | - | `runs_compare.SETS` |
| `FS` | FanGraphs Stuff+ (2025): his menu type with the highest grade, in every cell (`sp_s_<type>`) | - | `fangraphs_grades_2025.csv` |
| `FP` | FanGraphs Pitching+ (2025), `sp_p_<type>` | - | `fangraphs_grades_2025.csv` |
| `FB` | PitchingBot overall (2025), `pb_o_<type>` | - | `fangraphs_grades_2025.csv` |
| `LA` | Perceptual, type + zone: candidate (type, zone) with the lowest E[RV], RA features | - | `location_compare.SETS` |
| `LB` | Standard, type + zone: the same with RB features | - | `location_compare.SETS` |
| `LZ` | League location chart: candidate with the lowest 2025 league mean realized RV for type x zone x count x platoon (>= 50 pitches, else type x zone x platoon) | - | `location_compare.league_chart` |
