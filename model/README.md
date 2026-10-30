# model/

Go/no-go checks for the sequence tunnel score: each pitch is compared with the
previous pitch in the same plate appearance (see the tunnel score proposal).

- `features.py` rebuilds each pitch from Statcast's 9-parameter fit, views it
  from the batter's eye, and computes the separations against the previous pitch:
  angular at the commit point (`d_dec_deg`), and physical at the tracking limit
  (`d_track_in`) and at the plate (`d_plate_in`).
- `feasibility.py` runs the checks and writes `feasibility_report.md`.
- `robustness.py` repeats the lift test at 125/175/225 ms, on different-type
  pairs only, with more null draws, and writes `robustness_report.md`.

## Findings (2025)

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

The sequence tunnel score fails the go/no-go as designed.

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
