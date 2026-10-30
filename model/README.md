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
