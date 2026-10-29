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
