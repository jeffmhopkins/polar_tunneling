# Per-hitter timing and a batter-specific sequencer: pre-registration

Written and committed before any model in this step is trained or any 2026
outcome is looked at. Each part is run once. Train on 2025, score on 2026.

## Part 1. Per-hitter decision point

Today the commit point is 175 ms before arrival for the Bayesian-observer
features (P4), and the other commit features use a timing estimated from bat
tracking (swing length / bat speed), which added almost nothing (P1 ablation
-0.01 whiff, 0.37 swing). Here the commit point is fitted to each hitter from
his outcomes.

**Grid.** tau in {125, 150, 175, 200, 225} ms. At each grid value, every
commit feature (P2 fastball-prior miss, P3 optical motion, P4 commit
posteriors, P6 separation from the last fastball) is computed with every
hitter at that tau.

**Fit (2025 only).**
- Models: swing and whiff | swing, features = the speed-check model
  (CONTROLS + SPEED + all P blocks), fixed PARAMS, 400 rounds.
- 2-fold cross-fit on 2025, folds = game_pk parity, giving an out-of-fold
  log-likelihood for every pitch at every grid tau (swing on all pitches,
  whiff on swings; both summed per pitch).
- League curve: total log-likelihood by tau. tau* = the best grid value.
- Batter score S_b(tau) = sum over his pitches of (log-lik at tau minus the
  league per-pitch mean at tau for the same outcome), so a league-wide shift
  does not move individual hitters.
- Posterior over the grid: prior = discretized normal centered at tau*,
  sd 25 ms, times exp(S_b(tau)). Fitted tau_b = posterior mean. Hitters with
  no 2025 pitches get tau*.
- Commit posteriors (P4) at a non-grid tau are interpolated linearly between
  the two nearest grid values; P2, P3 and P6 are computed exactly.

**Test (2026, single run).** All arms trained on 2025:
- **A, current:** bat-tracking tau_b, P4 at 175 ms (the speed-check M1s).
- **B, fitted:** tau_b = fitted tau (from 2025), all commit features at it.
- **C, league:** everyone at tau* (to separate a league shift from the
  per-hitter part). Reported, not pass/fail.
- **Nulls:** B with fitted taus shuffled among hitters, 5 shuffles.

**Pass**, per outcome (whiff | swing, swing): log-loss gain of B over A
x1000 >= 0.3 and above the largest null gain over A. Part 1 passes if at
least one outcome passes and neither outcome has B worse than A by more than
0.3.

Also reported: split-half reliability of fitted tau (odd vs even 2025 games,
hitters with >= 500 pitches in each half), its correlation with the
bat-tracking tau, and the spread of fitted taus.

## Part 2. Batter-specific sequencer

**New batter features**, for the class of the pitch thrown (fastball
FF/SI/FC, breaking SL/ST/CU/KC/SV/CS, offspeed other):
- bat_cls_whiff: whiff per swing
- bat_cls_swing: swing rate
- bat_cls_chase: swing rate on pitches outside the zone (zone > 9)

Shrunk to the league rate for that class with a prior of 100 pitches (100
swings for whiff, 100 out-of-zone pitches for chase). In 2025 training rows
they leave out the current game. Applied to 2026, they come from all of 2025;
hitters without 2025 data get league rates.

**Model H** = the sequencer model (CONTROLS + SPEED + all P) plus the three
features, trained on 2025, for swing, whiff | swing and CSW. Timing: fitted
tau if Part 1 passes, otherwise bat-tracking tau. The average hitter is tau*
(Part 1 passes) or 175 ms, with median rates and league class rates.

**Scores.**
- Side-only score: the current sequencer construction (count x previous type
  x type, average hitter) from model H.
- Batter adjustment adj(pitcher, side, batter, count, type): mean over the
  same candidate pitches of predicted CSW with the hitter's features minus
  with the average hitter's, with the previous pitch set to the pitcher's
  primary fastball to that side (none at 0-0).
- Batter-specific score = side-only score + adj.
- Picks = the highest score in the cell.

**Test pitches.** 2026 pitches covered by the grid as in
PREREG_validation.md, where the hitter saw >= 200 pitches in 2025.

**Primary.** Pitches where the batter pick and the side pick differ and the
thrown type is one of the two. Within-count CSW (weighted by pitches per
count) when the thrown type is the batter pick, minus when it is the side
pick. 95% CI from a pitcher bootstrap (1,000 draws). **Pass** if > 0 with
the CI excluding 0.

**Secondary (reported, not pass/fail).**
- Matched-minus-unmatched within-count gap for the batter pick minus the
  same gap for the side pick, on all test pitches, with a bootstrap CI.
- Share of cells where the two picks differ.
- 2026 log-loss gain x1000 of model H over the same model without the batter
  class features, for swing, whiff | swing and CSW.

Reading: Part 2 asks whether knowing who is at the plate changes the best
pitch in a way that shows up in 2026 results. Pitchers also scout hitters,
so a pass is an upper bound on what the tool adds beyond that.
