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
