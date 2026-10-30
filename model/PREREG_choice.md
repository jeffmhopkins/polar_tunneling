# Tool vs pitcher choice: pre-registration

Written and committed before running. Single run on the same 2026 test
pitches and the same 2025-only sequencer as `PREREG_validation.md`.

## Question

Pitches that matched the sequencer's top pick had a CSW 4.48 pts higher than
others in the same count. Pitchers pick pitches knowing the batter and the
situation. How much of the 4.48 pts is left once that choice is accounted for?

## Covariates (no 2026 outcomes)

- **Batter:** from 2025, his CSW-against rate overall and against the class
  of the pitch thrown (fastball FF/SI/FC, breaking SL/ST/CU/KC/SV/CS,
  offspeed CH/FS/FO/other), shrunk to the league rate with a prior of 200
  pitches. Batters without 2025 data get the league rate and a flag.
- **Situation:** outs, runners on 1st/2nd/3rd, inning (1-3, 4-6, 7+),
  platoon.
- **Previous pitch:** its class and its result (ball, called strike, whiff,
  foul, other).
- **Fixed effects:** pitcher x batter side x count.

## Estimates

**A. Adjusted gap (primary).**
- A linear probability model of CSW on matched + the covariates + the fixed
  effects, estimated by within-group demeaning.
- Report the coefficient on matched (pts) and its share of the raw 4.48.
- 95% CI from a pitcher bootstrap (500 draws).
- The tool effect **survives** if the coefficient is > 0 with a CI excluding 0.

**B. Propensity check.**
- P(matched) is modeled from the covariates + pitcher + count + previous pitch
  type, with LightGBM and 5-fold cross-fitting. No outcomes are used.
- The estimate is an inverse-propensity-weighted gap (normalized weights,
  propensity trimmed to [0.05, 0.95]), with a pitcher-bootstrap CI (500
  draws, propensities held fixed).
- Also reported: the within-count gap in each propensity tercile. The lowest
  tercile is pitches where the top pick was unlikely given the situation, so
  a pitcher who threw it there broke his usual pattern.

Reading: if A and B both stay clearly above 0, a real share of the gain is
the tool's choice, not the pitcher's. Whatever remains is still an upper
bound, because unmeasured factors (catcher, scouting, game plan) are not
controlled.
