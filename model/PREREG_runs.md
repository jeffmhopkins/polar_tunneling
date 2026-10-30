# Model comparison by runs: pre-registration

Written and committed before any picker in this step is trained or scored.
Run once. Train on 2025, test on 2026.

## Why

`PREREG_game.md` compared pickers by CSW only. The perceptual pick tied the
standard model's (+0.16 CSW pts, CI -0.06 to +0.38). The run-value sequencer
(`PREREG_damage.md`) ranks pitches by expected runs, which changes the pick
on 46% of pitches. This step asks whether the perceptual features matter
when pitches are chosen and scored by runs.

## Pickers

All from 2025 only, average hitter, built as in `game_report.py` (each
candidate type represented by the same real 2026 pitches, placed in the
count after the previous type). Expected run value per pitch as in
`damage_sequencer.py` (`run_values`, `expected_rv`, six models).

- **RA. Perceptual, by runs**: the run-value sequencer's features
  (`hitter_sequencer.COLS_H`); pick = lowest E[RV].
- **RB. Standard, by runs**: the same minus every perceptual feature
  (`perceptual.ALL_P`).
- **RC. Stuff-only, by runs**: `game_report.COLS_C`.
- **A. Perceptual, by CSW**: highest predicted CSW from RA's CSW model.
- **D. Best pitch for the count** and **E. Pitcher's habit**: as in
  `PREREG_game.md`.

## Test

Test pitches: 2026 regular season, type, side, count and previous type in
the pitcher's grid (minimums as in `sequencing.py`). Realized value as in
`PREREG_damage.md` (2025 run values, xwOBA on contact).

For each picker: within-count realized runs per 100 pitches when the pitch
thrown matched its pick minus when it didn't (negative is better), pitcher
bootstrap, 1,000 draws.

**Primary: RA's gap minus RB's gap.** The perceptual features help pick by
runs if < 0 with a 95% CI excluding 0.

Secondary (reported, not pass/fail): RA minus RC, A, D and E; the same
comparisons on CSW; share of pitches where each pair agrees; realized runs
using actual delta_run_exp.

## Game report

The Orioles game page (game 823491) gets RA, RB and RC picks and RA's
runs saved per 100 for every candidate. Descriptive only.
