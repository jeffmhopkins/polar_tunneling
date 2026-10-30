# Location in the sequencer: pre-registration

Written and committed before any location picker is trained or scored. Run
once. Train on 2025, test on 2026.

## Why

Every picker so far chose a pitch type only. The perceptual features did not
change that choice (`PREREG_game.md`, `PREREG_runs.md`). Tunneling is about
where a pitch goes relative to the one before, so a picker that also
chooses the location, given the real previous pitch, is the fairest test of
whether the perceptual features matter for decisions.

## Candidates

For each pitcher x batter side, a candidate is a (pitch type, Statcast
zone) pair: zones 1-9 in the strike zone and 11-14 outside it. A pair needs
>= 8 of his 2026 pitches of that type to that side in that zone; types use
the menu rules of `sequencing.py`. Each candidate is represented by 4 of
those real pitches (shape, release and plate location; fixed seed).

## Context

Unlike the earlier grids, each test pitch is scored in its real context:
its count, pitch number and times through the order, the actual previous
pitch in the plate appearance (type, full trajectory, location and result)
and the last fastball seen in the PA. The hitter is the average hitter, as
before. Commit-point, surprise and build-up features are recomputed for each
candidate in that context, as `sequencing.build_cells` does.

## Pickers

Models: the 2025 run-value models from `runs_compare.py` (six models each,
`damage_sequencer.expected_rv`).

- **LA. Perceptual, type + zone**: candidate with the lowest E[RV], RA
  features.
- **LB. Standard, type + zone**: the same with RB features (no perceptual
  features).
- **LZ. League location chart**: candidate with the lowest 2025 league mean
  realized run value for that pitch type x zone x count x platoon (cells
  with >= 50 pitches, else type x zone x platoon). No model, no sequence.

## Test

Test pitches: a random 15% of 2026 regular-season pitches (seed 7) by
pitchers with a menu, where the pitch's type is in his menu and the previous
pitch, if any, is in the data. A pitch matches a picker when its type and
its zone both equal the pick. Realized value and the within-count matched
minus unmatched gap as in `PREREG_runs.md` (runs per 100 pitches, negative
is better; 1,000-draw pitcher bootstrap).

**Primary: LA's gap minus LB's gap.** The perceptual features help choose
type and location if < 0 with a 95% CI excluding 0.

Secondary (reported, not pass/fail):
- LA minus LZ: value beyond a league location chart.
- The same comparisons on CSW.
- Share of pitches matching each pick, and how often LA and LB agree on
  type, on zone and on both.

Caveat stated in advance: location outcomes are partly command. A pitch
that lands in the picked zone may have been aimed elsewhere, so this tests
the value of where pitches end up, not of where they were aimed.

## Game report

The Orioles game page (game 823491) gets LA, LB and LZ picks for every pitch
with a menu. Descriptive only.
