# Model comparison and game report: pre-registration

Written and committed before any comparison model's picks are scored
against 2026 outcomes. The season comparison is run once.

## Models

All are trained on 2025 only and scored the same way: each candidate pitch
type is represented by the same real 2026 pitches of that pitcher (shapes and
locations, as in `sequencing.py`), placed in the count and after the
previous pitch type, for an average hitter. The top pick is the type with the
highest predicted CSW in the pitcher x side x count x previous-type cell.

- **A. Perceptual sequencer**: the validated sequencer model (controls +
  speed + perceptual features), i.e. `sequencing.py --train-years 2025`.
- **B. Standard pitch model**: the same LightGBM on the controls + speed
  only (stuff, location, count, previous pitch, fastball differentials, arm
  angle, batter rates). No perceptual features.
- **C. Stuff-only model** (Stuff+-style): pitch type, hands, velocity,
  movement, spin, extension, release point, arm angle and fastball
  differentials. No location, count or previous pitch, so its pick is the
  pitcher's best-stuff pitch to that side whatever the situation.
- **D. Best pitch for the count**: the naive pick from
  `PREREG_validation.md` (highest observed 2025 CSW to that side in that
  count, >= 30 pitches, else highest overall).
- **E. Pitcher's habit**: the type he threw most often to that side in that
  count in 2026 (descriptive; it uses 2026 usage, not outcomes).

## Season comparison (2026 regular season)

Test pitches as in `PREREG_validation.md`: the pitch's type, side, count and
previous type must be in that pitcher's grid. For each model, the
within-count CSW gap of pitches that matched its pick minus pitches that
didn't, with a 2,000-draw pitcher bootstrap.

Primary: **A's gap minus B's gap** on the same pitches. A beats the standard
model if > 0 with a 95% interval excluding 0. Also reported: A minus C,
A minus D, A minus E, and the share of pitches where each pair agrees.

Caveat stated in advance: this is observational, like the earlier
validation. Pitchers choose with knowledge we don't have.

## Game report (descriptive, not a test)

One 2026 postseason game (default: World Series Game 5, game 849799). The
models never saw postseason pitches. For every pitch: each model's pick and
predicted CSW for each candidate, the pitch actually thrown, and the result.
Game pitchers get the same grids with lower minimums (>= 300 regular-season
pitches, >= 60 to a side, >= 15 of a type). One game is far too small to rank
models; the season comparison is the evidence.

Amendment (before any run): Jeff asked for an Orioles game, so the default
game is now 823491, Orioles at Yankees on 2026-09-25 (their latest game in
the data). It is a regular-season game, so its own pitches are left out of
the 2026 shapes and usage used to build the grids. The model still never saw
any 2026 outcome.
