# Whole at-bat test: pre-registration

Written and committed before any 2026 at-bat outcome is compared with the
sequencer's picks. Run once.

## Question

Pitches that match the sequencer's top pick get more called strikes and
whiffs (`validation_report.md`, `choice_report.md`). Does following the tool
across an at-bat also produce more strikeouts and weaker contact, or does the
per-pitch CSW gain wash out (for example because the picks get hit hard when
they are put in play)?

## Setup

- Sequencer: the 2025-only grid (`sequencing_2025model.json`), the same one
  used in `PREREG_validation.md` and `PREREG_choice.md`.
- Plate appearances: 2026 regular season, pitcher and batter side in the grid,
  ending in a strikeout, walk, hit by pitch or ball in play. Sacrifice bunts,
  catcher interference, intentional walks and truncated PAs are dropped.
- A pitch is **scored** when its count x previous-type cell is in the grid.
  matched = 1 if the pitch type thrown is that cell's top pick, else 0 (a type
  the grid doesn't list counts as not matched). PAs need >= 1 scored pitch.
- **Adherence** of a PA = mean over its scored pitches of (matched minus the
  pitcher's 2026 match rate in that pitcher x side x count). This removes the
  fact that the top pick is thrown more often in some counts, so a PA is not
  credited with adherence just because it reached a two-strike count. The base
  rates use only which pitch was thrown, never results.
- Effects are reported per +25 points of adherence (about a quarter more of
  the PA's pitches following the tool than that pitcher usually does).

## Outcomes (2026)

- K: PA ends in a strikeout.
- xwOBA: Statcast's PA-level expected wOBA (K = 0, walk/HBP = their weights,
  ball in play = from exit speed and angle).
- Contact xwOBA: xwOBA on PAs that end in a ball in play.
- Also: walk + HBP rate, hard-hit rate (exit speed >= 95 mph) on balls in
  play, actual wOBA.

## Controls (no 2026 outcomes)

- Fixed effects: pitcher x batter side.
- Batter, from 2025, shrunk to the league rate: K rate and walk + HBP rate
  (prior 200 PA), xwOBA per PA (prior 200 PA), contact xwOBA (prior 100 balls
  in play). Batters without 2025 data get the league rate and a flag.
- Situation: outs, runners on 1st/2nd/3rd, inning (1-3, 4-6, 7+), platoon,
  times the batter has faced the pitcher earlier in the game.

These are the controls from `PREREG_choice.md`, with the batter rates changed
to match the PA outcomes.

## Estimates

A linear probability (or linear, for xwOBA) model of each outcome on
adherence + controls, estimated by within-group demeaning on the fixed
effects. 95% CIs from a pitcher bootstrap (500 draws).

### Primary (three tests)

1. **More strikeouts:** K coefficient > 0, CI excluding 0.
2. **Fewer runs:** xwOBA coefficient < 0, CI excluding 0.
3. **Weaker contact:** contact xwOBA coefficient < 0, CI excluding 0. If the
   CI includes 0 but its upper end is below +0.005 per +25 pts, report that
   the picks are not hit harder, without calling it a pass.

### Secondary (reported, not pass/fail)

- **First pitch only.** The same models with adherence replaced by "the 0-0
  pitch matched the top pick" (minus that pitcher x side's 0-0 match rate).
  The first pitch is chosen before anything happens in the PA, so the count
  path cannot feed back into the exposure.
- **Pitch-level contact check.** On balls in play, contact xwOBA of pitches
  that matched the top pick vs not, with pitcher x side x count fixed effects
  and the same controls. This asks directly whether top picks get hit harder
  when they are put in play.
- **Implied vs observed K.** For scale, a simple chain: a = the adherence
  coefficient on the PA's number of CSW pitches, b = the coefficient of K on
  the number of CSW pitches (same controls and fixed effects). a x b is the K
  change a CSW-only effect would suggest; compare it with the K coefficient.
  This is descriptive.
- Walk + HBP rate, hard-hit rate, actual wOBA, and a table of adjusted
  outcomes by adherence quartile.

## Caveats stated in advance

This is observational, like the earlier checks. Catcher, scouting and game
plans are not measured, so the estimates are upper bounds on what the tool
adds. Within a PA, later pitches depend on earlier results, which is why the
first-pitch version is reported alongside.
