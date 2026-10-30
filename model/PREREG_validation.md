# Sequencer validation: pre-registration

Written and committed before any 2026 outcome is compared with the
sequencer's picks. Run once.

## Setup

- Out-of-sample sequencer: `sequencing.py --train-years 2025`. The model,
  the previous-result lookup and the batter medians use 2025 only. From 2026
  it uses only pitch shapes, locations and pitch-type usage, never results.
- Test pitches: 2026 regular-season pitches by the sequencer's pitchers. The
  pitch's type, the batter side and the count must be in that pitcher's grid,
  and the previous pitch in the PA must be in his grid (0-0 has none).
- CSW = called strike or whiff (`features.WHIFF` + `called_strike`).
- Top pick = the type with the highest predicted CSW in that pitcher x side x
  count x previous-type cell.
- Every comparison is made within count: the gap is computed per count, then
  averaged with weights equal to the number of test pitches in each count.
- 95% intervals come from a bootstrap that resamples pitchers (2,000 draws).

## Tests

1. **Primary: top pick vs not.** The within-count CSW% of pitches that matched
   the top pick, minus pitches that didn't. **Pass** if > 0 and the 95%
   interval excludes 0.
2. **Beats a naive pick.** Naive pick = the type with the pitcher's highest
   observed 2025 CSW to that side in that count (types with >= 30 pitches in
   that cell; otherwise his highest 2025 CSW type to that side overall). Same
   matched-minus-unmatched gap, restricted to pitches where both picks exist.
   **Pass** if the sequencer's gap minus the naive gap is > 0 with an interval
   excluding 0.
3. **After-what effect.** For each test pitch with a previous pitch:
   - seq_delta = the predicted CSW of (actual previous type -> actual type)
     minus the mean predicted CSW of that type over all previous types in
     that cell's count
   - within pitcher x type x side x count groups, compare the CSW of pitches
     with seq_delta above the group median vs at or below it (groups with
     >= 10 pitches on each side)
   - **Pass** if the pooled gap is > 0 with an interval excluding 0

   This asks whether the previous pitch the tool favors actually helps the
   same pitch in the same count.

Also reported, not pass/fail: the gap per count, and the share of 2026
pitches that matched the top pick.
