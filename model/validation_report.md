# Sequencer validation on real 2026 pitches (pre-registered, single run)

Sequencer trained on 2025 only. Test pitches: 471,215 from 265 pitchers. Share that matched the top pick: 28.6%.

## 1. Top pick vs not (primary)

Within-count CSW gap, matched minus unmatched: **+4.48 pts** (95% CI +3.99 to +4.98). Pass: **True**.

| count   |   pitches |   gap pts | top-pick share   |
|:--------|----------:|----------:|:-----------------|
| 0-0     |    123056 |      3.49 | 28%              |
| 0-1     |     60926 |      5.1  | 28%              |
| 0-2     |     31658 |      3.06 | 35%              |
| 1-0     |     45600 |      5.03 | 26%              |
| 1-1     |     47100 |      5.47 | 27%              |
| 1-2     |     45520 |      3.1  | 33%              |
| 2-0     |     15328 |      6.42 | 22%              |
| 2-1     |     24318 |      7.01 | 25%              |
| 2-2     |     38860 |      4.22 | 31%              |
| 3-0     |      4813 |      2.23 | 39%              |
| 3-1     |     10206 |     10.45 | 14%              |
| 3-2     |     23830 |      3.97 | 30%              |

## 2. Sequencer vs naive pick

On 418,453 pitches with a naive pick: sequencer gap +4.51 pts, naive gap +2.67 pts. Difference **+1.84 pts** (95% CI +1.24 to +2.39). Pass: **True**. The two picks agree in 36% of cells.

## 3. After-what effect

3,224 pitcher x type x side x count groups (124,022 pitches). Within group, CSW after a favored previous pitch minus after a less favored one: **+2.57 pts** (95% CI +2.04 to +3.11). Pass: **True**. For scale, the model predicted a gap of +1.39 pts between these halves.

