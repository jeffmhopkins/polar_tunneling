# Model comparison and game report (pre-registered in PREREG_game.md)

All models trained on 2025 only. Pickers: A = Perceptual sequencer; B = Standard pitch model; C = Stuff-only (Stuff+-style); D = Best pitch for the count; E = Pitcher's habit.

## Season comparison: matched minus unmatched within-count CSW, 2026

Test pitches: 470,321 from 265 pitchers (2026 regular season).

| picker | pitches | share thrown | CSW when thrown | gap pts (95% CI) |
|:--|--:|--:|--:|:--|
| A. Perceptual sequencer | 470,321 | 28.5% | 31.1% | +4.49 (+4.01 to +4.97) |
| B. Standard pitch model | 470,321 | 25.8% | 31.4% | +4.33 (+3.84 to +4.83) |
| C. Stuff-only (Stuff+-style) | 470,321 | 23.5% | 31.5% | +4.03 (+3.36 to +4.74) |
| D. Best pitch for the count | 417,749 | 31.7% | 29.8% | +2.67 (+2.17 to +3.19) |
| E. Pitcher's habit | 470,321 | 44.7% | 28.9% | +0.52 (+0.07 to +0.97) |

## A minus each other picker (same pitches)

| comparison | pitches | picks agree | A gap minus other gap (95% CI) |
|:--|--:|--:|:--|
| A vs B. Standard pitch model **(primary)** | 470,321 | 82% | +0.16 (-0.06 to +0.38) |
| A vs C. Stuff-only (Stuff+-style) | 470,321 | 44% | +0.46 (-0.12 to +1.04) |
| A vs D. Best pitch for the count | 417,749 | 36% | +1.85 (+1.26 to +2.41) |
| A vs E. Pitcher's habit | 470,321 | 31% | +3.97 (+3.35 to +4.60) |

Primary (A beats the standard model): **False**.

## Predicting the 2026 pitches actually thrown (log-loss, lower is better)

710,486 pitches. Gain x1000 vs the stuff-only model in brackets.

| model | swing | whiff (per pitch) | CSW |
|:--|--:|--:|--:|
| A. Perceptual sequencer | 0.4448 (+242.0) | 0.3138 (+41.9) | 0.4727 (+118.6) |
| B. Standard pitch model | 0.4505 (+236.4) | 0.3155 (+40.1) | 0.4748 (+116.5) |
| C. Stuff-only (Stuff+-style) | 0.6869 (+0.0) | 0.3557 (+0.0) | 0.5913 (+0.0) |

## Game: Regular Season, 2026-09-25: Baltimore Orioles 10 @ New York Yankees 2

323 pitches, 226 with picks (pitcher has a grid for that side and count). Actual CSW on those: 27.9%; model A's expected CSW of the pitches thrown: 27.3%.

| picker | matched / scored | CSW when matched | CSW otherwise | A's expected CSW if followed |
|:--|--:|--:|--:|--:|
| A. Perceptual sequencer | 59 / 226 | 29% | 28% | 32.7% |
| B. Standard pitch model | 63 / 226 | 32% | 26% | 32.4% |
| C. Stuff-only (Stuff+-style) | 42 / 226 | 33% | 27% | 31.0% |
| D. Best pitch for the count | 63 / 204 | 35% | 25% | 28.7% |
| E. Pitcher's habit | 113 / 226 | 27% | 28% | 27.6% |

CSW log-loss on the 323 game pitches with full features: A 0.5356, B 0.5433, C 0.6019.

