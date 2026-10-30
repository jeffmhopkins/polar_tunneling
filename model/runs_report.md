# Model comparison by runs (pre-registered in PREREG_runs.md, single run)

All pickers built from 2025 only, average hitter. RA = Perceptual, by runs; RB = Standard, by runs; RC = Stuff-only, by runs; A = Perceptual, by CSW; D = Best pitch for the count; E = Pitcher's habit; FS = FanGraphs Stuff+ (2025); FP = FanGraphs Pitching+ (2025); FB = PitchingBot (2025).

Test pitches: 470,321 from 265 pitchers (2026 regular season).

## Each picker: thrown the pick minus not, within count

Runs per 100 pitches (negative is better) and CSW pts (positive is better).

| picker | pitches | share thrown | runs per 100 (95% CI) | CSW pts |
|:--|--:|--:|:--|--:|
| RA. Perceptual, by runs | 470,321 | 32.0% | -0.60 (-0.72 to -0.49) | +1.15 |
| RB. Standard, by runs | 470,321 | 28.9% | -0.58 (-0.68 to -0.47) | +1.25 |
| RC. Stuff-only, by runs | 470,321 | 28.7% | -0.47 (-0.60 to -0.35) | +1.19 |
| A. Perceptual, by CSW | 470,321 | 28.7% | -0.52 (-0.63 to -0.40) | +4.19 |
| D. Best pitch for the count | 417,749 | 31.7% | -0.37 (-0.49 to -0.24) | +2.67 |
| E. Pitcher's habit | 470,321 | 44.7% | -0.53 (-0.62 to -0.42) | +0.52 |
| FS. FanGraphs Stuff+ (2025) | 425,045 | 26.6% | -0.39 (-0.52 to -0.24) | +2.84 |
| FP. FanGraphs Pitching+ (2025) | 425,045 | 27.6% | -0.36 (-0.50 to -0.23) | +1.22 |
| FB. PitchingBot (2025) | 425,045 | 28.5% | -0.29 (-0.43 to -0.15) | +1.21 |

## RA minus each other picker (same pitches)

| comparison | picks agree | runs per 100 (95% CI) | actual delta_run_exp per 100 | CSW pts |
|:--|--:|:--|--:|--:|
| RA vs RB. Standard, by runs **(primary)** | 78% | -0.03 (-0.12 to +0.06) | -0.07 | -0.10 |
| RA vs RC. Stuff-only, by runs | 36% | -0.13 (-0.29 to +0.01) | -0.30 | -0.04 |
| RA vs A. Perceptual, by CSW | 56% | -0.09 (-0.21 to +0.04) | -0.14 | -3.04 |
| RA vs D. Best pitch for the count | 31% | -0.21 (-0.36 to -0.07) | -0.23 | -1.58 |
| RA vs E. Pitcher's habit | 40% | -0.08 (-0.23 to +0.07) | +0.07 | +0.64 |
| RA vs FS. FanGraphs Stuff+ (2025) | 29% | -0.19 (-0.35 to +0.00) | -0.43 | -1.81 |
| RA vs FP. FanGraphs Pitching+ (2025) | 31% | -0.21 (-0.38 to -0.04) | -0.40 | -0.19 |
| RA vs FB. PitchingBot (2025) | 32% | -0.29 (-0.43 to -0.12) | -0.36 | -0.18 |

## Perceptual CSW pick (A) minus the public picks, on CSW

| comparison | picks agree | CSW pts (95% CI) |
|:--|--:|:--|
| A vs FS. FanGraphs Stuff+ (2025) | 37% | +1.32 (+0.67 to +2.02) |
| A vs FP. FanGraphs Pitching+ (2025) | 31% | +2.94 (+2.11 to +3.77) |
| A vs FB. PitchingBot (2025) | 33% | +2.95 (+2.12 to +3.78) |

Primary (perceptual features help pick by runs): **False**.

## Game 823491

Pitches where the pitcher threw each runs picker's choice: RA 82 of 226, RB 73 of 226, RC 63 of 226.

