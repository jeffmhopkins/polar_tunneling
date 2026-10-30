# Whole at-bat test (pre-registered, single run)

2026 plate appearances: 124,663 from 265 pitchers, sequencer trained on 2025 only. Mean share of scored pitches that matched the top pick: 27.6%. SD of adherence (match share minus the pitcher's usual rate in those counts): 0.254.

Effects are per +25 pts of adherence, with pitcher x side fixed effects, batter 2025 rates and situation controls. 95% CIs from a pitcher bootstrap (500 draws).

## Primary

1. **More strikeouts:** K rate +1.15 pts (95% CI +0.94 to +1.34). Pass: **True**.
2. **Fewer runs:** xwOBA per PA -0.0043 (95% CI -0.0063 to -0.0024). Pass: **True**.
3. **Weaker contact:** contact xwOBA +0.0003 (95% CI -0.0021 to +0.0026) on 84,677 balls in play. Pass: **False**. Not a pass, but the upper end is below +0.005, so the picks are not hit harder.

## All outcomes

| outcome                  |    PAs |   mean | per +25 pts adherence               |
|:-------------------------|-------:|-------:|:------------------------------------|
| K rate                   | 124663 |  0.226 | +1.15 pts (95% CI +0.94 to +1.34)   |
| xwOBA per PA             | 124349 |  0.312 | -0.0043 (95% CI -0.0063 to -0.0024) |
| contact xwOBA            |  84677 |  0.362 | +0.0003 (95% CI -0.0021 to +0.0026) |
| walk + HBP rate          | 124663 |  0.093 | -0.05 pts (95% CI -0.19 to +0.09)   |
| hard-hit rate on contact |  84677 |  0.389 | +0.15 pts (95% CI -0.19 to +0.47)   |
| actual wOBA              | 124663 |  0.317 | -0.0064 (95% CI -0.0091 to -0.0037) |

## Secondary: first pitch only

124,550 PAs with a scored 0-0 pitch (28% matched). Exposure: the 0-0 pitch matched the top pick (minus the pitcher x side 0-0 match rate). Effect of matching vs not.

| outcome                  |    PAs | 0-0 pitch was the top pick          |
|:-------------------------|-------:|:------------------------------------|
| K rate                   | 124550 | +0.38 pts (95% CI -0.17 to +0.85)   |
| xwOBA per PA             | 124236 | -0.0021 (95% CI -0.0070 to +0.0028) |
| contact xwOBA            |  84600 | +0.0008 (95% CI -0.0056 to +0.0076) |
| walk + HBP rate          | 124550 | -0.32 pts (95% CI -0.69 to +0.08)   |
| hard-hit rate on contact |  84600 | +0.58 pts (95% CI -0.20 to +1.46)   |
| actual wOBA              | 124550 | -0.0043 (95% CI -0.0106 to +0.0023) |

## Secondary: pitch-level contact check

83,080 balls in play on scored pitches (26% were the top pick), pitcher x side x count fixed effects. Top pick minus other pitches:

- contact xwOBA -0.0016 (95% CI -0.0086 to +0.0052)
- hard-hit rate -0.19 pts (95% CI -1.27 to +1.04)

## Secondary: implied vs observed K

Per +25 pts adherence, a PA gets +0.041 more CSW pitches; each extra CSW pitch goes with +29.50 pts of K rate. A CSW-only chain implies +1.21 pts (95% CI +0.97 to +1.43) of K rate, against the observed +1.15 pts (95% CI +0.94 to +1.34). (The per-pitch adjusted CSW gain from `choice_report.md` is +3.56 pts.)

## By adherence quartile

Adjusted difference from the least adherent quartile (same controls and fixed effects). Mean top-pick share by quartile: Q1 (least) 3%, Q2 10%, Q3 33%, Q4 (most) 64%.

| outcome       |     Q2 |     Q3 |   Q4 (most) |
|:--------------|-------:|-------:|------------:|
| K rate        |  6.05  | 13.53  |       5.89  |
| xwOBA per PA  | -0.003 | -0.034 |      -0.017 |
| contact xwOBA |  0.006 | -0.007 |       0     |

## Exploratory (added after the run, not pre-registered): same-length at-bats

Short PAs have extreme adherence (a one-pitch PA either matched or didn't) and cannot end in a strikeout, so the quartile table above mostly sorts PAs by length. Here quartiles are formed within PAs with the same number of scored pitches, and the fixed effects are pitcher x side x number of pitches (capped at 8). PA length is itself partly a result of the pitches, so this is a check on the shape of the effect, not a replacement for the primary test.

| outcome       |     Q2 |     Q3 |   Q4 (most) |
|:--------------|-------:|-------:|------------:|
| K rate        | -0.81  |  1.85  |       2.4   |
| xwOBA per PA  |  0.009 | -0.01  |      -0.007 |
| contact xwOBA |  0.006 | -0.005 |       0.003 |

Linear effect per +25 pts adherence within same-length PAs: K rate +0.86 pts (95% CI +0.68 to +1.04); xwOBA per PA -0.0031 (95% CI -0.0050 to -0.0012); contact xwOBA +0.0007 (95% CI -0.0017 to +0.0030).

