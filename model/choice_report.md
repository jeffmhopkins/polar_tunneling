# Tool vs pitcher choice (pre-registered, single run)

471,215 2026 test pitches, 265 pitchers. Raw within-count gap: +4.48 pts.

## A. Adjusted gap (primary)

With pitcher x side x count fixed effects only: +4.25 pts.

Adding batter, situation and previous-pitch controls: **+3.56 pts** (95% CI +3.13 to +4.00), **80%** of the raw gap. Survives: **True**.

## B. Propensity check

Note: the propensity model uses the batter's 2025 rate against the class of the *top pick* in place of the class of the pitch thrown (which is what is being chosen, so it cannot predict the choice).

Propensity of throwing the top pick: median 0.27, IQR 0.18-0.37.

Inverse-propensity-weighted gap: **+4.16 pts** (95% CI +3.67 to +4.69), 93% of the raw gap.

| propensity tercile   |   pitches | matched share   |   within-count gap pts |
|:---------------------|----------:|:----------------|-----------------------:|
| low (broke pattern)  |    157072 | 13%             |                   5.33 |
| middle               |    157071 | 27%             |                   3.97 |
| high (usual choice)  |    157072 | 46%             |                   3.76 |

