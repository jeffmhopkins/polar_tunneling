"""
Robustness of the feasibility lift test (feasibility.py section 3):
commit point 125/175/225 ms, different-type pairs only, and 10 null draws.
The comparison is +prev +tunnel against +prev, so the tunnel features are
judged on what they add beyond the previous pitch's type, location and result.

Usage:
    python model/robustness.py            # writes model/robustness_report.md
"""

from pathlib import Path

import numpy as np
import pandas as pd

import features as F
import feasibility as S

OUT = Path(__file__).resolve().parent / "robustness_report.md"
N_NULL = 10

def main():
    rows = []
    rng = np.random.default_rng(S.SEED)
    for tau in (0.125, 0.175, 0.225):
        F.TAU = tau
        df = S.prep()
        for subset, d in [("all pairs", df), ("different-type pairs", df[~df.same_type])]:
            for y, data in [("whiff | swing", d[d.swing]), ("swing", d)]:
                col = "whiff" if y.startswith("whiff") else "swing"
                tr, te = data[data.month <= 7], data[data.month >= 8]
                _, prev = S.fit(tr, te, col, S.BASE + S.PREV)
                _, full = S.fit(tr, te, col, S.BASE + S.PREV + S.TUNNEL)
                n_null = N_NULL if tau == 0.175 and subset == "all pairs" else 3
                nulls = []
                for _ in range(n_null):
                    s = S.shuffled(data, rng)
                    nulls.append(S.fit(s[s.month <= 7], s[s.month >= 8], col, S.BASE + S.PREV + S.TUNNEL)[1])
                gain = (prev - full) * 1000
                ng = (prev - np.array(nulls)) * 1000
                rows.append({"commit ms": int(tau * 1000), "pairs": subset, "outcome": y,
                             "gain x1000": round(gain, 2), "null mean": round(ng.mean(), 2),
                             "null max": round(ng.max(), 2), "null draws": n_null,
                             "share of nulls beaten": f"{(gain > ng).mean():.0%}"})
                print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    OUT.write_text("# Lift robustness (log-loss gain x 1000 of +tunnel over +prev, test Aug-Oct)\n\n"
                   + t.to_markdown(index=False) + "\n")

if __name__ == "__main__":
    main()
