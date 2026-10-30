"""
Out-of-season test of the arsenal tunnel signal: 2025 group features and a model
trained on all of 2025, scored on 2026 pitches from the same pitcher x pitch
type x side groups.

The question is whether a pitch type that tunneled well off its fastball in
2025 beats its stuff, location and fastball differentials in 2026. 2026 stuff
and location are in the controls, so only the 2025 tunnel features carry over.

Usage:
    python model/arsenal_holdout.py      # writes model/arsenal_holdout_report.md
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import arsenal as A
import features as F
import feasibility as S

OUT = Path(__file__).resolve().parent / "arsenal_holdout_report.md"
KEYS = ["pitcher", "stand", "pitch_type"]
N_NULL = 10

def with_groups(df, g):
    df = df.copy()
    for c in ["pitch_type", "stand"]:
        df[c] = df[c].astype(str)
    d = df.merge(g, on=KEYS)
    for c in ["pitch_type", "stand", "p_throws"]:
        d[c] = d[c].astype("category")
    return d

def main():
    d25, d26 = F.build(2025), F.build(2026)
    d25 = d25[d25.release_spin_rate_rpm.notna()]
    d26 = d26[d26.release_spin_rate_rpm.notna()]
    g25 = A.group_table(d25)
    g26 = A.group_table(d26)
    for g in (g25, g26):
        for c in ["pitch_type", "stand"]:
            g[c] = g[c].astype(str)
    # Tunnel features from 2025; fastball differentials from 2026 (current stuff).
    g = g25[KEYS + A.ARS].merge(g26[KEYS + A.DIFF], on=KEYS)
    tr, te = with_groups(d25, g25[KEYS + A.ARS + A.DIFF]), with_groups(d26, g)
    stab = g25[KEYS + A.ARS].merge(g26[KEYS + A.ARS], on=KEYS, suffixes=("_25", "_26"))
    lines = ["# Arsenal tunnel check, 2025 -> 2026", "",
             f"{len(g):,} pitcher x pitch type x side groups in both seasons; {len(te):,} 2026 pitches.", "",
             "Year-to-year correlation of the group tunnel features: "
             + ", ".join(f"{f} {stab[f + '_25'].corr(stab[f + '_26']):.2f}" for f in A.ARS), ""]
    rng = np.random.default_rng(S.SEED)
    rows = []
    for y in ["whiff", "swing", "csw"]:
        a, b = (tr[tr.swing], te[te.swing]) if y == "whiff" else (tr, te)
        base = A.fit(a, b, y, S.BASE + A.DIFF)
        full = A.fit(a, b, y, S.BASE + A.DIFF + A.ARS)
        nulls = []
        for _ in range(N_NULL):
            # Shuffle the 2025 tunnel features between 2026 groups of the same pitch type.
            gs = g.copy()
            for _, idx in gs.groupby("pitch_type").groups.items():
                gs.loc[idx, A.ARS] = gs.loc[rng.permutation(idx), A.ARS].to_numpy()
            bs = b.drop(columns=A.ARS).astype({"pitch_type": str, "stand": str}).merge(gs[KEYS + A.ARS], on=KEYS)
            bs = bs.astype({"pitch_type": a.pitch_type.dtype, "stand": a.stand.dtype})
            nulls.append(A.fit(a, bs, y, S.BASE + A.DIFF + A.ARS))
        ng = (base - np.array(nulls)) * 1000
        gain = (base - full) * 1000
        rows.append({"outcome": {"whiff": "whiff \\| swing", "swing": "swing", "csw": "CSW"}[y],
                     "2026 n": len(b), "gain x1000": round(gain, 2), "null mean": round(ng.mean(), 2),
                     "null max": round(ng.max(), 2), "share of nulls beaten": f"{(gain > ng).mean():.0%}"})
        print(rows[-1], flush=True)
    lines += ["Log-loss gain x 1000 on 2026 of +2025 arsenal tunnel over stuff + location + count + fastball differentials "
              "(model trained on 2025):", "", pd.DataFrame(rows).to_markdown(index=False), ""]

    sw25, sw26 = tr[tr.swing], te[te.swing].copy()
    m = lgb.train(S.PARAMS, lgb.Dataset(sw25[S.BASE + A.DIFF], sw25.whiff.astype(int)), S.ROUNDS)
    sw26["resid"] = sw26.whiff - m.predict(sw26[S.BASE + A.DIFF])
    r = sw26.groupby(KEYS, observed=True).agg(n=("resid", "size"), resid=("resid", "mean"))
    r = r[r.n >= 40].join(g.set_index(KEYS)[A.ARS])
    corr = pd.Series({f: r.resid.corr(r[f], method="spearman") for f in A.ARS}, name="Spearman r")
    lines += [f"Group level ({len(r)} groups with >= 40 swings in 2026): rank correlation of 2025 tunnel features "
              "with 2026 residual whiff/swing (tunneling predicts negative for a_dec_deg):", "",
              corr.round(3).to_frame().to_markdown(), ""]
    r["dec_q"] = pd.qcut(r.a_dec_deg, 4, labels=["closest", "2", "3", "farthest"])
    q = r.groupby("dec_q", observed=True).agg(groups=("resid", "size"), resid_whiff_pct=("resid", lambda x: x.mean() * 100))
    lines += ["Mean 2026 residual whiff/swing (pct points) by quartile of 2025 commit-point angle to the fastball:", "",
              q.round(2).to_markdown(), ""]
    # Bootstrap the closest-minus-farthest quartile gap.
    gaps = []
    for _ in range(2000):
        s = r.sample(len(r), replace=True, random_state=int(rng.integers(1 << 31)))
        qq = s.groupby("dec_q", observed=True).resid.mean()
        gaps.append((qq["closest"] - qq["farthest"]) * 100)
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    lines += [f"Closest minus farthest quartile: {np.mean(gaps):.2f} pct points (95% bootstrap interval {lo:.2f} to {hi:.2f}).", ""]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
