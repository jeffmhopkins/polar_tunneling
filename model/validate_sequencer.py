"""
Validate the sequencer against real 2026 pitches, as pre-registered in
PREREG_validation.md. Uses the 2025-only sequencer (sequencing_2025model.json).

Usage:
    python model/validate_sequencer.py   # writes model/validation_report.md
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import features as F

HERE = Path(__file__).resolve().parent
GRID = HERE / "sequencing_2025model.json"
OUT = HERE / "validation_report.md"
N_BOOT = 2000
SEED = 7
NAIVE_MIN = 30
GROUP_MIN = 10

def test_pitches(grid):
    df = F.build(2026)
    df["prev"] = np.where(df["count"] == "0-0", "none", df.prev_pitch_type.astype(str))
    rows = []
    for r in df[["pitcher", "stand", "count", "prev", "pitch_type", "csw"]].itertuples(index=False):
        p = grid.get(str(r.pitcher))
        sd = p and p["sides"].get(r.stand)
        cell = sd and sd["grid"].get(r.count, {}).get(r.prev)
        if not cell or r.pitch_type not in cell:
            continue
        top = max(cell, key=lambda t: cell[t][2])
        col = [sd["grid"][r.count][q][r.pitch_type][2] for q in sd["grid"][r.count]]
        rows.append((r.pitcher, r.stand, r.count, r.prev, r.pitch_type, int(r.csw), top,
                     cell[r.pitch_type][2] - np.mean(col)))
    return pd.DataFrame(rows, columns=["pitcher", "stand", "count", "prev", "pitch_type", "csw", "top", "seq_delta"])

def naive_picks():
    d = F.build(2025)
    by_cell = d.groupby(["pitcher", "stand", "count", "pitch_type"]).csw.agg(["mean", "size"]).reset_index()
    by_cell = by_cell[by_cell["size"] >= NAIVE_MIN].sort_values("mean").groupby(["pitcher", "stand", "count"]).tail(1)
    by_side = d.groupby(["pitcher", "stand", "pitch_type"]).csw.agg(["mean", "size"]).reset_index()
    by_side = by_side[by_side["size"] >= NAIVE_MIN].sort_values("mean").groupby(["pitcher", "stand"]).tail(1)
    return (by_cell.set_index(["pitcher", "stand", "count"]).pitch_type.to_dict(),
            by_side.set_index(["pitcher", "stand"]).pitch_type.to_dict())

def gap_table(t, flag):
    """Per pitcher x count sums for matched / unmatched CSW, so the bootstrap is cheap."""
    g = t.assign(m=t[flag]).groupby(["pitcher", "count", "m"]).csw.agg(["sum", "size"]).unstack("m", fill_value=0)
    g.columns = [f"{a}_{int(b)}" for a, b in g.columns]
    return g.reindex(columns=["sum_1", "size_1", "sum_0", "size_0"], fill_value=0).reset_index()

def within_count_gap(g, weights=None):
    w = g if weights is None else g.assign(**{c: g[c] * weights for c in ["sum_1", "size_1", "sum_0", "size_0"]})
    c = w.groupby("count")[["sum_1", "size_1", "sum_0", "size_0"]].sum()
    c = c[(c.size_1 > 0) & (c.size_0 > 0)]
    gap = c.sum_1 / c.size_1 - c.sum_0 / c.size_0
    n = c.size_1 + c.size_0
    return (gap * n).sum() / n.sum() * 100, gap * 100, n

def boot(fn, pitchers, rng):
    """fn(weights per pitcher as a Series) -> statistic; resample pitchers with replacement."""
    out = []
    for _ in range(N_BOOT):
        draw = pd.Series(rng.choice(pitchers, len(pitchers))).value_counts()
        out.append(fn(draw))
    return np.percentile(out, [2.5, 97.5])

def main():
    grid = json.loads(GRID.read_text())["pitchers"]
    t = test_pitches(grid)
    t["matched"] = t.pitch_type == t.top
    cell_pick, side_pick = naive_picks()
    t["naive"] = [cell_pick.get((p, s, c), side_pick.get((p, s))) for p, s, c in zip(t.pitcher, t.stand, t["count"])]
    t["naive_matched"] = t.pitch_type == t.naive
    pitchers = t.pitcher.unique()
    rng = np.random.default_rng(SEED)
    lines = ["# Sequencer validation on real 2026 pitches (pre-registered, single run)", "",
             f"Sequencer trained on 2025 only. Test pitches: {len(t):,} from {len(pitchers)} pitchers. "
             f"Share that matched the top pick: {t.matched.mean():.1%}.", ""]

    # 1. Top pick vs not.
    g1 = gap_table(t, "matched")
    est, per_count, n = within_count_gap(g1)
    w = lambda draw, g: g.pitcher.map(draw).fillna(0)
    ci = boot(lambda d: within_count_gap(g1, w(d, g1))[0], pitchers, rng)
    p1 = est > 0 and ci[0] > 0
    lines += ["## 1. Top pick vs not (primary)", "",
              f"Within-count CSW gap, matched minus unmatched: **{est:+.2f} pts** (95% CI {ci[0]:+.2f} to {ci[1]:+.2f}). "
              f"Pass: **{p1}**.", ""]
    pc = pd.DataFrame({"pitches": n.astype(int), "gap pts": per_count.round(2)})
    pc["top-pick share"] = t.groupby("count").matched.mean().reindex(pc.index).map("{:.0%}".format)
    lines += [pc.to_markdown(), ""]

    # 2. Sequencer gap minus naive gap, where both picks exist.
    both = t[t.naive.notna()]
    ga, gb = gap_table(both, "matched"), gap_table(both, "naive_matched")
    diff = lambda wa=None, wb=None: within_count_gap(ga, wa)[0] - within_count_gap(gb, wb)[0]
    est2 = diff()
    ci2 = boot(lambda d: diff(w(d, ga), w(d, gb)), both.pitcher.unique(), rng)
    p2 = est2 > 0 and ci2[0] > 0
    lines += ["## 2. Sequencer vs naive pick", "",
              f"On {len(both):,} pitches with a naive pick: sequencer gap {within_count_gap(ga)[0]:+.2f} pts, "
              f"naive gap {within_count_gap(gb)[0]:+.2f} pts. Difference **{est2:+.2f} pts** "
              f"(95% CI {ci2[0]:+.2f} to {ci2[1]:+.2f}). Pass: **{p2}**. "
              f"The two picks agree in {(both.top == both.naive).mean():.0%} of cells.", ""]

    # 3. After-what: same pitcher, type, side and count; favored previous pitch vs not.
    s = t[t.prev != "none"].copy()
    keys = ["pitcher", "pitch_type", "stand", "count"]
    s["above"] = s.seq_delta > s.groupby(keys).seq_delta.transform("median")
    gg = s.groupby(keys + ["above"]).csw.agg(["sum", "size"]).unstack("above", fill_value=0)
    gg.columns = [f"{a}_{int(b)}" for a, b in gg.columns]
    gg = gg[(gg.size_1 >= GROUP_MIN) & (gg.size_0 >= GROUP_MIN)].reset_index()
    def pooled(wt=None):
        x = gg if wt is None else gg.assign(**{c: gg[c] * wt for c in ["sum_1", "size_1", "sum_0", "size_0"]})
        n = x.size_1 + x.size_0
        keep = n > 0
        gap = x.sum_1[keep] / x.size_1[keep] - x.sum_0[keep] / x.size_0[keep]
        return (gap * n[keep]).sum() / n[keep].sum() * 100
    est3 = pooled()
    ci3 = boot(lambda d: pooled(w(d, gg)), gg.pitcher.unique(), rng)
    p3 = est3 > 0 and ci3[0] > 0
    spread = (s.groupby(keys + ["above"]).seq_delta.mean().unstack("above").dropna())
    lines += ["## 3. After-what effect", "",
              f"{len(gg):,} pitcher x type x side x count groups ({int((gg.size_1 + gg.size_0).sum()):,} pitches). "
              f"Within group, CSW after a favored previous pitch minus after a less favored one: **{est3:+.2f} pts** "
              f"(95% CI {ci3[0]:+.2f} to {ci3[1]:+.2f}). Pass: **{p3}**. "
              f"For scale, the model predicted a gap of {(spread[True] - spread[False]).mean():+.2f} pts between these halves.", ""]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
