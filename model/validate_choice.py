"""
How much of the sequencer's top-pick CSW gain survives controls for pitcher
choice? Pre-registered in PREREG_choice.md; same test pitches and 2025-only
sequencer as validate_sequencer.py.

Usage:
    python model/validate_choice.py      # writes model/choice_report.md
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import features as F
import validate_sequencer as V

OUT = Path(__file__).resolve().parent / "choice_report.md"
N_BOOT = 500
SEED = 7
PRIOR = 200
RAW_GAP = 4.48
BREAKING = {"SL", "ST", "CU", "KC", "SV", "CS"}

def pclass(t):
    t = t.astype(str)
    return np.where(t.isin(F.FASTBALLS), "fb", np.where(t.isin(BREAKING), "brk", "off"))

def batter_rates():
    d = F.build(2025)
    d["cls"] = pclass(d.pitch_type)
    league = d.csw.mean()
    league_cls = d.groupby("cls").csw.mean()
    o = d.groupby("batter").csw.agg(["sum", "size"])
    overall = (o["sum"] + PRIOR * league) / (o["size"] + PRIOR)
    c = d.groupby(["batter", "cls"]).csw.agg(["sum", "size"]).reset_index()
    c["rate"] = (c["sum"] + PRIOR * c.cls.map(league_cls)) / (c["size"] + PRIOR)
    return overall, c.set_index(["batter", "cls"]).rate, league, league_cls

def test_frame():
    grid = json.loads(V.GRID.read_text())["pitchers"]
    df = F.build(2026)
    df["prev"] = np.where(df["count"] == "0-0", "none", df.prev_pitch_type.astype(str))
    keep, top = [], []
    for i, r in enumerate(df[["pitcher", "stand", "count", "prev", "pitch_type"]].itertuples(index=False)):
        p = grid.get(str(r.pitcher))
        sd = p and p["sides"].get(r.stand)
        cell = sd and sd["grid"].get(r.count, {}).get(r.prev)
        if cell and r.pitch_type in cell:
            keep.append(i)
            top.append(max(cell, key=lambda t: cell[t][2]))
    t = df.iloc[keep].copy()
    t["top"] = top
    t["matched"] = (t.pitch_type.astype(str) == t.top).astype(float)
    return t

def covariates(t):
    overall, by_cls, league, league_cls = batter_rates()
    t["cls"] = pclass(t.pitch_type)
    t["bat_all"] = t.batter.map(overall)
    t["bat_new"] = t.bat_all.isna().astype(float)
    t["bat_all"] = t.bat_all.fillna(league)
    t["bat_cls"] = [by_cls.get((b, c), np.nan) for b, c in zip(t.batter, t.cls)]
    t["bat_cls"] = t.bat_cls.fillna(t.cls.map(league_cls))
    d = t.prev_description.fillna("")
    t["prev_res"] = np.select([d.isin(F.WHIFF), d == "called_strike", d.str.startswith("foul"),
                               d.isin({"ball", "blocked_ball"}), d == ""],
                              ["whiff", "called", "foul", "ball", "none"], "other")
    t["prev_cls"] = np.where(t.prev_pitch_type.isna(), "none", pclass(t.prev_pitch_type.fillna("")))
    t["inn"] = pd.cut(t.inning, [0, 3, 6, 99], labels=["1-3", "4-6", "7+"]).astype(str)
    for b in ["on_1b", "on_2b", "on_3b"]:
        t[b + "_occ"] = t[b].notna().astype(float)
    t["fe"] = t.pitcher.astype(str) + "|" + t.stand.astype(str) + "|" + t["count"].astype(str)
    dummies = pd.get_dummies(t[["outs_when_up", "inn", "prev_res", "prev_cls"]].astype(str), drop_first=True, dtype=float)
    X = pd.concat([t[["matched", "bat_all", "bat_cls", "bat_new", "platoon", "on_1b_occ", "on_2b_occ", "on_3b_occ"]].astype(float),
                   dummies], axis=1)
    return t, X

def within_ols(y, X, fe, w=None):
    """Coefficient on the first column of X after demeaning by fe (weights = bootstrap counts)."""
    w = np.ones(len(y)) if w is None else w
    df = pd.DataFrame(X).assign(_y=y, _w=w, _fe=fe.to_numpy())
    df = df[df._w > 0]
    cols = [c for c in df.columns if c not in ("_w", "_fe")]
    wm = df[cols].mul(df._w, axis=0).groupby(df._fe).transform("sum").div(df._w.groupby(df._fe).transform("sum"), axis=0)
    dm = df[cols] - wm
    sw = np.sqrt(df._w.to_numpy())[:, None]
    Xd = dm.drop(columns="_y").to_numpy() * sw
    yd = dm._y.to_numpy() * sw[:, 0]
    beta = np.linalg.lstsq(Xd, yd, rcond=None)[0]
    return beta[0] * 100

def propensity(t):
    feats = ["count", "prev", "prev_res", "outs_when_up", "inn", "on_1b_occ", "on_2b_occ", "on_3b_occ",
             "platoon", "bat_all", "bat_new", "pitcher", "stand"]
    # Batter-vs-class of the *top pick* (what the pitcher weighs), not of the pitch thrown.
    overall, by_cls, league, league_cls = batter_rates()
    tc = pclass(t.top)
    t["bat_top"] = [by_cls.get((b, c), np.nan) for b, c in zip(t.batter, tc)]
    t["bat_top"] = t.bat_top.fillna(pd.Series(tc, index=t.index).map(league_cls))
    feats.append("bat_top")
    X = t[feats].copy()
    for c in ["count", "prev", "prev_res", "inn", "pitcher", "stand"]:
        X[c] = X[c].astype(str).astype("category")
    e = np.zeros(len(t))
    folds = np.random.default_rng(SEED).integers(0, 5, len(t))
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=SEED)
    for k in range(5):
        m = lgb.train(params, lgb.Dataset(X[folds != k], t.matched[folds != k]), 300)
        e[folds == k] = m.predict(X[folds == k])
    return np.clip(e, 0.05, 0.95)

def ipw_gap(y, d, e, w):
    a = w * d / e
    b = w * (1 - d) / (1 - e)
    return ((a * y).sum() / a.sum() - (b * y).sum() / b.sum()) * 100

def main():
    t, X = covariates(test_frame())
    y = t.csw.astype(float).to_numpy()
    pitchers = t.pitcher.unique()
    rng = np.random.default_rng(SEED)
    draws = [pd.Series(rng.choice(pitchers, len(pitchers))).value_counts() for _ in range(N_BOOT)]
    weight = lambda d: t.pitcher.map(d).fillna(0).to_numpy()

    a = within_ols(y, X.to_numpy(), t.fe)
    a_ci = np.percentile([within_ols(y, X.to_numpy(), t.fe, weight(d)) for d in draws], [2.5, 97.5])
    a_only_fe = within_ols(y, X[["matched"]].to_numpy(), t.fe)

    e = propensity(t)
    d_ = t.matched.to_numpy()
    b = ipw_gap(y, d_, e, np.ones(len(t)))
    b_ci = np.percentile([ipw_gap(y, d_, e, weight(d)) for d in draws], [2.5, 97.5])
    t["e"] = e
    t["tercile"] = pd.qcut(t.e, 3, labels=["low (broke pattern)", "middle", "high (usual choice)"])
    rows = []
    for name, g in t.groupby("tercile", observed=True):
        gt = V.gap_table(g.assign(matched=g.matched.astype(bool), csw=g.csw.astype(int)), "matched")
        rows.append({"propensity tercile": name, "pitches": len(g), "matched share": f"{g.matched.mean():.0%}",
                     "within-count gap pts": round(V.within_count_gap(gt)[0], 2)})
    lines = ["# Tool vs pitcher choice (pre-registered, single run)", "",
             f"{len(t):,} 2026 test pitches, {len(pitchers)} pitchers. Raw within-count gap: {RAW_GAP:+.2f} pts.", "",
             "## A. Adjusted gap (primary)", "",
             f"With pitcher x side x count fixed effects only: {a_only_fe:+.2f} pts.", "",
             f"Adding batter, situation and previous-pitch controls: **{a:+.2f} pts** (95% CI {a_ci[0]:+.2f} to {a_ci[1]:+.2f}), "
             f"**{a / RAW_GAP:.0%}** of the raw gap. Survives: **{a > 0 and a_ci[0] > 0}**.", "",
             "## B. Propensity check", "",
             "Note: the propensity model uses the batter's 2025 rate against the class of the *top pick* in place of the "
             "class of the pitch thrown (which is what is being chosen, so it cannot predict the choice).", "",
             f"Propensity of throwing the top pick: median {np.median(e):.2f}, IQR {np.percentile(e, 25):.2f}-{np.percentile(e, 75):.2f}.", "",
             f"Inverse-propensity-weighted gap: **{b:+.2f} pts** (95% CI {b_ci[0]:+.2f} to {b_ci[1]:+.2f}), {b / RAW_GAP:.0%} of the raw gap.", "",
             pd.DataFrame(rows).to_markdown(index=False), ""]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
