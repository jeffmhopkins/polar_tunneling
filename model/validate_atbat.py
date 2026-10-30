"""
Does following the sequencer over a whole at-bat lead to more strikeouts and
weaker contact? Pre-registered in PREREG_atbat.md; same 2025-only sequencer as
validate_sequencer.py and validate_choice.py.

Usage:
    python model/validate_atbat.py       # writes model/atbat_report.md
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import features as F
import validate_sequencer as V

OUT = Path(__file__).resolve().parent / "atbat_report.md"
N_BOOT = 500
SEED = 7
PRIOR_PA = 200
PRIOR_BIP = 100
STEP = 0.25  # effects are per +25 pts of adherence
CHOICE_CSW_GAIN = 3.56  # choice_report.md, adjusted per-pitch gap
STRIKEOUT = {"strikeout", "strikeout_double_play"}
FREE = {"walk", "hit_by_pitch"}
DROP = {"sac_bunt", "sac_bunt_double_play", "catcher_interf", "intent_walk", "truncated_pa"}
HARD = 95.0

def pa_results(year):
    """One row per PA from the raw pitch files: the result on its last pitch."""
    files = sorted((F.ROOT / "datamining" / "outputs").glob(f"pitches_{year}_*.csv"))
    cols = ["game_pk", "at_bat_number", "pitch_number", "game_type", "events", "xwoba", "woba_value",
            "launch_speed_mph", "batter"]
    d = pd.concat([pd.read_csv(f, usecols=cols) for f in files], ignore_index=True)
    d = d[(d.game_type == "R") & d.events.notna()].sort_values("pitch_number")
    d = d.groupby(["game_pk", "at_bat_number"]).tail(1)
    d = d[~d.events.isin(DROP)].copy()
    d["k"] = d.events.isin(STRIKEOUT).astype(float)
    d["free"] = d.events.isin(FREE).astype(float)
    d["bip"] = ((d.k == 0) & (d.free == 0)).astype(float)
    d["hard"] = np.where(d.bip == 1, (d.launch_speed_mph >= HARD).astype(float), np.nan)
    d.loc[d.launch_speed_mph.isna() & (d.bip == 1), "hard"] = np.nan
    d["cxwoba"] = np.where(d.bip == 1, d.xwoba, np.nan)
    return d[["game_pk", "at_bat_number", "batter", "events", "k", "free", "bip", "hard", "xwoba", "cxwoba", "woba_value"]]

def batter_rates():
    """2025 batter K, walk + HBP, xwOBA per PA and contact xwOBA, shrunk to the league rate."""
    d = pa_results(2025)
    out = {}
    for col, prior in [("k", PRIOR_PA), ("free", PRIOR_PA), ("xwoba", PRIOR_PA), ("cxwoba", PRIOR_BIP)]:
        x = d.dropna(subset=[col])
        league = x[col].mean()
        g = x.groupby("batter")[col].agg(["sum", "size"])
        out[col] = ((g["sum"] + prior * league) / (g["size"] + prior), league)
    return out

def scored_pitches():
    grid = json.loads(V.GRID.read_text())["pitchers"]
    df = F.build(2026)
    df["prev"] = np.where(df["count"] == "0-0", "none", df.prev_pitch_type.astype(str))
    top = []
    for r in df[["pitcher", "stand", "count", "prev"]].itertuples(index=False):
        p = grid.get(str(r.pitcher))
        sd = p and p["sides"].get(r.stand)
        cell = sd and sd["grid"].get(r.count, {}).get(r.prev)
        top.append(max(cell, key=lambda t: cell[t][2]) if cell else None)
    df["top"] = top
    df["in_grid"] = [bool((p := grid.get(str(a)))) and b in p["sides"] for a, b in zip(df.pitcher, df.stand)]
    df["scored"] = df.top.notna()
    df["matched"] = np.where(df.scored, (df.pitch_type.astype(str) == df.top).astype(float), np.nan)
    s = df[df.scored]
    base = s.groupby(["pitcher", "stand", "count"]).matched.transform("mean")
    df.loc[df.scored, "resid"] = s.matched - base
    return df

def pa_table(df):
    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"])
    keys = ["game_pk", "at_bat_number"]
    g = df.groupby(keys)
    first = g.head(1).set_index(keys)
    pa = pd.DataFrame({
        "pitcher": first.pitcher, "stand": first.stand, "in_grid": first.in_grid,
        "outs": first.outs_when_up, "inning": first.inning, "platoon": first.platoon,
        "tto": first.n_priorpa_thisgame.fillna(0),
        "on1": first.on_1b.notna().astype(float), "on2": first.on_2b.notna().astype(float),
        "on3": first.on_3b.notna().astype(float),
        "adh": g.resid.mean(), "n_scored": g.scored.sum(), "share": g.matched.mean(),
        "n_csw": g.csw.sum().astype(float), "n_pitches": g.size(),
    })
    fp = df[(df["count"] == "0-0") & df.scored].set_index(keys)
    pa["first"] = fp.resid.reindex(pa.index)
    return pa.reset_index()

def design(pa, rates):
    for col in ["k", "free", "xwoba", "cxwoba"]:
        r, league = rates[col]
        pa["bat_" + col] = pa.batter.map(r)
    pa["bat_new"] = pa.bat_k.isna().astype(float)
    for col in ["k", "free", "xwoba", "cxwoba"]:
        pa["bat_" + col] = pa["bat_" + col].fillna(rates[col][1])
    pa["inn"] = pd.cut(pa.inning, [0, 3, 6, 99], labels=["1-3", "4-6", "7+"]).astype(str)
    pa["tto_c"] = np.minimum(pa.tto, 3).astype(int).astype(str)
    dummies = pd.get_dummies(pa[["outs", "inn", "tto_c"]].astype(str), drop_first=True, dtype=float)
    ctrl = pd.concat([pa[["bat_k", "bat_free", "bat_xwoba", "bat_cxwoba", "bat_new", "platoon", "on1", "on2", "on3"]]
                      .astype(float), dummies], axis=1)
    pa["fe"] = pa.pitcher.astype(str) + "|" + pa.stand.astype(str)
    return pa, ctrl

def within(y, X, fe_codes, w):
    """Weighted within-group OLS; returns all coefficients. w = bootstrap multiplicities."""
    keep = w > 0
    y, X, fe, w = y[keep], X[keep], fe_codes[keep], w[keep]
    _, fe = np.unique(fe, return_inverse=True)
    sw = np.bincount(fe, w)
    Z = np.column_stack([y, X])
    means = np.stack([np.bincount(fe, w * Z[:, j]) for j in range(Z.shape[1])], axis=1) / sw[:, None]
    Z = (Z - means[fe]) * np.sqrt(w)[:, None]
    return np.linalg.lstsq(Z[:, 1:], Z[:, 0], rcond=None)[0]

class Fit:
    """Coefficient on the first regressor, point estimate and pitcher-bootstrap CI."""
    def __init__(self, frame, ycol, xcols, ctrl, fe, draws):
        ok = frame[ycol].notna().to_numpy() & frame[xcols].notna().all(axis=1).to_numpy()
        self.n = int(ok.sum())
        y = frame[ycol].to_numpy(float)[ok]
        X = np.column_stack([frame[xcols].to_numpy(float), ctrl.to_numpy(float)])[ok]
        codes = pd.factorize(frame[fe])[0][ok]
        pitch = frame.pitcher.to_numpy()[ok]
        self.coef = within(y, X, codes, np.ones(len(y)))
        boots = [within(y, X, codes, pd.Series(pitch).map(d).fillna(0).to_numpy()) for d in draws]
        self.boot = np.array(boots)
        self.est = self.coef[0]
        self.ci = np.percentile(self.boot[:, 0], [2.5, 97.5])

def fmt(est, ci, scale, unit):
    if unit == "pts":
        return f"{est * scale * 100:+.2f} pts (95% CI {ci[0] * scale * 100:+.2f} to {ci[1] * scale * 100:+.2f})"
    return f"{est * scale:+.4f} (95% CI {ci[0] * scale:+.4f} to {ci[1] * scale:+.4f})"

OUTCOMES = [("k", "K rate", "pts"), ("xwoba", "xwOBA per PA", "woba"), ("cxwoba", "contact xwOBA", "woba"),
            ("free", "walk + HBP rate", "pts"), ("hard", "hard-hit rate on contact", "pts"),
            ("woba_value", "actual wOBA", "woba")]

def main():
    rates = batter_rates()
    pitches = scored_pitches()
    pa = pa_table(pitches)
    res = pa_results(2026)
    pa = pa.merge(res, on=["game_pk", "at_bat_number"], how="inner")
    pa = pa[pa.in_grid & (pa.n_scored >= 1)].reset_index(drop=True)
    pa, ctrl = design(pa, rates)
    pitchers = pa.pitcher.unique()
    rng = np.random.default_rng(SEED)
    draws = [pd.Series(rng.choice(pitchers, len(pitchers))).value_counts() for _ in range(N_BOOT)]

    lines = ["# Whole at-bat test (pre-registered, single run)", "",
             f"2026 plate appearances: {len(pa):,} from {len(pitchers)} pitchers, sequencer trained on 2025 only. "
             f"Mean share of scored pitches that matched the top pick: {pa.share.mean():.1%}. "
             f"SD of adherence (match share minus the pitcher's usual rate in those counts): {pa.adh.std():.3f}.", "",
             "Effects are per +25 pts of adherence, with pitcher x side fixed effects, batter 2025 rates and "
             "situation controls. 95% CIs from a pitcher bootstrap (500 draws).", ""]

    fits = {y: Fit(pa, y, ["adh"], ctrl, "fe", draws) for y, _, _ in OUTCOMES}
    p1 = fits["k"].est > 0 and fits["k"].ci[0] > 0
    p2 = fits["xwoba"].est < 0 and fits["xwoba"].ci[1] < 0
    p3 = fits["cxwoba"].est < 0 and fits["cxwoba"].ci[1] < 0
    not_harder = (not p3) and fits["cxwoba"].ci[1] * STEP < 0.005
    lines += ["## Primary", "",
              f"1. **More strikeouts:** K rate {fmt(fits['k'].est, fits['k'].ci, STEP, 'pts')}. Pass: **{p1}**.",
              f"2. **Fewer runs:** xwOBA per PA {fmt(fits['xwoba'].est, fits['xwoba'].ci, STEP, 'woba')}. Pass: **{p2}**.",
              f"3. **Weaker contact:** contact xwOBA {fmt(fits['cxwoba'].est, fits['cxwoba'].ci, STEP, 'woba')} "
              f"on {fits['cxwoba'].n:,} balls in play. Pass: **{p3}**."
              + (" Not a pass, but the upper end is below +0.005, so the picks are not hit harder." if not_harder else ""),
              ""]

    rows = [{"outcome": name, "PAs": fits[y].n, "mean": f"{pa[y].mean():.3f}",
             "per +25 pts adherence": fmt(fits[y].est, fits[y].ci, STEP, unit)} for y, name, unit in OUTCOMES]
    lines += ["## All outcomes", "", pd.DataFrame(rows).to_markdown(index=False), ""]

    # Secondary: first pitch only.
    fp = pa[pa["first"].notna()].reset_index(drop=True)
    cf = ctrl.loc[pa["first"].notna().to_numpy()].reset_index(drop=True)
    rows = []
    for y, name, unit in OUTCOMES:
        f = Fit(fp, y, ["first"], cf, "fe", draws)
        rows.append({"outcome": name, "PAs": f.n, "0-0 pitch was the top pick": fmt(f.est, f.ci, 1, unit)})
    lines += ["## Secondary: first pitch only", "",
              f"{len(fp):,} PAs with a scored 0-0 pitch ({(fp['first'] > 0).mean():.0%} matched). "
              "Exposure: the 0-0 pitch matched the top pick (minus the pitcher x side 0-0 match rate). Effect of matching "
              "vs not.", "", pd.DataFrame(rows).to_markdown(index=False), ""]

    # Secondary: pitch-level contact check.
    keys = ["game_pk", "at_bat_number"]
    bip = pitches[pitches.scored & (pitches.description == "hit_into_play")].copy()
    bip = bip.merge(pa[keys + ["bat_k", "bat_free", "bat_xwoba", "bat_cxwoba", "bat_new", "on1", "on2", "on3"]],
                    on=keys, how="inner")
    bip["hard"] = np.where(bip.launch_speed_mph.notna(), (bip.launch_speed_mph >= HARD).astype(float), np.nan)
    bip["inn"] = pd.cut(bip.inning, [0, 3, 6, 99], labels=["1-3", "4-6", "7+"]).astype(str)
    bip["fe"] = bip.pitcher.astype(str) + "|" + bip.stand.astype(str) + "|" + bip["count"].astype(str)
    bip = bip.reset_index(drop=True)
    bctrl = pd.concat([bip[["bat_k", "bat_free", "bat_xwoba", "bat_cxwoba", "bat_new", "platoon", "on1", "on2", "on3"]]
                       .astype(float),
                       pd.get_dummies(bip[["outs_when_up", "inn"]].astype(str), drop_first=True, dtype=float)], axis=1)
    fx = Fit(bip, "xwoba", ["matched"], bctrl, "fe", draws)
    fh = Fit(bip, "hard", ["matched"], bctrl, "fe", draws)
    lines += ["## Secondary: pitch-level contact check", "",
              f"{fx.n:,} balls in play on scored pitches ({bip.matched.mean():.0%} were the top pick), pitcher x side x "
              "count fixed effects. Top pick minus other pitches:", "",
              f"- contact xwOBA {fmt(fx.est, fx.ci, 1, 'woba')}",
              f"- hard-hit rate {fmt(fh.est, fh.ci, 1, 'pts')}", ""]

    # Secondary: implied vs observed K.
    fa = Fit(pa, "n_csw", ["adh"], ctrl, "fe", draws)
    fb = Fit(pa, "k", ["n_csw"], ctrl, "fe", draws)
    implied = fa.est * fb.est
    implied_ci = np.percentile(fa.boot[:, 0] * fb.boot[:, 0], [2.5, 97.5])
    lines += ["## Secondary: implied vs observed K", "",
              f"Per +25 pts adherence, a PA gets {fa.est * STEP:+.3f} more CSW pitches; each extra CSW pitch goes with "
              f"{fb.est * 100:+.2f} pts of K rate. A CSW-only chain implies {fmt(implied, implied_ci, STEP, 'pts')} of K "
              f"rate, against the observed {fmt(fits['k'].est, fits['k'].ci, STEP, 'pts')}. "
              f"(The per-pitch adjusted CSW gain from `choice_report.md` is {CHOICE_CSW_GAIN:+.2f} pts.)", ""]

    # Quartile table.
    pa["q"] = pd.qcut(pa.adh.rank(method="first"), 4, labels=["Q1 (least)", "Q2", "Q3", "Q4 (most)"])
    qd = pd.get_dummies(pa.q, drop_first=True, dtype=float)
    qcols = list(qd.columns)
    pq = pd.concat([pa, qd], axis=1)
    rows = []
    for y, name, unit in OUTCOMES[:3]:
        ok = pq[y].notna().to_numpy()
        X = np.column_stack([pq[qcols].to_numpy(float), ctrl.to_numpy(float)])[ok]
        c = within(pq[y].to_numpy(float)[ok], X, pd.factorize(pq.fe)[0][ok], np.ones(ok.sum()))[:3]
        sc = 100 if unit == "pts" else 1
        rows.append({"outcome": name, **{q: f"{v * sc:+.3f}" if unit != "pts" else f"{v * sc:+.2f}"
                                         for q, v in zip(qcols, c)}})
    share = pa.groupby("q", observed=True).share.mean().map("{:.0%}".format)
    lines += ["## By adherence quartile", "",
              "Adjusted difference from the least adherent quartile (same controls and fixed effects). "
              f"Mean top-pick share by quartile: {', '.join(f'{k} {v}' for k, v in share.items())}.", "",
              pd.DataFrame(rows).to_markdown(index=False), ""]

    # Exploratory, added after the run: the quartiles mix short and long PAs.
    # Short PAs have extreme adherence (one pitch is either matched or not) and
    # cannot end in a K, so compare quartiles within PAs of the same length.
    pa["len_q"] = pa.groupby("n_scored").adh.rank(method="first", pct=True)
    pa["q2"] = pd.cut(pa.len_q, [0, 0.25, 0.5, 0.75, 1], labels=["Q1 (least)", "Q2", "Q3", "Q4 (most)"])
    pa["fe_len"] = pa.fe + "|" + np.minimum(pa.n_pitches, 8).astype(str)
    qd = pd.get_dummies(pa.q2, drop_first=True, dtype=float)
    pq = pd.concat([pa, qd], axis=1)
    rows = []
    for y, name, unit in OUTCOMES[:3]:
        ok = pq[y].notna().to_numpy()
        X = np.column_stack([pq[qcols].to_numpy(float), ctrl.to_numpy(float)])[ok]
        c = within(pq[y].to_numpy(float)[ok], X, pd.factorize(pq.fe_len)[0][ok], np.ones(ok.sum()))[:3]
        sc = 100 if unit == "pts" else 1
        rows.append({"outcome": name, **{q: f"{v * sc:+.3f}" if unit != "pts" else f"{v * sc:+.2f}"
                                         for q, v in zip(qcols, c)}})
    fl = {y: Fit(pa, y, ["adh"], ctrl, "fe_len", draws) for y, _, _ in OUTCOMES[:3]}
    lines += ["## Exploratory (added after the run, not pre-registered): same-length at-bats", "",
              "Short PAs have extreme adherence (a one-pitch PA either matched or didn't) and cannot end in a "
              "strikeout, so the quartile table above mostly sorts PAs by length. Here quartiles are formed within "
              "PAs with the same number of scored pitches, and the fixed effects are pitcher x side x number of "
              "pitches (capped at 8). PA length is itself partly a result of the pitches, so this is a check on the "
              "shape of the effect, not a replacement for the primary test.", "",
              pd.DataFrame(rows).to_markdown(index=False), "",
              "Linear effect per +25 pts adherence within same-length PAs: "
              + "; ".join(f"{name} {fmt(fl[y].est, fl[y].ci, STEP, unit)}" for y, name, unit in OUTCOMES[:3]) + ".", ""]

    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
