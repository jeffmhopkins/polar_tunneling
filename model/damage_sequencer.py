"""
Contact quality in the sequencer, as pre-registered in PREREG_damage.md:
rank pitches by expected run value (balls, strikes, fouls and xwOBA on balls
in play) instead of CSW alone.

Usage:
    python model/damage_sequencer.py     # validation: 2025 model, 2026 test once
                                         # writes model/damage_report.md
    python model/damage_sequencer.py --production
        # 2025 + 2026 model and hitter data; writes model/sequencing.json and
        # model/batter_adj/*.json for the Pitch Sequencer page
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests

import hitter_sequencer as HS
import perceptual as M
import sequencing as Q
import speed_check as SC
import validate_sequencer as V

HERE = Path(__file__).resolve().parent
OUT = HERE / "damage_report.md"
N_ADJ = 10
N_BOOT = 1000
SEED = 7
COLS = HS.COLS_H
COLS0 = M.CONTROLS + SC.SPEED
FOUL = {"foul", "foul_bunt", "bunt_foul_tip"}
OUTCOMES = ["ball", "cs", "whiff", "foul"]

# ---- outcomes and run values (2025) ----

def add_outcomes(d):
    desc = d.description.astype(str)
    d["cs"] = desc == "called_strike"
    d["foul"] = desc.isin(FOUL)
    d["bip"] = desc == "hit_into_play"
    d["contact"] = d.swing & ~d.whiff
    d["outcome"] = np.select([~d.swing & d.cs, ~d.swing, d.whiff, d.foul], ["cs", "ball", "whiff", "foul"], "bip")
    return d

def run_values(d):
    """RV(count, outcome) for non-contact outcomes, and the ball-in-play fit on count + xwOBA."""
    nb = d[d.outcome != "bip"]
    tab = nb.groupby(["count", "outcome"]).delta_run_exp.mean().unstack("outcome")[OUTCOMES]
    b = d[(d.outcome == "bip") & d.xwoba.notna() & d.delta_run_exp.notna()]
    X = pd.get_dummies(b["count"]).astype(float)
    X["xwoba"] = b.xwoba.to_numpy()
    coef = np.linalg.lstsq(X.to_numpy(), b.delta_run_exp.to_numpy(), rcond=None)[0]
    icpt = pd.Series(coef[:-1], index=X.columns[:-1])
    bip_mean = d[d.outcome == "bip"].groupby("count").delta_run_exp.mean()
    return {"tab": tab, "icpt": icpt, "slope": coef[-1], "bip_mean": bip_mean}

def expected_rv(p, count, rv):
    """E[RV] from model probabilities p (dict of arrays) and the rows' counts."""
    t = rv["tab"].reindex(count)
    ball, cs, wh, fo = (t[c].to_numpy() for c in OUTCOMES)
    bip = rv["icpt"].reindex(count).to_numpy() + rv["slope"] * p["xw"]
    take = (1 - p["swing"]) * (p["cs"] * cs + (1 - p["cs"]) * ball)
    swing = p["swing"] * (p["whiff"] * wh + (1 - p["whiff"]) * (p["foul"] * fo + (1 - p["foul"]) * bip))
    return take + swing

def realized_rv(d, rv):
    out = rv["tab"].stack().reindex(pd.MultiIndex.from_arrays([d["count"], d.outcome])).to_numpy()
    bip = d.outcome.to_numpy() == "bip"
    fit = rv["icpt"].reindex(d["count"]).to_numpy() + rv["slope"] * d.xwoba.to_numpy()
    fill = rv["bip_mean"].reindex(d["count"]).to_numpy()
    return np.where(bip, np.where(np.isnan(fit), fill, fit), out)

# ---- models ----

def train(d, cols=COLS):
    sets = [("swing", d, "swing", "binary"), ("whiff", d[d.swing], "whiff", "binary"),
            ("foul", d[d.contact], "foul", "binary"), ("cs", d[~d.swing], "cs", "binary"),
            ("csw", d, "csw", "binary"), ("xw", d[d.bip & d.xwoba.notna()], "xwoba", "regression")]
    out = {}
    for name, data, y, obj in sets:
        out[name] = lgb.train(dict(M.S.PARAMS, objective=obj), lgb.Dataset(data[cols], data[y].astype(float)), M.S.ROUNDS)
        print("trained", name, flush=True)
    return out

def predict(models, rows):
    X = rows[COLS]
    return {k: m.predict(X) for k, m in models.items()}

# ---- grid and hitter adjustments ----

def build(models, rv, fit_on, d26, cats, hitters, avg, batters_for, type_median=None):
    """Like hitter_sequencer.build, with runs saved per 100 pitches (-100 E[RV]) next to CSW.
    Grid values: [swing%, whiff%, CSW%, deception, runs saved/100]; adjustments: (CSW pts, runs saved/100)."""
    base, tables = Q.usage_tables(d26)
    prev_desc = Q.typical_prev_desc(fit_on)
    medians = pd.Series({"bat_whiff_rate": avg.bat_whiff_rate, "bat_swing_rate": avg.bat_swing_rate})
    avg_rows = lambda n: pd.DataFrame(np.repeat(avg[HS.BAT_COLS].to_numpy(float)[None], n, 0), columns=HS.BAT_COLS)
    out, adj = {}, {}
    for i, (pid, side, sub, types, use) in enumerate(HS.pitcher_sides(d26)):
        cells = Q.build_cells(sub, pid, side, types, base, tables, prev_desc, medians)
        for c in Q.CATS:
            cells[c] = pd.Categorical(cells[c].astype(str), categories=cats[c])
        cells = HS.apply_hitter(cells, avg_rows(len(cells)))
        p = predict(models, cells)
        cells["sw_p"], cells["wh_p"], cells["csw_p"] = p["swing"], p["swing"] * p["whiff"], p["csw"]
        cells["rs"] = -expected_rv(p, cells["count"].astype(str), rv)
        cells["lift"] = 0.0
        if type_median is not None:
            cf = cells.copy()
            cf[Q.LOOK] = type_median.reindex(cells.pitch_type.astype(str)).to_numpy()
            cells["lift"] = cells.csw_p - models["csw"].predict(cf[COLS])
        g = cells.groupby(["_count", "_prev", "_type"], sort=False)[["sw_p", "wh_p", "csw_p", "lift", "rs"]].mean()
        grid = {}
        for (count, prev, t), r in g.iterrows():
            grid.setdefault(count, {}).setdefault(prev, {})[t] = [round(r.sw_p * 100, 1), round(r.wh_p * 100, 1),
                                                                 round(r.csw_p * 100, 1), round(r.lift * 100, 1),
                                                                 round(r.rs * 100, 2)]
        out.setdefault(pid, {})[side] = {"types": types, "usage": [round(use[t] / len(sub), 3) for t in types], "grid": grid}

        fb = sub.fb_type.iloc[0]
        prev = fb if fb in types else types[0]
        rows = cells[(cells._prev == prev) | (cells._count == "0-0")]
        rows = rows.groupby(["_count", "_type"], sort=False).head(N_ADJ).reset_index(drop=True)
        ids = [b for b in batters_for(pid, side) if b in hitters.index]
        if not ids:
            continue
        a0 = rows.groupby(["_count", "_type"], sort=False)[["csw_p", "rs"]].mean()
        for chunk in np.array_split(np.array(ids), max(1, len(ids) // 100)):
            rep = rows.loc[np.tile(np.arange(len(rows)), len(chunk))].reset_index(drop=True)
            who = np.repeat(chunk, len(rows))
            rep = HS.apply_hitter(rep, hitters.loc[who, HS.BAT_COLS].reset_index(drop=True))
            p = predict(models, rep)
            rep["csw_p"], rep["rs"], rep["_b"] = p["csw"], -expected_rv(p, rep["count"].astype(str), rv), who
            a = rep.groupby(["_b", "_count", "_type"], sort=False)[["csw_p", "rs"]].mean()
            a = a - a0.reindex(a.index.droplevel(0)).to_numpy()
            for (b, count, t), r in a.iterrows():
                adj.setdefault((pid, side), {}).setdefault(int(b), {}).setdefault(count, {})[t] = (r.csw_p * 100, r.rs * 100)
        if i % 25 == 0:
            print(i, pid, side, len(ids), "hitters", flush=True)
    return out, adj

def with_2025_class_rates(te, hit, tr, tau_avg):
    te = te.copy()
    cls = HS.pclass(te.pitch_type)
    pick = pd.Categorical(cls, categories=HS.CLASSES).codes
    avg = HS.average_hitter(hit, tr, tau_avg)
    for k in HS.BAT_CLS:
        v = np.stack([hit[f"{k}_{c}"].reindex(te.batter).to_numpy() for c in HS.CLASSES], axis=1)[np.arange(len(te)), pick]
        league = np.array([avg[f"{k}_{c}"] for c in HS.CLASSES])[pick]
        te[k] = np.where(np.isnan(v), league, v)
    return te

def validate():
    tau_avg = M.TAU0  # per-hitter timing did not pass (PREREG_hitter.md): bat-tracking timing
    s = HS.prep([2025], None, tau_avg)
    tr, te = add_outcomes(s[2025]), add_outcomes(s[2026])
    cats = {c: tr[c].cat.categories for c in Q.CATS}
    rv = run_values(tr)
    hit25 = HS.batter_table(tr, tr.groupby("batter").bat_tau.first(), tau_avg)
    models = train(tr)

    # Secondary: xwOBA on balls in play, with vs without the perceptual and hitter features.
    xw0 = lgb.train(dict(M.S.PARAMS, objective="regression"),
                    lgb.Dataset(tr.loc[tr.bip & tr.xwoba.notna(), COLS0], tr.loc[tr.bip & tr.xwoba.notna(), "xwoba"]), M.S.ROUNDS)
    tb = with_2025_class_rates(te[te.bip & te.xwoba.notna()], hit25, tr, tau_avg)
    r2 = lambda pred: 1 - ((tb.xwoba - pred) ** 2).sum() / ((tb.xwoba - tb.xwoba.mean()) ** 2).sum()
    r2_full, r2_base = r2(models["xw"].predict(tb[COLS])), r2(xw0.predict(tb[COLS0]))
    print("R2", r2_full, r2_base, flush=True)

    te["prev"] = np.where(te["count"] == "0-0", "none", te.prev_pitch_type.astype(str))
    eligible = hit25.index[hit25.n >= HS.MIN_2025]
    faced = te[te.batter.isin(eligible)].groupby(["pitcher", te.stand.astype(str)]).batter.unique()
    batters_for = lambda pid, side: list(faced.get((pid, side), []))
    avg = HS.average_hitter(hit25, tr, tau_avg)
    grid, adj = build(models, rv, tr, te, cats, hit25, avg, batters_for)

    te["real"] = realized_rv(te, rv)
    rows = []
    cols = ["pitcher", "batter", "stand", "count", "prev", "pitch_type", "csw", "real", "delta_run_exp", "xwoba", "bip"]
    for r in te[cols].itertuples(index=False):
        sd = grid.get(r.pitcher, {}).get(str(r.stand))
        cell = sd and sd["grid"].get(r.count, {}).get(r.prev)
        pt = str(r.pitch_type)
        if not cell or pt not in cell or r.batter not in eligible:
            continue
        a = adj.get((r.pitcher, str(r.stand)), {}).get(int(r.batter), {}).get(r.count)
        if a is None:
            continue
        csw_pick = max(cell, key=lambda t: cell[t][2] + a.get(t, (0, 0))[0])
        rv_pick = max(cell, key=lambda t: cell[t][4] + a.get(t, (0, 0))[1])
        rows.append((r.pitcher, r.count, pt, int(r.csw), r.real, r.delta_run_exp, r.xwoba, r.bip, csw_pick, rv_pick))
    t = pd.DataFrame(rows, columns=["pitcher", "count", "pitch_type", "csw_hit", "real", "dre", "xwoba", "bip", "csw_pick", "rv_pick"])
    differ = t.csw_pick != t.rv_pick
    t["m_rv"] = t.pitch_type == t.rv_pick
    p = t[differ & (t.m_rv | (t.pitch_type == t.csw_pick))].copy()
    rng = np.random.default_rng(SEED)
    V.N_BOOT = N_BOOT
    w = lambda draw, g: g.pitcher.map(draw).fillna(0)

    def gap(frame, value, scale=1.0):
        f = frame.assign(csw=frame[value].astype(float) * scale)
        g = V.gap_table(f, "m_rv")
        est, per_count, n = V.within_count_gap(g)
        ci = V.boot(lambda d: V.within_count_gap(g, w(d, g))[0], f.pitcher.unique(), rng)
        return est / 100, ci / 100, per_count / 100, n  # within_count_gap reports x100

    est, ci, per_count, n = gap(p, "real", 100)  # runs per 100 pitches
    ok = est < 0 and ci[1] < 0
    dre = gap(p[p.dre.notna()], "dre", 100)
    csw = gap(p, "csw_hit", 100)
    xw = gap(p[p.bip & p.xwoba.notna()], "xwoba")
    pc = pd.DataFrame({"pitches": n.astype(int), "runs per 100, RV pick minus CSW pick": per_count.round(2)})
    lines = ["# Contact quality in the sequencer (pre-registered, single run)", "",
             f"Test pitches: {len(t):,} from {t.pitcher.nunique()} pitchers, hitters with >= {HS.MIN_2025} pitches in 2025. "
             f"The run-value pick differs from the CSW pick on {differ.mean():.1%} of them.", "",
             "## Primary: run-value pick vs CSW pick where they differ", "",
             f"{len(p):,} pitches where the picks differ and the pitch thrown was one of the two "
             f"({p.m_rv.mean():.0%} were the run-value pick).", "",
             f"Within-count realized run value (xwOBA on contact), RV pick minus CSW pick: **{est:+.2f} runs per 100 pitches** "
             f"(95% CI {ci[0]:+.2f} to {ci[1]:+.2f}). Negative is fewer runs. Pass: **{ok}**.", "",
             pc.to_markdown(), "",
             "## Secondary", "",
             f"- Actual delta_run_exp: {dre[0]:+.2f} runs per 100 (95% CI {dre[1][0]:+.2f} to {dre[1][1]:+.2f}).",
             f"- CSW: {csw[0]:+.2f} pts (95% CI {csw[1][0]:+.2f} to {csw[1][1]:+.2f}).",
             f"- xwOBA on balls in play: {xw[0]:+.3f} (95% CI {xw[1][0]:+.3f} to {xw[1][1]:+.3f}).",
             f"- Picks differ on {differ.mean():.1%} of test pitches.",
             f"- 2026 R^2 of xwOBA on balls in play: {r2_full:.4f} with all features vs {r2_base:.4f} with CONTROLS + SPEED only.", ""]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

def teams(ids):
    """Current MLB team abbreviation for each person id."""
    abbr = {t["id"]: t["abbreviation"] for t in
            requests.get("https://statsapi.mlb.com/api/v1/teams", params={"sportId": 1}, timeout=60).json()["teams"]}
    out, ids = {}, list(ids)
    for i in range(0, len(ids), 100):
        r = requests.get("https://statsapi.mlb.com/api/v1/people",
                         params={"personIds": ",".join(map(str, ids[i:i + 100])), "hydrate": "currentTeam"}, timeout=60)
        r.raise_for_status()
        for p in r.json()["people"]:
            out[p["id"]] = abbr.get(p.get("currentTeam", {}).get("id"), "")
    return out

def production():
    tau_avg = M.TAU0
    s = HS.prep([2025, 2026], None, tau_avg)
    both = pd.concat([s[2025], s[2026]], ignore_index=True)
    for c in Q.CATS:
        both[c] = pd.Categorical(both[c].astype(str), categories=s[2025][c].cat.categories)
    both = add_outcomes(both)
    cats = {c: both[c].cat.categories for c in Q.CATS}
    d26 = s[2026]
    rv = run_values(both)
    hitters = HS.batter_table(both, both.groupby("batter").bat_tau.last(), tau_avg)
    page = hitters.index[hitters.n >= HS.PAGE_MIN_PITCHES]
    seen = d26[d26.batter.isin(page)]
    stand = seen.stand.astype(str).groupby(seen.batter).agg(lambda x: set(x))
    models = train(both)
    avg = HS.average_hitter(hitters, both, tau_avg)
    type_median = d26.groupby("pitch_type", observed=True)[Q.LOOK].median()
    type_median.index = type_median.index.astype(str)
    batters_for = lambda pid, side: [b for b in page if side in stand.get(b, set())]
    grid, adj = build(models, rv, both, d26, cats, hitters, avg, batters_for, type_median)
    who = Q.names(set(grid) | set(page))
    team = teams(set(grid) | set(page))
    out = {"counts": Q.COUNTS, "pitchers": {}, "shards": HS.N_SHARDS,
           "hitters": {str(int(b)): {"name": who.get(int(b), str(b)), "stand": sorted(stand.get(b, set())),
                                     "team": team.get(int(b), ""), "n26": int((seen.batter == b).sum())} for b in page}}
    for pid, sides in grid.items():
        p = d26[d26.pitcher == pid]
        out["pitchers"][str(int(pid))] = {"name": who.get(int(pid), str(pid)), "throws": str(p.p_throws.iloc[0]),
                                          "team": team.get(int(pid), ""), "n": int(len(p)), "sides": sides}
    (HERE / "sequencing.json").write_text(json.dumps(out, separators=(",", ":")))
    HS.ADJ_DIR.mkdir(exist_ok=True)
    for f in HS.ADJ_DIR.glob("adj_*.json"):
        f.unlink()
    shards = {}
    # Per hitter and count: CSW adjustments then runs-saved adjustments, in the pitcher's type order, x10 as integers.
    for (pid, side), by_b in adj.items():
        types = grid[pid][side]["types"]
        shards.setdefault(int(pid) % HS.N_SHARDS, {}).setdefault(str(int(pid)), {})[side] = {
            str(b): {c: [int(round(v[t][0] * 10)) for t in types] + [int(round(v[t][1] * 10)) for t in types]
                     for c, v in counts.items()}
            for b, counts in by_b.items()}
    for k, v in shards.items():
        (HS.ADJ_DIR / f"adj_{k:02d}.json").write_text(json.dumps(v, separators=(",", ":")))
    print("wrote sequencing.json and", len(shards), "adjustment shards for", len(page), "hitters")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", action="store_true")
    production() if parser.parse_args().production else validate()
