"""
Batter-specific sequencer, as pre-registered in PREREG_hitter.md (Part 2).

Model H = the sequencer model plus the hitter's whiff, swing and chase rates
against the class of the pitch (fastball / breaking / offspeed). The
side-only score is the usual sequencer grid for an average hitter; the
batter adjustment is how much the predicted CSW of each pitch type in each
count changes when the actual hitter is at the plate.

Usage:
    python model/hitter_sequencer.py     # validation: 2025 model, 2026 test once
                                         # writes model/hitter_sequencer_report.md
    python model/hitter_sequencer.py --production
        # 2025 + 2026 model and hitter data; writes model/sequencing.json and
        # model/batter_adj/*.json for the Pitch Sequencer page
"""

import argparse
import json
import re
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

import features as F
import hitter_timing as H
import perceptual as M
import sequencing as Q
import validate_sequencer as V

HERE = Path(__file__).resolve().parent
OUT = HERE / "hitter_sequencer_report.md"
ADJ_DIR = HERE / "batter_adj"
PRIOR = 100
MIN_2025 = 200
N_BOOT = 1000
SEED = 7
N_SHARDS = 24
PAGE_MIN_PITCHES = 2500  # hitters on the page: seen >= this many pitches in 2025 + 2026
CLASSES = ["fb", "brk", "off"]
BREAKING = {"SL", "ST", "CU", "KC", "SV", "CS"}
BAT_CLS = ["bat_cls_whiff", "bat_cls_swing", "bat_cls_chase"]
COLS_H = H.COLS + BAT_CLS
BAT_COLS = ["tau", "bat_whiff_rate", "bat_swing_rate"] + [f"{k}_{c}" for k in BAT_CLS for c in CLASSES]

def pclass(t):
    t = pd.Series(t).astype(str)
    return np.where(t.isin(F.FASTBALLS), "fb", np.where(t.isin(BREAKING), "brk", "off"))

def class_sums(d):
    zone = d.zone.to_numpy() if "zone" in d else np.full(len(d), np.nan)
    ooz = (zone > 9) & ~np.isnan(zone)
    return pd.DataFrame({"batter": d.batter.to_numpy(), "cls": pclass(d.pitch_type), "game_pk": d.game_pk.to_numpy(),
                         "n": 1.0, "sw": d.swing.to_numpy(float), "wh": d.whiff.to_numpy(float),
                         "oz": ooz.astype(float), "ozsw": (ooz & d.swing.to_numpy()).astype(float)})

def shrink(s, league):
    lw = s.cls.map(league.wh / league.sw)
    ls = s.cls.map(league.sw / league.n)
    lc = s.cls.map(league.ozsw / league.oz)
    return pd.DataFrame({"bat_cls_whiff": (s.wh + PRIOR * lw) / (s.sw + PRIOR),
                         "bat_cls_swing": (s.sw + PRIOR * ls) / (s.n + PRIOR),
                         "bat_cls_chase": (s.ozsw + PRIOR * lc) / (s.oz + PRIOR)}, index=s.index)

def add_class_rates_logo(d):
    """Training rows: the hitter's class rates from the same season, leaving out the current game."""
    s = class_sums(d)
    league = s.groupby("cls")[["n", "sw", "wh", "oz", "ozsw"]].sum()
    v = ["n", "sw", "wh", "oz", "ozsw"]
    tot = s.groupby(["batter", "cls"])[v].transform("sum")
    game = s.groupby(["batter", "cls", "game_pk"])[v].transform("sum")
    loo = (tot - game).assign(cls=s.cls)
    r = shrink(loo, league)
    for c in BAT_CLS:
        d[c] = r[c].to_numpy()
    return d

def batter_table(d, tau, tau_default):
    """One row per hitter from season(s) d: timing, overall rates and class rates (all of d)."""
    s = class_sums(d)
    league = s.groupby("cls")[["n", "sw", "wh", "oz", "ozsw"]].sum()
    g = s.groupby(["batter", "cls"])[["n", "sw", "wh", "oz", "ozsw"]].sum().reset_index()
    r = shrink(g, league).assign(batter=g.batter, cls=g.cls)
    wide = r.pivot(index="batter", columns="cls", values=BAT_CLS)
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    for k in BAT_CLS:
        for c in CLASSES:
            col = f"{k}_{c}"
            fill = shrink(pd.DataFrame({"cls": [c], **{x: [0.0] for x in ["n", "sw", "wh", "oz", "ozsw"]}}), league)[k].iloc[0]
            wide[col] = wide[col].fillna(fill) if col in wide else fill
    o = s.groupby("batter")[["n", "sw", "wh"]].sum()
    lw, ls = s.wh.sum() / s.sw.sum(), s.sw.sum() / s.n.sum()
    wide["bat_whiff_rate"] = (o.wh + M.BATTER_PRIOR * lw) / (o.sw + M.BATTER_PRIOR)
    wide["bat_swing_rate"] = (o.sw + M.BATTER_PRIOR * ls) / (o.n + M.BATTER_PRIOR)
    wide["tau"] = pd.Series(tau).reindex(wide.index).fillna(tau_default)
    wide["n"] = o.n
    return wide

def average_hitter(table, fit_on, tau_avg):
    """Median overall rates (as the sequencer always used) and league class rates."""
    avg = {"tau": tau_avg, **fit_on[["bat_whiff_rate", "bat_swing_rate"]].median().to_dict()}
    s = class_sums(fit_on)
    league = s.groupby("cls")[["n", "sw", "wh", "oz", "ozsw"]].sum()
    for c in CLASSES:
        z = shrink(pd.DataFrame({"cls": [c], **{x: [0.0] for x in ["n", "sw", "wh", "oz", "ozsw"]}}), league)
        for k in BAT_CLS:
            avg[f"{k}_{c}"] = z[k].iloc[0]
    return pd.Series(avg)

def apply_hitter(rows, h):
    """Put hitter(s) h (DataFrame aligned with rows, BAT_COLS) into candidate rows and recompute timing features."""
    rows["bat_whiff_rate"] = h.bat_whiff_rate.to_numpy()
    rows["bat_swing_rate"] = h.bat_swing_rate.to_numpy()
    cls = pclass(rows.pitch_type)
    for k in BAT_CLS:
        vals = np.stack([h[f"{k}_{c}"].to_numpy() for c in CLASSES], axis=1)
        rows[k] = vals[np.arange(len(rows)), pd.Categorical(cls, categories=CLASSES).codes]
    return H.set_tau(rows, h.tau.to_numpy())

def timing_choice():
    rep = (HERE / "hitter_timing_report.md").read_text()
    passed = "**Part 1 passes: True.**" in rep
    tau_star = float(re.search(r"League best tau\*: (\d+) ms", rep).group(1)) / 1000
    return passed, tau_star

def train(d, cols):
    return {y: lgb.train(M.S.PARAMS, lgb.Dataset(data[cols], data[y].astype(int)), M.S.ROUNDS)
            for y, data in [("swing", d), ("whiff", d[d.swing]), ("csw", d)]}

def pitcher_sides(d26):
    counts = d26.groupby("pitcher").size()
    for pid in counts[counts >= Q.MIN_PITCHES].index:
        p = d26[d26.pitcher == pid]
        for side in ["R", "L"]:
            sub = p[p.stand.astype(str) == side]
            if len(sub) < Q.MIN_SIDE:
                continue
            use = sub.pitch_type.astype(str).value_counts()
            types = [t for t in use.index if use[t] >= Q.MIN_TYPE and use[t] / len(sub) >= Q.MIN_USAGE]
            if types:
                sub = sub.copy()
                sub["pitch_type"] = sub.pitch_type.astype(str)
                yield pid, side, sub, types, use

def build(models, fit_on, d26, cats, hitters, avg, batters_for, type_median=None):
    """Side-only grid (average hitter) and batter adjustments, per pitcher x side.
    batters_for(pid, side) -> list of hitter ids to compute adjustments for. With type_median (LOOK
    features by pitch type), the grid also carries the deception lift, as in sequencing.py."""
    base, tables = Q.usage_tables(d26)
    prev_desc = Q.typical_prev_desc(fit_on)
    medians = pd.Series({"bat_whiff_rate": avg.bat_whiff_rate, "bat_swing_rate": avg.bat_swing_rate})
    out, adj = {}, {}
    for i, (pid, side, sub, types, use) in enumerate(pitcher_sides(d26)):
        cells = Q.build_cells(sub, pid, side, types, base, tables, prev_desc, medians)
        for c in Q.CATS:
            cells[c] = pd.Categorical(cells[c].astype(str), categories=cats[c])
        cells = apply_hitter(cells, pd.DataFrame(np.repeat(avg[BAT_COLS].to_numpy(float)[None], len(cells), 0), columns=BAT_COLS))
        cells["csw_p"] = models["csw"].predict(cells[COLS_H])
        cells["sw_p"] = models["swing"].predict(cells[COLS_H])
        cells["wh_p"] = cells.sw_p * models["whiff"].predict(cells[COLS_H])
        cells["lift"] = 0.0
        if type_median is not None:
            cf = cells.copy()
            cf[Q.LOOK] = type_median.reindex(cells.pitch_type.astype(str)).to_numpy()
            cells["lift"] = cells.csw_p - models["csw"].predict(cf[COLS_H])
        g = cells.groupby(["_count", "_prev", "_type"], sort=False)[["sw_p", "wh_p", "csw_p", "lift"]].mean()
        grid = {}
        for (count, prev, t), r in g.iterrows():
            grid.setdefault(count, {}).setdefault(prev, {})[t] = [round(r.sw_p * 100, 1), round(r.wh_p * 100, 1),
                                                                 round(r.csw_p * 100, 1), round(r.lift * 100, 1)]
        out.setdefault(pid, {})[side] = {"types": types, "usage": [round(use[t] / len(sub), 3) for t in types], "grid": grid}

        # Adjustment rows: previous pitch = his primary fastball to this side (none at 0-0).
        fb = sub.fb_type.iloc[0]
        prev = fb if fb in types else types[0]
        rows = cells[(cells._prev == prev) | (cells._count == "0-0")].reset_index(drop=True)
        ids = [b for b in batters_for(pid, side) if b in hitters.index]
        if not ids:
            continue
        a0 = rows.groupby(["_count", "_type"], sort=False).csw_p.mean()
        for chunk in np.array_split(np.array(ids), max(1, len(ids) // 60)):
            rep = rows.loc[np.tile(np.arange(len(rows)), len(chunk))].reset_index(drop=True)
            who = np.repeat(chunk, len(rows))
            rep = apply_hitter(rep, hitters.loc[who, BAT_COLS].reset_index(drop=True))
            rep["csw_p"] = models["csw"].predict(rep[COLS_H])
            rep["_b"] = who
            a = rep.groupby(["_b", "_count", "_type"], sort=False).csw_p.mean()
            a = a - a0.reindex(a.index.droplevel(0)).to_numpy()
            for (b, count, t), v in a.items():
                adj.setdefault((pid, side), {}).setdefault(int(b), {}).setdefault(count, {})[t] = round(v * 100, 2)
        if i % 25 == 0:
            print(i, pid, side, len(ids), "hitters", flush=True)
    return out, adj

def prep(years_fit, tau_map, tau_default):
    """Load seasons, add class rates (same-season, leave-one-game-out) and set hitter timing."""
    seasons = {y: H.load(y) for y in (2025, 2026)}
    for d in seasons.values():
        add_class_rates_logo(d)
    H.categorize(*seasons.values())
    for d in seasons.values():
        if tau_map is not None:
            H.set_tau(d, d.batter.map(tau_map).fillna(tau_default).to_numpy())
    return seasons

def validate():
    passed, tau_star = timing_choice()
    fitted = pd.read_csv(H.TAUS_OUT, index_col="batter")
    tau_map = fitted.tau_fit if passed else None
    tau_avg = tau_star if passed else M.TAU0
    s = prep([2025], tau_map, tau_avg)
    tr, te = s[2025], s[2026]
    cats = {c: tr[c].cat.categories for c in Q.CATS}
    lines = ["# Batter-specific sequencer (pre-registered, single run)", "",
             f"Timing: {'fitted per-hitter tau (Part 1 passed)' if passed else 'bat-tracking tau (Part 1 did not pass)'}; "
             f"average hitter at {tau_avg * 1000:.0f} ms.", ""]

    # Secondary (c): does the model know more with the class features? 2026 class rates come from 2025.
    hit25 = batter_table(tr, fitted.tau_fit if passed else tr.groupby("batter").bat_tau.first(), tau_avg)
    te_c = te.copy()
    cls = pclass(te_c.pitch_type)
    for k in BAT_CLS:
        avg_k = {c: hit25[f"{k}_{c}"].median() for c in CLASSES}
        vals = [hit25[f"{k}_{c}"].reindex(te_c.batter).to_numpy() for c in CLASSES]
        pick = pd.Categorical(cls, categories=CLASSES).codes
        v = np.stack(vals, axis=1)[np.arange(len(te_c)), pick]
        league = np.array([average_hitter(hit25, tr, tau_avg)[f"{k}_{c}"] for c in CLASSES])[pick]
        te_c[k] = np.where(np.isnan(v), league, v)
    models = train(tr, COLS_H)
    models0 = train(tr, H.COLS)
    ll = {}
    for y in ["swing", "whiff", "csw"]:
        b = te_c[te_c.swing] if y == "whiff" else te_c
        yy = b[y].astype(int)
        ll[y] = (log_loss(yy, models0[y].predict(b[H.COLS])) - log_loss(yy, models[y].predict(b[COLS_H]))) * 1000
    print("log-loss gains", ll, flush=True)

    # Test pitches and the hitters to adjust for.
    te["prev"] = np.where(te["count"] == "0-0", "none", te.prev_pitch_type.astype(str))
    eligible = hit25.index[hit25.n >= MIN_2025]
    faced = te[te.batter.isin(eligible)].groupby(["pitcher", te.stand.astype(str)]).batter.unique()
    batters_for = lambda pid, side: list(faced.get((pid, side), []))
    avg = average_hitter(hit25, tr, tau_avg)
    grid, adj = build(models, tr, te, cats, hit25, avg, batters_for)

    rows = []
    for r in te[["pitcher", "batter", "stand", "count", "prev", "pitch_type", "csw"]].itertuples(index=False):
        sd = grid.get(r.pitcher, {}).get(str(r.stand))
        cell = sd and sd["grid"].get(r.count, {}).get(r.prev)
        pt = str(r.pitch_type)
        if not cell or pt not in cell or r.batter not in eligible:
            continue
        a = adj.get((r.pitcher, str(r.stand)), {}).get(int(r.batter), {}).get(r.count)
        if a is None:
            continue
        side_pick = max(cell, key=lambda t: cell[t][2])
        bat_pick = max(cell, key=lambda t: cell[t][2] + a.get(t, 0.0))
        rows.append((r.pitcher, r.count, pt, int(r.csw), side_pick, bat_pick))
    t = pd.DataFrame(rows, columns=["pitcher", "count", "pitch_type", "csw", "side_pick", "bat_pick"])
    t["m_side"], t["m_bat"] = t.pitch_type == t.side_pick, t.pitch_type == t.bat_pick
    differ = t.side_pick != t.bat_pick
    rng = np.random.default_rng(SEED)
    V.N_BOOT = N_BOOT
    w = lambda draw, g: g.pitcher.map(draw).fillna(0)

    # Primary.
    p = t[differ & (t.m_side | t.m_bat)].copy()
    gp = V.gap_table(p, "m_bat")
    est, per_count, n = V.within_count_gap(gp)
    ci = V.boot(lambda d: V.within_count_gap(gp, w(d, gp))[0], p.pitcher.unique(), rng)
    ok = est > 0 and ci[0] > 0
    # Secondary (a).
    ga, gb = V.gap_table(t, "m_bat"), V.gap_table(t, "m_side")
    diff = lambda wa=None, wb=None: V.within_count_gap(ga, wa)[0] - V.within_count_gap(gb, wb)[0]
    est2 = diff()
    ci2 = V.boot(lambda d: diff(w(d, ga), w(d, gb)), t.pitcher.unique(), rng)

    pc = pd.DataFrame({"pitches": n.astype(int), "gap pts": per_count.round(2)})
    lines += [f"Test pitches: {len(t):,} from {t.pitcher.nunique()} pitchers, hitters with >= {MIN_2025} pitches in 2025. "
              f"The batter pick differs from the side pick on {differ.mean():.1%} of them.", "",
              "## Primary: batter pick vs side pick where they differ", "",
              f"{len(p):,} pitches where the picks differ and the pitch thrown was one of the two "
              f"({p.m_bat.mean():.0%} were the batter pick).", "",
              f"Within-count CSW, thrown = batter pick minus thrown = side pick: **{est:+.2f} pts** "
              f"(95% CI {ci[0]:+.2f} to {ci[1]:+.2f}). Pass: **{ok}**.", "",
              pc.to_markdown(), "",
              "## Secondary", "",
              f"(a) Matched-minus-unmatched gap: batter pick {V.within_count_gap(ga)[0]:+.2f} pts, side pick "
              f"{V.within_count_gap(gb)[0]:+.2f} pts. Difference **{est2:+.2f} pts** (95% CI {ci2[0]:+.2f} to {ci2[1]:+.2f}).", "",
              f"(b) Picks differ on {differ.mean():.1%} of test pitches.", "",
              "(c) 2026 log-loss gain x1000 from the batter class features: "
              + ", ".join(f"{k} {v:+.2f}" for k, v in ll.items()) + ".", ""]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

def production():
    passed, tau_star = timing_choice()
    fitted = pd.read_csv(H.TAUS_OUT, index_col="batter")
    tau_map = fitted.tau_fit if passed else None
    tau_avg = tau_star if passed else M.TAU0
    s = prep([2025, 2026], tau_map, tau_avg)
    both = pd.concat([s[2025], s[2026]], ignore_index=True)
    for c in Q.CATS:
        both[c] = pd.Categorical(both[c].astype(str), categories=s[2025][c].cat.categories)
    cats = {c: both[c].cat.categories for c in Q.CATS}
    d26 = s[2026]
    bat_tau = both.groupby("batter").bat_tau.last()
    hitters = batter_table(both, fitted.tau_fit if passed else bat_tau, tau_avg)
    page = hitters.index[hitters.n >= PAGE_MIN_PITCHES]
    seen = d26[d26.batter.isin(page)]
    stand = seen.stand.astype(str).groupby(seen.batter).agg(lambda x: set(x))
    models = train(both, COLS_H)
    avg = average_hitter(hitters, both, tau_avg)
    batters_for = lambda pid, side: [b for b in page if side in stand.get(b, set())]
    type_median = d26.groupby("pitch_type", observed=True)[Q.LOOK].median()
    type_median.index = type_median.index.astype(str)
    grid, adj = build(models, both, d26, cats, hitters, avg, batters_for, type_median)
    who = Q.names(set(grid) | set(page))
    out = {"counts": Q.COUNTS, "pitchers": {}, "shards": N_SHARDS,
           "hitters": {str(int(b)): {"name": who.get(int(b), str(b)), "stand": sorted(stand.get(b, set()))} for b in page}}
    for pid, sides in grid.items():
        p = d26[d26.pitcher == pid]
        out["pitchers"][str(int(pid))] = {"name": who.get(int(pid), str(pid)), "throws": str(p.p_throws.iloc[0]),
                                          "n": int(len(p)), "sides": sides}
    (HERE / "sequencing.json").write_text(json.dumps(out, separators=(",", ":")))
    ADJ_DIR.mkdir(exist_ok=True)
    shards = {}
    for (pid, side), by_b in adj.items():
        shards.setdefault(int(pid) % N_SHARDS, {}).setdefault(str(int(pid)), {})[side] = {
            str(b): {c: [round(v[t], 1) for t in grid[pid][side]["types"]] for c, v in counts.items()}
            for b, counts in by_b.items()}
    for k, v in shards.items():
        (ADJ_DIR / f"adj_{k:02d}.json").write_text(json.dumps(v, separators=(",", ":")))
    print("wrote sequencing.json and", len(shards), "adjustment shards for", len(page), "hitters")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", action="store_true")
    production() if parser.parse_args().production else validate()
