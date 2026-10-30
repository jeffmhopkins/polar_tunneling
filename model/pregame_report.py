"""
Pre-game pitch plan for one starter against the opposing lineup.

Everything is fitted on 2025 plus 2026 regular-season pitches before the game
date, so the plan only uses what was known the morning of the game.

- Picks come from the standard model by runs (runs_compare.py arm RB): it tied
  the perceptual model on picks (runs_report.md, location_report.md) and is
  simpler.
- Swing and whiff forecasts and the deception lift come from the perceptual
  model, which forecasts swings and whiffs better (perceptual_report.md).
- Each hitter's plan is compared with the pitcher's habit (his most-used type
  in that count against that side), in model runs saved.

Usage:
    python model/pregame_report.py --game 823491 --pitcher 669432
        # writes report_generation/pregame_<pk>_<pitcher>.json
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import damage_sequencer as DS
import features as F
import hitter_sequencer as HS
import hitter_timing as H
import perceptual as M
import runs_compare as R
import sequencing as Q

HERE = Path(__file__).resolve().parent
PAGE_DIR = HERE.parent / "report_generation"
CACHE = Path("/tmp/pregame_cache_{}.pkl")
COLS_STD = R.SETS["RB"]
COLS_PERC = HS.COLS_H
HABIT_MIN = 10       # pitches in a count before his habit there is trusted; else his overall top type
LOW_SAMPLE = 1000    # hitters seen fewer pitches than this get a caution flag
MENU_MIN = 0.05      # a pick must be a pitch he throws at least this often in that count against that side

def game_info(pk):
    box = requests.get(f"https://statsapi.mlb.com/api/v1/game/{pk}/boxscore", timeout=60).json()
    feed = requests.get(f"https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live", timeout=60).json()["gameData"]
    out = {"pk": pk, "date": feed["datetime"]["officialDate"], "venue": feed["venue"]["name"], "teams": {}}
    for s in ("away", "home"):
        t = box["teams"][s]
        lineup = []
        for pid in t["battingOrder"]:
            p = t["players"][f"ID{pid}"]
            lineup.append({"id": pid, "name": p["person"]["fullName"], "pos": p["position"]["abbreviation"]})
        out["teams"][s] = {"name": t["team"]["name"], "abbr": feed["teams"][s]["abbreviation"],
                           "starter": t["pitchers"][0], "lineup": lineup}
    return out

def load_before(date):
    """Both seasons with features, 2026 cut to games before `date` (season-level features too)."""
    path = Path(str(CACHE).format(date))
    if path.exists():
        return pickle.loads(path.read_bytes())
    orig = F.load
    F.load = lambda year: (lambda d: d[d.game_date < date].reset_index(drop=True))(orig(year))
    try:
        seasons = {y: H.load(y) for y in (2025, 2026)}
    finally:
        F.load = orig
    for d in seasons.values():
        HS.add_class_rates_logo(d)
    H.categorize(*seasons.values())
    path.write_bytes(pickle.dumps(seasons))
    return seasons

def reach(fit):
    """Share of plate appearances that pass through each count."""
    pa = fit.groupby(["game_pk", "at_bat_number"]).ngroups
    return (fit.groupby("count", observed=True).size() / pa).reindex(Q.COUNTS).fillna(0).to_dict()

def habit(sub, types):
    """His most-used type and his usage share of each type, per count (overall when the count is thin)."""
    use = sub[sub.pitch_type.isin(types)]
    overall = use.pitch_type.value_counts(normalize=True)
    hab, share = {}, {}
    for c in Q.COUNTS:
        v = use[use["count"].astype(str) == c].pitch_type.value_counts()
        v = v / v.sum() if v.sum() >= HABIT_MIN else overall
        hab[c] = v.index[0]
        share[c] = {t: round(float(v.get(t, 0.0)), 3) for t in types}
    return hab, share

def score(cells, h, std, perc, type_median, rv, weights):
    """Per count and type for hitter row(s) h: [runs saved/100, CSW%, swing%, whiff%, deception pts], prev-weighted."""
    rows = HS.apply_hitter(cells.copy(), h)
    p = {k: m.predict(rows[COLS_STD]) for k, m in std.items()}
    rows["rs"] = -100 * DS.expected_rv(p, rows["count"].astype(str), rv)
    rows["csw"] = 100 * p["csw"]
    q = {k: m.predict(rows[COLS_PERC]) for k, m in perc.items()}
    rows["sw"], rows["wh"] = 100 * q["swing"], 100 * q["swing"] * q["whiff"]
    cf = rows.copy()
    cf[Q.LOOK] = type_median.reindex(rows.pitch_type.astype(str)).to_numpy()
    rows["lift"] = 100 * (q["csw"] - perc["csw"].predict(cf[COLS_PERC]))
    rows["w"] = rows._prev.map(weights).fillna(1.0)
    vals = ["rs", "csw", "sw", "wh", "lift"]
    g = rows.groupby(["_count", "_prev", "_type"], sort=False)[vals + ["w"]].mean()
    g[vals] = g[vals].mul(g.w, axis=0)
    g = g.groupby(level=["_count", "_type"], sort=False).sum()
    g[vals] = g[vals].div(g.w, axis=0)
    return g[vals]

def plan(g, types, hab, share, rch):
    """Pick the best runs-saved type among those he actually throws in each count."""
    counts, stake = {}, 0.0
    for c in Q.COUNTS:
        cell = {t: [round(float(x), 2) for x in g.loc[(c, t)]] for t in types}
        menu = [t for t in types if share[c][t] >= MENU_MIN] or [hab[c]]
        pick = max(menu, key=lambda t: cell[t][0])
        best = max(types, key=lambda t: cell[t][0])
        delta = cell[pick][0] - cell[hab[c]][0]
        stake += rch[c] * delta
        counts[c] = {"cells": cell, "pick": pick, "habit": hab[c], "delta": round(delta, 2), "use": share[c],
                     "off_menu": best if best != pick else None}
    return counts, round(stake, 2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, default=823491)
    ap.add_argument("--pitcher", type=int, default=669432)
    args = ap.parse_args()
    info = game_info(args.game)
    side_of = "home" if info["teams"]["away"]["starter"] == args.pitcher else "away"
    opp = info["teams"][side_of]
    s = load_before(info["date"])
    cats = {c: s[2025][c].cat.categories for c in Q.CATS}
    fit = DS.add_outcomes(pd.concat([s[2025], s[2026]], ignore_index=True))
    for c in Q.CATS:
        fit[c] = pd.Categorical(fit[c].astype(str), categories=cats[c])
    d26 = s[2026]
    rv = DS.run_values(fit)
    hitters = HS.batter_table(fit, fit.groupby("batter").bat_tau.last(), M.TAU0)
    avg = HS.average_hitter(hitters, fit, M.TAU0)
    mpath = Path(str(CACHE).format(f"models_{info['date']}"))
    if mpath.exists():
        std, perc = pickle.loads(mpath.read_bytes())
    else:
        std = R.train(fit, COLS_STD)
        print("trained standard", flush=True)
        perc = HS.train(fit, COLS_PERC)
        print("trained perceptual", flush=True)
        mpath.write_bytes(pickle.dumps((std, perc)))
    type_median = d26.groupby("pitch_type", observed=True)[Q.LOOK].median()
    type_median.index = type_median.index.astype(str)
    base, tables = Q.usage_tables(d26)
    prev_desc = Q.typical_prev_desc(fit)
    medians = pd.Series({"bat_whiff_rate": avg.bat_whiff_rate, "bat_swing_rate": avg.bat_swing_rate})
    rch = reach(fit)

    me = d26[d26.pitcher == args.pitcher]
    throws = str(me.p_throws.iloc[0])
    names = Q.names([args.pitcher])
    out = {"game": info, "pitcher": {"id": args.pitcher, "name": names[args.pitcher], "throws": throws,
                                    "n26": int(len(me)), "n25": int((s[2025].pitcher == args.pitcher).sum())},
           "cutoff": info["date"], "reach": {c: round(v, 3) for c, v in rch.items()}, "sides": {}, "hitters": []}
    seen = fit.groupby("batter").size()
    stands = fit.stand.astype(str).groupby(fit.batter).agg(lambda x: set(x))
    for side in ["R", "L"]:
        sub = me[me.stand.astype(str) == side].copy()
        sub["pitch_type"] = sub.pitch_type.astype(str)
        use = sub.pitch_type.value_counts()
        types = [t for t in use.index if use[t] >= Q.MIN_TYPE and use[t] / len(sub) >= Q.MIN_USAGE]
        if len(sub) < Q.MIN_SIDE or not types:
            continue
        cells = Q.build_cells(sub, args.pitcher, side, types, base, tables, prev_desc, medians)
        for c in Q.CATS:
            cells[c] = pd.Categorical(cells[c].astype(str), categories=cats[c])
        weights = (use[types] / use[types].sum()).to_dict()
        hab, share = habit(sub, types)
        a = score(cells, pd.DataFrame(np.repeat(avg[HS.BAT_COLS].to_numpy(float)[None], len(cells), 0), columns=HS.BAT_COLS),
                  std, perc, type_median, rv, weights)
        counts, stake = plan(a, types, hab, share, rch)
        arsenal = sub[sub.pitch_type.isin(types)].groupby("pitch_type").agg(
            mph=("release_speed_mph", "mean"), hb=("pfx_x_ft", "mean"), ivb=("pfx_z_ft", "mean"))
        out["sides"][side] = {"n": int(len(sub)), "types": types, "usage": {t: round(use[t] / len(sub), 3) for t in types},
                              "arsenal": {t: {"mph": round(r.mph, 1), "hb_in": round(r.hb * 12, 1), "ivb_in": round(r.ivb * 12, 1)}
                                          for t, r in arsenal.iterrows()},
                              "average": {"counts": counts, "stake": stake}}
        for slot, b in enumerate(opp["lineup"], 1):
            st = stands.get(b["id"], set())
            bats = ("L" if throws == "R" else "R") if len(st) > 1 else next(iter(st), None)
            if bats != side:
                continue
            row = hitters.loc[[b["id"]]] if b["id"] in hitters.index else avg[HS.BAT_COLS].to_frame().T
            g = score(cells, pd.DataFrame(np.repeat(row[HS.BAT_COLS].to_numpy(float), len(cells), 0), columns=HS.BAT_COLS),
                      std, perc, type_median, rv, weights)
            counts, stake = plan(g, types, hab, share, rch)
            out["hitters"].append({**b, "slot": slot, "bats": bats, "switch": len(st) > 1, "seen": int(seen.get(b["id"], 0)),
                                   "low_sample": int(seen.get(b["id"], 0)) < LOW_SAMPLE, "counts": counts, "stake": stake})
            print(slot, b["name"], bats, "stake", stake, flush=True)
    out["hitters"].sort(key=lambda h: h["slot"])
    path = PAGE_DIR / f"pregame_{args.game}_{args.pitcher}.json"
    path.write_text(json.dumps(out, separators=(",", ":")))
    print("wrote", path)

if __name__ == "__main__":
    main()
