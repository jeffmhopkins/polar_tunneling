"""
Model comparison and game report, as pre-registered in PREREG_game.md.

Five pitch pickers, all built from 2025 only (D and E use 2025 results and
2026 usage):
  A  perceptual sequencer (the validated sequencer model)
  B  standard pitch model: same LightGBM without the perceptual features
  C  stuff-only model (Stuff+-style: no location, count or previous pitch)
  D  best pitch for the count (pitcher's best 2025 CSW type in that count)
  E  pitcher's habit (his most-used 2026 type in that count)

Season comparison: matched-minus-unmatched within-count CSW gap of each
picker on 2026 regular-season pitches, A minus each other one (pitcher
bootstrap). Game report: every pitch of one postseason game, with each
picker's choice, the predicted CSW of each candidate, and what happened.

Usage:
    python model/game_report.py                  # season comparison + game 823491 (Orioles at Yankees, Sept 25)
    python model/game_report.py --game 849803 --skip-season
        # writes model/game_report.md and report_generation/game_<pk>.json
"""

import argparse
import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import log_loss

import features as F
import perceptual as M
import sequencing as Q
import speed_check as SC
import validate_sequencer as V

HERE = Path(__file__).resolve().parent
OUT = HERE / "game_report.md"
PAGE_DIR = HERE.parent / "report_generation"
CACHE = Path("/tmp/game_report_cache.pkl")
SEED = 7
GAME = 823491
MODELS = ["A", "B", "C", "D", "E"]
LABELS = {"A": "Perceptual sequencer", "B": "Standard pitch model", "C": "Stuff-only (Stuff+-style)",
          "D": "Best pitch for the count", "E": "Pitcher's habit"}
COLS_A = Q.COLS
COLS_B = M.CONTROLS + SC.SPEED
COLS_C = ["pitch_type", "stand", "p_throws", "release_speed_mph", "mov_x", "pfx_z_ft", "release_spin_rate_rpm",
          "release_extension_ft", "release_pos_z_ft", "rel_x", "arm_angle_deg"] + M.DIFF
GAME_MIN = dict(pitches=300, side=60, type=15)

def load(games):
    """Both seasons with features; postseason rows of `games` are kept in 2026 (marked `_game`)."""
    if CACHE.exists():
        d = pickle.loads(CACHE.read_bytes())
        if d["games"] == set(games):
            return d["df"]
    orig = F.load

    def load_with_games(year):
        files = sorted((F.ROOT / "datamining" / "outputs").glob(f"pitches_{year}_*.csv"))
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        df = df[((df.game_type == "R") | df.game_pk.isin(games)) & ~df.pitch_type.isin(F.EXCLUDE)]
        need = ["vx0_ft_s", "vy0_ft_s", "vz0_ft_s", "ax_ft_s2", "ay_ft_s2", "az_ft_s2",
                "release_pos_y_ft", "plate_x_ft", "plate_z_ft", "pitch_type", "stand"]
        return df.dropna(subset=need).reset_index(drop=True)

    F.load = load_with_games
    try:
        df = pd.concat([SC.add_speed(M.season(y)) for y in (2025, 2026)], ignore_index=True)
    finally:
        F.load = orig
    for c in Q.CATS:
        df[c] = pd.Categorical(df[c].astype(object).fillna("none").astype(str).replace("nan", "none"))
    df["_game"] = df.game_pk.isin(games)
    CACHE.write_bytes(pickle.dumps({"games": set(games), "df": df}))
    return df

def train(tr):
    out = {}
    for name, cols in [("A", COLS_A), ("B", COLS_B), ("C", COLS_C)]:
        out[name] = {y: lgb.train(M.S.PARAMS, lgb.Dataset(data[cols], data[y].astype(int)), M.S.ROUNDS)
                     for y, data in [("swing", tr), ("whiff", tr[tr.swing]), ("csw", tr)]}
        print("trained", name, flush=True)
    return out

def predict(models, name, X):
    cols = {"A": COLS_A, "B": COLS_B, "C": COLS_C}[name]
    m = models[name]
    sw = m["swing"].predict(X[cols])
    return sw, sw * m["whiff"].predict(X[cols]), m["csw"].predict(X[cols])

def sides(d26, mins):
    counts = d26.groupby("pitcher").size()
    for pid in counts[counts >= mins["pitches"]].index:
        p = d26[d26.pitcher == pid]
        for side in ["R", "L"]:
            sub = p[p.stand.astype(str) == side]
            if len(sub) < mins["side"]:
                continue
            use = sub.pitch_type.astype(str).value_counts()
            types = [t for t in use.index if use[t] >= mins["type"] and use[t] / len(sub) >= Q.MIN_USAGE]
            if types:
                sub = sub.copy()
                sub["pitch_type"] = sub.pitch_type.astype(str)
                yield pid, side, sub, types, use

def grids(models, tr, d26, cats, mins, only=None):
    """{(pid, side): {"types", "usage", "A"/"B"/"C": {count: {prev: {type: [sw, wh, csw]}}}}}."""
    base, tables = Q.usage_tables(d26)
    prev_desc = Q.typical_prev_desc(tr)
    medians = tr[["bat_whiff_rate", "bat_swing_rate"]].median()
    out = {}
    for i, (pid, side, sub, types, use) in enumerate(sides(d26 if only is None else d26[d26.pitcher.isin(only)], mins)):
        cells = Q.build_cells(sub, pid, side, types, base, tables, prev_desc, medians)
        for c in Q.CATS:
            cells[c] = pd.Categorical(cells[c].astype(str), categories=cats[c])
        entry = {"types": types, "usage": {t: round(use[t] / len(sub), 3) for t in types}}
        for name in ["A", "B", "C"]:
            cells["sw"], cells["wh"], cells["csw"] = predict(models, name, cells)
            g = cells.groupby(["_count", "_prev", "_type"], sort=False)[["sw", "wh", "csw"]].mean()
            grid = {}
            for (count, prev, t), r in g.iterrows():
                grid.setdefault(count, {}).setdefault(prev, {})[t] = [r.sw, r.wh, r.csw]
            entry[name] = grid
        out[(pid, side)] = entry
        if i % 50 == 0:
            print(i, pid, side, flush=True)
    return out

def habit_picks(d26):
    n = d26.assign(pitch_type=d26.pitch_type.astype(str), stand=d26.stand.astype(str)) \
        .groupby(["pitcher", "stand", "count", "pitch_type"]).size().reset_index(name="n")
    return n.sort_values("n").groupby(["pitcher", "stand", "count"]).tail(1).set_index(["pitcher", "stand", "count"]).pitch_type.to_dict()

def cell_of(entry, name, count, prev):
    """Candidate -> [sw, wh, csw] for a count and previous type; unknown previous types average over his types."""
    by_prev = entry[name].get(count, {})
    if prev in by_prev:
        return by_prev[prev]
    rows = list(by_prev.values())
    return {t: list(np.mean([r[t] for r in rows], axis=0)) for t in entry["types"]}

def picks(entry, count, prev, naive, habit):
    out = {}
    for name in ["A", "B", "C"]:
        cell = cell_of(entry, name, count, prev)
        out[name] = max(cell, key=lambda t: cell[t][2])
    out["D"], out["E"] = naive, habit
    return out

# ---- season comparison ----

def season_compare(g, d26, cell_pick, side_pick, habit, lines):
    t = d26[~d26._game].copy()
    t["prev"] = np.where(t["count"] == "0-0", "none", t.prev_pitch_type.astype(str))
    rows = []
    for r in t[["pitcher", "stand", "count", "prev", "pitch_type", "csw"]].itertuples(index=False):
        e = g.get((r.pitcher, str(r.stand)))
        pt = str(r.pitch_type)
        if not e or r.prev not in e["A"].get(r.count, {}) or pt not in e["types"]:
            continue
        p = picks(e, r.count, r.prev, cell_pick.get((r.pitcher, str(r.stand), r.count), side_pick.get((r.pitcher, str(r.stand)))),
                  habit.get((r.pitcher, str(r.stand), r.count)))
        rows.append((r.pitcher, r.count, pt, int(r.csw), *[p[m] for m in MODELS]))
    t = pd.DataFrame(rows, columns=["pitcher", "count", "pitch_type", "csw"] + MODELS)
    for m in MODELS:
        t["m_" + m] = t.pitch_type == t[m]
    rng = np.random.default_rng(SEED)
    w = lambda draw, gg: gg.pitcher.map(draw).fillna(0)
    lines += [f"Test pitches: {len(t):,} from {t.pitcher.nunique()} pitchers (2026 regular season).", "",
              "| picker | pitches | share thrown | CSW when thrown | gap pts (95% CI) |", "|:--|--:|--:|--:|:--|"]
    res = {}
    for m in MODELS:
        s = t[t[m].notna()]
        gg = V.gap_table(s, "m_" + m)
        est = V.within_count_gap(gg)[0]
        ci = V.boot(lambda d: V.within_count_gap(gg, w(d, gg))[0], s.pitcher.unique(), rng)
        res[m] = dict(n=len(s), share=s["m_" + m].mean(), csw=s[s["m_" + m]].csw.mean(), gap=est, ci=list(ci))
        lines.append(f"| {m}. {LABELS[m]} | {len(s):,} | {s['m_' + m].mean():.1%} | {s[s['m_' + m]].csw.mean():.1%} | "
                     f"{est:+.2f} ({ci[0]:+.2f} to {ci[1]:+.2f}) |")
    lines += ["", "## A minus each other picker (same pitches)", "",
              "| comparison | pitches | picks agree | A gap minus other gap (95% CI) |", "|:--|--:|--:|:--|"]
    for m in MODELS[1:]:
        s = t[t[m].notna()]
        ga, gb = V.gap_table(s, "m_A"), V.gap_table(s, "m_" + m)
        diff = lambda wa=None, wb=None: V.within_count_gap(ga, wa)[0] - V.within_count_gap(gb, wb)[0]
        est = diff()
        ci = V.boot(lambda d: diff(w(d, ga), w(d, gb)), s.pitcher.unique(), rng)
        res["A-" + m] = dict(diff=est, ci=list(ci), agree=(s.A == s[m]).mean())
        tag = " **(primary)**" if m == "B" else ""
        lines.append(f"| A vs {m}. {LABELS[m]}{tag} | {len(s):,} | {(s.A == s[m]).mean():.0%} | "
                     f"{est:+.2f} ({ci[0]:+.2f} to {ci[1]:+.2f}) |")
    ok = res["A-B"]["diff"] > 0 and res["A-B"]["ci"][0] > 0
    lines += ["", f"Primary (A beats the standard model): **{ok}**.", ""]
    res["primary_pass"] = bool(ok)
    return res

def season_logloss(models, d26, lines):
    """How well each model predicts the 2026 pitches actually thrown (not picks)."""
    te = d26[~d26._game]
    rows = []
    for name in ["A", "B", "C"]:
        sw, wh, csw = predict(models, name, te)
        rows.append({"model": f"{name}. {LABELS[name]}", "swing": log_loss(te.swing.astype(int), sw),
                     "whiff": log_loss(te.whiff.astype(int), wh), "CSW": log_loss(te.csw.astype(int), csw)})
    r = pd.DataFrame(rows).set_index("model")
    base = r.iloc[2]
    lines += ["## Predicting the 2026 pitches actually thrown (log-loss, lower is better)", "",
              f"{len(te):,} pitches. Gain x1000 vs the stuff-only model in brackets.", "",
              "| model | swing | whiff (per pitch) | CSW |", "|:--|--:|--:|--:|"]
    for k, x in r.iterrows():
        lines.append("| " + k + " | " + " | ".join(f"{x[c]:.4f} ({(base[c] - x[c]) * 1000:+.1f})" for c in r.columns) + " |")
    lines.append("")
    return {k: {c: float(x[c]) for c in r.columns} for k, x in r.iterrows()}

# ---- game report ----

def people(ids):
    out = {}
    ids = [int(i) for i in ids]
    for i in range(0, len(ids), 100):
        r = requests.get("https://statsapi.mlb.com/api/v1/people", params={"personIds": ",".join(map(str, ids[i:i + 100]))}, timeout=60)
        r.raise_for_status()
        out.update({p["id"]: p["fullName"] for p in r.json()["people"]})
    return out

def game_info(pk):
    r = requests.get("https://statsapi.mlb.com/api/v1/schedule", params={"sportId": 1, "gamePks": pk}, timeout=60).json()
    g = r["dates"][0]["games"][0]
    t = g["teams"]
    return {"pk": pk, "date": g["officialDate"], "series": g.get("seriesDescription", ""), "game": g.get("seriesGameNumber"),
            "away": t["away"]["team"]["name"], "home": t["home"]["team"]["name"],
            "away_score": t["away"].get("score"), "home_score": t["home"].get("score")}

def game_report(pk, models, tr, d26_all, cats, cell_pick, side_pick, habit, lines):
    d26 = d26_all[~d26_all._game]
    gm = d26_all[d26_all.game_pk == pk].sort_values(["at_bat_number", "pitch_number"])
    raw = pd.concat([pd.read_csv(f) for f in sorted((F.ROOT / "datamining" / "outputs").glob("pitches_2026_*.csv"))])
    raw = raw[raw.game_pk == pk].sort_values(["at_bat_number", "pitch_number"]).reset_index(drop=True)
    g = grids(models, tr, d26, cats, GAME_MIN, only=set(raw.pitcher))
    info = game_info(pk)
    who = people(set(raw.pitcher) | set(raw.batter))

    # Per-pitch predictions for the pitch actually thrown (where features exist).
    feats = {}
    if len(gm):
        for name in ["A", "B", "C"]:
            sw, wh, csw = predict(models, name, gm)
            for k, a, b, c in zip(zip(gm.at_bat_number, gm.pitch_number), sw, wh, csw):
                feats.setdefault(k, {})[name] = [round(a * 100, 1), round(b * 100, 1), round(c * 100, 1)]
    raw["prev_type"] = raw.groupby("at_bat_number").pitch_type.shift(1)
    pitches = []
    for r in raw.itertuples(index=False):
        count = f"{int(r.balls)}-{int(r.strikes)}"
        prev = "none" if count == "0-0" else str(r.prev_type)
        e = g.get((r.pitcher, r.stand))
        d = str(r.description)
        csw = d in F.WHIFF or d == "called_strike"
        rec = {"ab": int(r.at_bat_number), "n": int(r.pitch_number), "inning": int(r.inning), "half": r.inning_topbot,
               "outs": int(r.outs_when_up), "pitcher": int(r.pitcher), "batter": int(r.batter), "stand": r.stand,
               "count": count, "prev": prev, "type": str(r.pitch_type), "mph": None if pd.isna(r.release_speed_mph) else round(r.release_speed_mph, 1),
               "desc": d, "event": None if pd.isna(r.events) else str(r.events), "csw": bool(csw), "swing": d in F.SWING,
               "whiff": d in F.WHIFF, "ev": None if pd.isna(r.launch_speed_mph) else round(r.launch_speed_mph, 1),
               "xwoba": None if pd.isna(r.xwoba) else round(float(r.xwoba), 3),
               "rv": None if pd.isna(r.delta_run_exp) else round(float(r.delta_run_exp), 3),
               "px": None if pd.isna(r.plate_x_ft) else round(r.plate_x_ft, 2), "pz": None if pd.isna(r.plate_z_ft) else round(r.plate_z_ft, 2),
               "actual_pred": feats.get((r.at_bat_number, r.pitch_number))}
        if e and count in e["A"]:
            rec["cands"] = {name: {t: [round(v * 100, 1) for v in x] for t, x in cell_of(e, name, count, prev).items()}
                            for name in ["A", "B", "C"]}
            rec["picks"] = picks(e, count, prev, cell_pick.get((r.pitcher, r.stand, count), side_pick.get((r.pitcher, r.stand))),
                                 habit.get((r.pitcher, r.stand, count)))
        pitches.append(rec)
    p = pd.DataFrame(pitches)

    # Summary per picker on this game.
    scored = p[p.picks.notna()].copy()
    summary = {}
    for m in MODELS:
        pick = scored.picks.map(lambda x: x[m])
        has = pick.notna()
        match = (pick == scored.type) & has
        a_pick = [c["A"][pk_][2] if pk_ in c["A"] else np.nan for c, pk_ in zip(scored.cands, pick)]
        summary[m] = {"pitches": int(has.sum()), "matched": int(match.sum()),
                      "csw_matched": float(scored[match].csw.mean()) if match.any() else None,
                      "csw_other": float(scored[has & ~match].csw.mean()) if (has & ~match).any() else None,
                      "a_csw_if_followed": float(np.nanmean(a_pick)) if has.any() else None}
    a_actual = [c["A"].get(t, [np.nan] * 3)[2] for c, t in zip(scored.cands, scored.type)]
    ll = {}
    withp = p[p.actual_pred.notna()]
    for name in ["A", "B", "C"]:
        pr = np.array([x[name][2] / 100 for x in withp.actual_pred])
        ll[name] = float(log_loss(withp.csw.astype(int), np.clip(pr, 1e-4, 1 - 1e-4), labels=[0, 1]))
    pitchers = {}
    for (pid, side), e in g.items():
        pitchers.setdefault(str(int(pid)), {"name": who.get(int(pid), str(pid)), "sides": {}})["sides"][side] = {"types": e["types"], "usage": e["usage"]}
    data = {"game": info, "labels": LABELS, "names": {str(k): v for k, v in who.items()}, "pitchers": pitchers,
            "pitches": pitches, "summary": summary, "a_csw_actual": float(np.nanmean(a_actual)) if len(scored) else None,
            "csw_actual": float(scored.csw.mean()) if len(scored) else None, "game_logloss": ll, "n_scored": int(len(scored)),
            "n_pitches": int(len(p))}

    title = info["series"] + (f" Game {info['game']}" if info["series"] != "Regular Season" else "")
    lines += [f"## Game: {title}, {info['date']}: {info['away']} {info['away_score']} @ "
              f"{info['home']} {info['home_score']}", "",
              f"{len(p)} pitches, {len(scored)} with picks (pitcher has a grid for that side and count). "
              f"Actual CSW on those: {data['csw_actual']:.1%}; model A's expected CSW of the pitches thrown: "
              f"{data['a_csw_actual']:.1f}%.", "",
              "| picker | matched / scored | CSW when matched | CSW otherwise | A's expected CSW if followed |", "|:--|--:|--:|--:|--:|"]
    fmt = lambda x: "-" if x is None else f"{x:.0%}"
    for m in MODELS:
        s = summary[m]
        lines.append(f"| {m}. {LABELS[m]} | {s['matched']} / {s['pitches']} | {fmt(s['csw_matched'])} | {fmt(s['csw_other'])} | "
                     + ("-" if s["a_csw_if_followed"] is None else f"{s['a_csw_if_followed']:.1f}%") + " |")
    lines += ["", f"CSW log-loss on the {len(withp)} game pitches with full features: "
              + ", ".join(f"{k} {v:.4f}" for k, v in ll.items()) + ".", ""]
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=int, default=GAME)
    parser.add_argument("--skip-season", action="store_true")
    args = parser.parse_args()
    df = load([args.game])
    season = df.game_date.astype(str).str[:4].astype(int)
    tr, d26 = df[season == 2025], df[season == 2026]
    cats = {c: df[c].cat.categories for c in Q.CATS}
    models = train(tr)
    cell_pick, side_pick = V.naive_picks()
    habit = habit_picks(d26[~d26._game])
    lines = ["# Model comparison and game report (pre-registered in PREREG_game.md)", "",
             "All models trained on 2025 only. Pickers: " + "; ".join(f"{k} = {v}" for k, v in LABELS.items()) + ".", ""]
    result = {}
    if not args.skip_season:
        lines += ["## Season comparison: matched minus unmatched within-count CSW, 2026", ""]
        g = grids(models, tr, d26[~d26._game], cats, dict(pitches=Q.MIN_PITCHES, side=Q.MIN_SIDE, type=Q.MIN_TYPE))
        result["season"] = season_compare(g, d26, cell_pick, side_pick, habit, lines)
        result["logloss"] = season_logloss(models, d26, lines)
    data = game_report(args.game, models, tr, d26, cats, cell_pick, side_pick, habit, lines)
    data["season"] = result
    PAGE_DIR.joinpath(f"game_{args.game}.json").write_text(json.dumps(data, separators=(",", ":"), default=float))
    out = OUT if not args.skip_season else HERE / f"game_report_{args.game}.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
