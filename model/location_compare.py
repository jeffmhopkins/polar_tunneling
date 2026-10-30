"""
Location in the sequencer, as pre-registered in PREREG_location.md.

Each test pitch is scored in its real context (count, the actual previous
pitch and last fastball in the PA) against every (type, zone) candidate in
the pitcher's 2026 menu. Pickers: LA perceptual and LB standard run-value
models (2025), LZ a 2025 league location chart.

Usage:
    python model/location_compare.py     # writes model/location_report.md and
                                         # report_generation/game_<pk>_location.json
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import damage_sequencer as DS
import game_report as G
import hitter_sequencer as HS
import hitter_timing as H
import perceptual as M
import runs_compare as RC
import sequencing as Q
import validate_sequencer as V

HERE = Path(__file__).resolve().parent
OUT = HERE / "location_report.md"
MODELS_CACHE = Path("/tmp/location_models.pkl")
SEED = 7
N_BOOT = 1000
TEST_SHARE = 0.15
MIN_ZONE = 8
N_PER = 4
ZONES = list(range(1, 10)) + [11, 12, 13, 14]
CHART_MIN = 50
CHUNK = 150_000
SETS = {"LA": RC.SETS["RA"], "LB": RC.SETS["RB"]}
PICKERS = ["LA", "LB", "LZ"]
LABELS = {"LA": "Perceptual, type + zone", "LB": "Standard, type + zone", "LZ": "League location chart"}
PREV_STATE = ["rx", "ry", "rz", "rvx", "rvy", "rvz", "ax_ft_s2", "ay_ft_s2", "az_ft_s2", "tf"]
CONTEXT = (["balls", "strikes", "pitch_number", "n_priorpa_thisgame", "fb_seen_pa", "prev_desc", "prev_loc_x", "prev_loc_z",
            "prev_plate_x_ft", "prev_plate_z_ft"] + ["prev_" + c for c in PREV_STATE] + ["lfb_" + c for c in H.LFB])

def pool(sub, types):
    """Candidate rows: up to N_PER real pitches per (type, zone) with >= MIN_ZONE pitches."""
    rng = np.random.default_rng(SEED + int(sub.pitcher.iloc[0]))
    out = []
    for t in types:
        for z in ZONES:
            s = sub[(sub.pitch_type.astype(str) == t) & (sub.zone == z)]
            if len(s) >= MIN_ZONE:
                out.append(s.sample(N_PER, random_state=int(rng.integers(1 << 31))).assign(_type=t, _zone=z))
    return pd.concat(out, ignore_index=True) if out else None

def score(models, rv, cands, ctx, avg, base, tables, pid, side, cats):
    """Mean E[RV] and CSW per (context row, candidate) for each model set. ctx rows carry the real context."""
    n, k = len(ctx), len(cands)
    r = cands.loc[np.tile(np.arange(k), n)].reset_index(drop=True)
    c = ctx.iloc[np.repeat(np.arange(n), k)].reset_index(drop=True)
    for col in CONTEXT:
        r[col] = c[col].to_numpy()
    r["count"] = c["count"].astype(str).to_numpy()
    r["prev_pitch_type"] = c.prev_pitch_type.astype(str).to_numpy()
    prev_speed = (c.release_speed_mph - c.prev_dv).to_numpy()
    r["prev_dv"] = r.release_speed_mph.to_numpy() - prev_speed
    r["prev_dtf_ms"] = (r.tf - r.prev_tf) * 1000
    r["pitch_type"] = r["_type"]
    for col, by, name in [("surprise_count", "count", "count"), ("surprise_prev", "prev_pitch_type", "prev")]:
        keys = pd.MultiIndex.from_arrays([r[by].astype(str), r._type])
        u = keys.unique()
        val = {(a, b): (np.nan if name == "prev" and a in ("none", "nan") else Q.surprise(base, tables, name, pid, side, a, b)) for a, b in u}
        r[col] = [val[x] for x in keys]
    for col in Q.CATS:
        r[col] = pd.Categorical(r[col].astype(object).fillna("none").astype(str).replace("nan", "none"), categories=cats[col])
    r = HS.apply_hitter(r, pd.DataFrame(np.repeat(avg[HS.BAT_COLS].to_numpy(float)[None], len(r), 0), columns=HS.BAT_COLS))
    r["_ctx"] = np.repeat(np.arange(n), k)
    out = {}
    for name, cols in SETS.items():
        p = {key: m.predict(r[cols]) for key, m in models[name].items()}
        r["erv"], r["csw_p"] = DS.expected_rv(p, r["count"].astype(str), rv), p["csw"]
        out[name] = r.groupby(["_ctx", "_type", "_zone"], sort=True)[["erv", "csw_p"]].mean()
    return out

def league_chart(tr, rv):
    d = tr[tr.zone.notna()].copy()
    d["real"] = DS.realized_rv(d, rv)
    d["plat"] = (d.stand.astype(str) == d.p_throws.astype(str)).astype(int)
    d["pt"] = d.pitch_type.astype(str)
    fine = d.groupby(["pt", "zone", "count", "plat"]).real.agg(["mean", "size"])
    coarse = d.groupby(["pt", "zone", "plat"]).real.mean()
    return fine[fine["size"] >= CHART_MIN]["mean"].to_dict(), coarse.to_dict()

def chart_pick(chart, cands, count, plat):
    fine, coarse = chart
    best, val = None, np.inf
    for t, z in cands:
        v = fine.get((t, float(z), count, plat), coarse.get((t, float(z), plat), np.nan))
        if not np.isnan(v) and v < val:
            best, val = (t, z), v
    return best

def run(models, rv, tr, te, cats, avg, chart, test_ids, exclude_pk, game_pk, mins, only=None):
    """Picks for test rows (index labels in test_ids) and game game_pk, for pitchers with a menu."""
    rest = te[te.game_pk != exclude_pk]
    base, tables = Q.usage_tables(rest)
    rows, game = [], {}
    src = rest if only is None else rest[rest.pitcher.isin(only)]
    for i, (pid, side, sub, types, use) in enumerate(G.sides(src, mins)):
        cands = pool(sub, types)
        if cands is None:
            continue
        mine = te[(te.pitcher == pid) & (te.stand.astype(str) == side)]
        mine = mine[mine.index.isin(test_ids) | (mine.game_pk == game_pk)]
        mine = mine[mine.pitch_type.astype(str).isin(types)]
        mine = mine[(mine["count"].astype(str) == "0-0") | mine.prev_tf.notna()]
        if not len(mine):
            continue
        pairs = list(cands[["_type", "_zone"]].drop_duplicates().itertuples(index=False, name=None))
        step = max(1, CHUNK // len(cands))
        for s in range(0, len(mine), step):
            ctx = mine.iloc[s:s + step]
            sc = score(models, rv, cands, ctx, avg, base, tables, pid, side, cats)
            for j, (idx, x) in enumerate(ctx.iterrows()):
                pk = {}
                for name in SETS:
                    g = sc[name].loc[j]
                    pk[name] = g.erv.idxmin()
                plat = int(side == str(x.p_throws))
                pk["LZ"] = chart_pick(chart, pairs, str(x["count"]), plat)
                rec = dict(pitcher=pid, count=str(x["count"]), pitch_type=str(x.pitch_type), zone=x.zone, csw_hit=int(x.csw),
                           real=x.real, **{f"{k}_t": pk[k][0] if pk[k] else None for k in PICKERS},
                           **{f"{k}_z": pk[k][1] if pk[k] else None for k in PICKERS})
                if x.game_pk == game_pk:
                    g = sc["LA"].loc[j]
                    game[f"{int(x.at_bat_number)}-{int(x.pitch_number)}"] = {
                        "picks": {k: [pk[k][0], int(pk[k][1])] if pk[k] else None for k in PICKERS},
                        "top": [[t, int(z), round(-v * 100, 2)] for (t, z), v in g.erv.nsmallest(5).items()]}
                else:
                    rows.append(rec)
        if i % 25 == 0:
            print(i, pid, side, len(mine), "pitches", len(cands), "candidates", flush=True)
    return pd.DataFrame(rows), game

def season(t, lines):
    t = t.copy()
    for k in PICKERS:
        t["m_" + k] = (t.pitch_type == t[k + "_t"]) & (t.zone == t[k + "_z"])
    rng = np.random.default_rng(SEED)
    V.N_BOOT = N_BOOT
    w = lambda draw, gg: gg.pitcher.map(draw).fillna(0)

    def gap(frame, flag, value):
        gg = V.gap_table(frame.assign(csw=frame[value].astype(float) * 100), flag)
        return gg, V.within_count_gap(gg)[0] / 100

    res = {"n": len(t), "pitchers": int(t.pitcher.nunique())}
    lines += [f"Test pitches: {len(t):,} from {t.pitcher.nunique()} pitchers ({TEST_SHARE:.0%} sample of 2026).", "",
              "## Each picker: thrown the pick (type and zone) minus not, within count", "",
              "| picker | pitches | matched | runs per 100 (95% CI) | CSW pts |", "|:--|--:|--:|:--|--:|"]
    for k in PICKERS:
        s = t[t[k + "_t"].notna()]
        gg, est = gap(s, "m_" + k, "real")
        ci = V.boot(lambda d: V.within_count_gap(gg, w(d, gg))[0] / 100, s.pitcher.unique(), rng)
        csw = gap(s, "m_" + k, "csw_hit")[1]
        res[k] = dict(n=len(s), share=float(s["m_" + k].mean()), runs=est, ci=list(ci), csw=csw)
        lines.append(f"| {k}. {LABELS[k]} | {len(s):,} | {s['m_' + k].mean():.1%} | {est:+.2f} ({ci[0]:+.2f} to {ci[1]:+.2f}) | {csw:+.2f} |")
    lines += ["", "## LA minus each other picker (same pitches)", "",
              "| comparison | same type | same zone | same both | runs per 100 (95% CI) | CSW pts (95% CI) |", "|:--|--:|--:|--:|:--|:--|"]
    for k in PICKERS[1:]:
        s = t[t[k + "_t"].notna()]
        out = {}
        for value in ["real", "csw_hit"]:
            ga, gb = gap(s, "m_LA", value)[0], gap(s, "m_" + k, value)[0]
            diff = lambda wa=None, wb=None: (V.within_count_gap(ga, wa)[0] - V.within_count_gap(gb, wb)[0]) / 100
            ci = V.boot(lambda d: diff(w(d, ga), w(d, gb)), s.pitcher.unique(), rng)
            out[value] = (diff(), list(ci))
        st, sz = (s.LA_t == s[k + "_t"]).mean(), (s.LA_z == s[k + "_z"]).mean()
        sb = ((s.LA_t == s[k + "_t"]) & (s.LA_z == s[k + "_z"])).mean()
        res["LA-" + k] = dict(diff=out["real"][0], ci=out["real"][1], csw=out["csw_hit"][0], csw_ci=out["csw_hit"][1],
                              same_type=float(st), same_zone=float(sz), agree=float(sb))
        tag = " **(primary)**" if k == "LB" else ""
        (r, rc), (c, cc) = out["real"], out["csw_hit"]
        lines.append(f"| LA vs {k}. {LABELS[k]}{tag} | {st:.0%} | {sz:.0%} | {sb:.0%} | {r:+.2f} ({rc[0]:+.2f} to {rc[1]:+.2f}) | "
                     f"{c:+.2f} ({cc[0]:+.2f} to {cc[1]:+.2f}) |")
    ok = res["LA-LB"]["diff"] < 0 and res["LA-LB"]["ci"][1] < 0
    res["primary_pass"] = bool(ok)
    lines += ["", f"Primary (perceptual features help choose type and location): **{ok}**.", ""]
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=int, default=G.GAME)
    args = parser.parse_args()
    s = HS.prep([2025], None, M.TAU0)
    tr, te = DS.add_outcomes(s[2025]), DS.add_outcomes(s[2026])
    cats = {c: tr[c].cat.categories for c in Q.CATS}
    rv = DS.run_values(tr)
    hit25 = HS.batter_table(tr, tr.groupby("batter").bat_tau.first(), M.TAU0)
    avg = HS.average_hitter(hit25, tr, M.TAU0)
    if MODELS_CACHE.exists():
        models = pickle.loads(MODELS_CACHE.read_bytes())
    else:
        models = {name: RC.train(tr, cols) for name, cols in SETS.items()}
        MODELS_CACHE.write_bytes(pickle.dumps(models))
    print("models ready", flush=True)
    chart = league_chart(tr, rv)
    te["real"] = DS.realized_rv(te, rv)
    pool_rows = te[(te.game_pk != args.game) & te.zone.notna()]
    test_ids = pool_rows.sample(frac=TEST_SHARE, random_state=SEED).index
    t, _ = run(models, rv, tr, te, cats, avg, chart, set(test_ids), args.game, -1,
               dict(pitches=Q.MIN_PITCHES, side=Q.MIN_SIDE, type=Q.MIN_TYPE))
    t.to_pickle("/tmp/location_test_table.pkl")
    _, game = run(models, rv, tr, te, cats, avg, chart, set(), args.game, args.game, G.GAME_MIN,
                  only=set(te[te.game_pk == args.game].pitcher))
    lines = ["# Location in the sequencer (pre-registered in PREREG_location.md, single run)", "",
             "Pickers built from 2025 only, average hitter, real previous pitch. "
             + "; ".join(f"{k} = {v}" for k, v in LABELS.items()) + ".", ""]
    res = season(t, lines)
    lines += [f"## Game {args.game}", "", f"{len(game)} pitches with location picks.", ""]
    (G.PAGE_DIR / f"game_{args.game}_location.json").write_text(json.dumps({"labels": LABELS, "season": res, "pitches": game},
                                                                             separators=(",", ":"), default=float))
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
