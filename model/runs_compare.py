"""
Model comparison by runs, as pre-registered in PREREG_runs.md.

Pickers (2025 only, average hitter): RA perceptual by runs, RB standard by
runs, RC stuff-only by runs, A perceptual by CSW, D best pitch for the
count, E pitcher's habit. Season test on 2026 and picks for one game.

Usage:
    python model/runs_compare.py     # writes model/runs_report.md and
                                     # report_generation/game_<pk>_runs.json
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import damage_sequencer as DS
import game_report as G
import hitter_sequencer as HS
import perceptual as M
import sequencing as Q
import validate_sequencer as V

HERE = Path(__file__).resolve().parent
OUT = HERE / "runs_report.md"
SEED = 7
N_BOOT = 1000
SETS = {"RA": HS.COLS_H, "RB": [c for c in HS.COLS_H if c not in M.ALL_P], "RC": G.COLS_C}
PUBLIC = {"FS": "sp_s_", "FP": "sp_p_", "FB": "pb_o_"}
PICKERS = ["RA", "RB", "RC", "A", "D", "E"] + list(PUBLIC)
LABELS = {"RA": "Perceptual, by runs", "RB": "Standard, by runs", "RC": "Stuff-only, by runs",
          "A": "Perceptual, by CSW", "D": "Best pitch for the count", "E": "Pitcher's habit",
          "FS": "FanGraphs Stuff+ (2025)", "FP": "FanGraphs Pitching+ (2025)", "FB": "PitchingBot (2025)"}
FG_TYPE = {"ST": "SL", "SV": "SL"}
GRADES = pd.read_csv(HERE.parent / "datamining" / "outputs" / "fangraphs_grades_2025.csv").set_index("pitcher")

def public_pick(pid, types, prefix):
    """Highest 2025 FanGraphs grade among his menu types; None if fewer than two are graded."""
    if pid not in GRADES.index:
        return None
    row = GRADES.loc[pid]
    g = {t: row.get(prefix + FG_TYPE.get(t, t)) for t in types}
    g = {t: v for t, v in g.items() if pd.notna(v)}
    return max(g, key=g.get) if len(g) >= 2 else None

def train(d, cols):
    sets = [("swing", d, "swing", "binary"), ("whiff", d[d.swing], "whiff", "binary"),
            ("foul", d[d.contact], "foul", "binary"), ("cs", d[~d.swing], "cs", "binary"),
            ("csw", d, "csw", "binary"), ("xw", d[d.bip & d.xwoba.notna()], "xwoba", "regression")]
    return {name: lgb.train(dict(M.S.PARAMS, objective=obj), lgb.Dataset(data[cols], data[y].astype(float)), M.S.ROUNDS)
            for name, data, y, obj in sets}

def grids(models, rv, tr, d26, cats, avg, mins, only=None):
    """{(pid, side): {"types", name: {count: {prev: {type: [csw, E[RV]]}}}}} for RA, RB, RC."""
    base, tables = Q.usage_tables(d26)
    prev_desc = Q.typical_prev_desc(tr)
    medians = pd.Series({"bat_whiff_rate": avg.bat_whiff_rate, "bat_swing_rate": avg.bat_swing_rate})
    out = {}
    src = d26 if only is None else d26[d26.pitcher.isin(only)]
    for i, (pid, side, sub, types, use) in enumerate(G.sides(src, mins)):
        cells = Q.build_cells(sub, pid, side, types, base, tables, prev_desc, medians)
        for c in Q.CATS:
            cells[c] = pd.Categorical(cells[c].astype(str), categories=cats[c])
        cells = HS.apply_hitter(cells, pd.DataFrame(np.repeat(avg[HS.BAT_COLS].to_numpy(float)[None], len(cells), 0), columns=HS.BAT_COLS))
        entry = {"types": types}
        for name, cols in SETS.items():
            p = {k: m.predict(cells[cols]) for k, m in models[name].items()}
            cells["csw_p"], cells["erv"] = p["csw"], DS.expected_rv(p, cells["count"].astype(str), rv)
            g = cells.groupby(["_count", "_prev", "_type"], sort=False)[["csw_p", "erv"]].mean()
            grid = {}
            for (count, prev, t), r in g.iterrows():
                grid.setdefault(count, {}).setdefault(prev, {})[t] = [r.csw_p, r.erv]
            entry[name] = grid
        out[(pid, side)] = entry
        if i % 50 == 0:
            print(i, pid, side, flush=True)
    return out

def picks(entry, count, prev, naive, habit, pid=None):
    out = {k: public_pick(pid, entry["types"], pre) for k, pre in PUBLIC.items()}
    for name in SETS:
        cell = G.cell_of(entry, name, count, prev)
        out[name] = min(cell, key=lambda t: cell[t][1])
        if name == "RA":
            out["A"] = max(cell, key=lambda t: cell[t][0])
    out["D"], out["E"] = naive, habit
    return out

def season(g, te, rv, cell_pick, side_pick, habit, lines):
    te = te.copy()
    te["prev"] = np.where(te["count"] == "0-0", "none", te.prev_pitch_type.astype(str))
    te["real"] = DS.realized_rv(te, rv)
    rows = []
    for r in te[["pitcher", "stand", "count", "prev", "pitch_type", "csw", "real", "delta_run_exp"]].itertuples(index=False):
        e = g.get((r.pitcher, str(r.stand)))
        pt = str(r.pitch_type)
        if not e or r.prev not in e["RA"].get(r.count, {}) or pt not in e["types"]:
            continue
        p = picks(e, r.count, r.prev, cell_pick.get((r.pitcher, str(r.stand), r.count), side_pick.get((r.pitcher, str(r.stand)))),
                  habit.get((r.pitcher, str(r.stand), r.count)), r.pitcher)
        rows.append((r.pitcher, r.count, pt, int(r.csw), r.real, r.delta_run_exp, *[p[k] for k in PICKERS]))
    t = pd.DataFrame(rows, columns=["pitcher", "count", "pitch_type", "csw_hit", "real", "dre"] + PICKERS)
    for k in PICKERS:
        t["m_" + k] = t.pitch_type == t[k]
    rng = np.random.default_rng(SEED)
    V.N_BOOT = N_BOOT
    w = lambda draw, gg: gg.pitcher.map(draw).fillna(0)

    def gap(frame, flag, value, scale):
        f = frame.assign(csw=frame[value].astype(float) * scale)
        gg = V.gap_table(f, flag)
        return gg, V.within_count_gap(gg)[0] / 100

    res = {"n": len(t), "pitchers": int(t.pitcher.nunique())}
    lines += [f"Test pitches: {len(t):,} from {t.pitcher.nunique()} pitchers (2026 regular season).", "",
              "## Each picker: thrown the pick minus not, within count", "",
              "Runs per 100 pitches (negative is better) and CSW pts (positive is better).", "",
              "| picker | pitches | share thrown | runs per 100 (95% CI) | CSW pts |", "|:--|--:|--:|:--|--:|"]
    for k in PICKERS:
        s = t[t[k].notna()]
        gg, est = gap(s, "m_" + k, "real", 100)
        ci = V.boot(lambda d: V.within_count_gap(gg, w(d, gg))[0] / 100, s.pitcher.unique(), rng)
        csw = gap(s, "m_" + k, "csw_hit", 100)[1]
        res[k] = dict(n=len(s), share=float(s["m_" + k].mean()), runs=est, ci=list(ci), csw=csw)
        lines.append(f"| {k}. {LABELS[k]} | {len(s):,} | {s['m_' + k].mean():.1%} | {est:+.2f} ({ci[0]:+.2f} to {ci[1]:+.2f}) | {csw:+.2f} |")
    lines += ["", "## RA minus each other picker (same pitches)", "",
              "| comparison | picks agree | runs per 100 (95% CI) | actual delta_run_exp per 100 | CSW pts |", "|:--|--:|:--|--:|--:|"]
    for k in PICKERS[1:]:
        s = t[t[k].notna()]
        ga, gb = gap(s, "m_RA", "real", 100)[0], gap(s, "m_" + k, "real", 100)[0]
        diff = lambda wa=None, wb=None: (V.within_count_gap(ga, wa)[0] - V.within_count_gap(gb, wb)[0]) / 100
        est = diff()
        ci = V.boot(lambda d: diff(w(d, ga), w(d, gb)), s.pitcher.unique(), rng)
        sd = s[s.dre.notna()]
        dre = gap(sd, "m_RA", "dre", 100)[1] - gap(sd, "m_" + k, "dre", 100)[1]
        csw = gap(s, "m_RA", "csw_hit", 100)[1] - gap(s, "m_" + k, "csw_hit", 100)[1]
        res["RA-" + k] = dict(diff=est, ci=list(ci), agree=float((s.RA == s[k]).mean()), dre=dre, csw=csw)
        tag = " **(primary)**" if k == "RB" else ""
        lines.append(f"| RA vs {k}. {LABELS[k]}{tag} | {(s.RA == s[k]).mean():.0%} | {est:+.2f} ({ci[0]:+.2f} to {ci[1]:+.2f}) | "
                     f"{dre:+.2f} | {csw:+.2f} |")
    lines += ["", "## Perceptual CSW pick (A) minus the public picks, on CSW", "",
              "| comparison | picks agree | CSW pts (95% CI) |", "|:--|--:|:--|"]
    for k in PUBLIC:
        s = t[t[k].notna()]
        ga, gb = gap(s, "m_A", "csw_hit", 100)[0], gap(s, "m_" + k, "csw_hit", 100)[0]
        diff = lambda wa=None, wb=None: (V.within_count_gap(ga, wa)[0] - V.within_count_gap(gb, wb)[0]) / 100
        est = diff()
        ci = V.boot(lambda d: diff(w(d, ga), w(d, gb)), s.pitcher.unique(), rng)
        res["A-" + k] = dict(diff=est, ci=list(ci), agree=float((s.A == s[k]).mean()))
        lines.append(f"| A vs {k}. {LABELS[k]} | {(s.A == s[k]).mean():.0%} | {est:+.2f} ({ci[0]:+.2f} to {ci[1]:+.2f}) |")
    t.to_pickle("/tmp/runs_test_table.pkl")
    ok = res["RA-RB"]["diff"] < 0 and res["RA-RB"]["ci"][1] < 0
    res["primary_pass"] = bool(ok)
    lines += ["", f"Primary (perceptual features help pick by runs): **{ok}**.", ""]
    return res

def game(pk, models, rv, tr, d26, cats, avg, cell_pick, side_pick, habit, lines):
    gm = d26[d26.game_pk == pk]
    g = grids(models, rv, tr, d26[d26.game_pk != pk], cats, avg, G.GAME_MIN, only=set(gm.pitcher))
    raw = pd.concat([pd.read_csv(f) for f in sorted((G.F.ROOT / "datamining" / "outputs").glob("pitches_2026_*.csv"))])
    raw = raw[raw.game_pk == pk].sort_values(["at_bat_number", "pitch_number"]).reset_index(drop=True)
    raw["prev_type"] = raw.groupby("at_bat_number").pitch_type.shift(1)
    out = {}
    for r in raw.itertuples(index=False):
        count = f"{int(r.balls)}-{int(r.strikes)}"
        prev = "none" if count == "0-0" else str(r.prev_type)
        e = g.get((r.pitcher, r.stand))
        if not e or count not in e["RA"]:
            continue
        cell = G.cell_of(e, "RA", count, prev)
        p = picks(e, count, prev, cell_pick.get((r.pitcher, r.stand, count), side_pick.get((r.pitcher, r.stand))),
                  habit.get((r.pitcher, r.stand, count)), r.pitcher)
        out[f"{int(r.at_bat_number)}-{int(r.pitch_number)}"] = {
            "picks": {k: p[k] for k in ["RA", "RB", "RC"] + list(PUBLIC)},
            "rs": {t: round(-v[1] * 100, 2) for t, v in cell.items()}}
    rows = []
    for k in ["RA", "RB", "RC"]:
        n = sum(1 for key, v in out.items())
        rows.append(f"{k} {sum(1 for key, v in out.items() if v['picks'][k] == raw.set_index(raw.at_bat_number.astype(str) + '-' + raw.pitch_number.astype(str)).pitch_type.get(key))} of {n}")
    lines += [f"## Game {pk}", "", "Pitches where the pitcher threw each runs picker's choice: " + ", ".join(rows) + ".", ""]
    return out

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
    models = {}
    for name, cols in SETS.items():
        models[name] = train(tr, cols)
        print("trained", name, flush=True)
    cell_pick, side_pick = V.naive_picks()
    rest = te[te.game_pk != args.game]
    habit = G.habit_picks(rest)
    lines = ["# Model comparison by runs (pre-registered in PREREG_runs.md, single run)", "",
             "All pickers built from 2025 only, average hitter. " + "; ".join(f"{k} = {v}" for k, v in LABELS.items()) + ".", ""]
    g = grids(models, rv, tr, rest, cats, avg, dict(pitches=Q.MIN_PITCHES, side=Q.MIN_SIDE, type=Q.MIN_TYPE))
    res = season(g, rest, rv, cell_pick, side_pick, habit, lines)
    per_pitch = game(args.game, models, rv, tr, te, cats, avg, cell_pick, side_pick, habit, lines)
    (G.PAGE_DIR / f"game_{args.game}_runs.json").write_text(json.dumps({"labels": LABELS, "season": res, "pitches": per_pitch},
                                                                         separators=(",", ":"), default=float))
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
