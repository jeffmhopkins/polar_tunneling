"""
Pitch sequencing tool data: for each pitcher, batter side, count and previous
pitch type, the predicted result of each pitch type he could throw next.

Uses the perceptual model (perceptual.py, validated 2025 -> 2026 and against
speed controls in speed_check.py), retrained on 2025 + 2026. Each candidate
pitch type is represented by up to N_SAMPLES of the pitcher's real 2026
pitches of that type to that side (their actual shapes and locations), placed
into the chosen context:
- count, and the most common previous-pitch result leading into that count
  (0-0 has no previous pitch; every other count does)
- previous pitch = his average pitch of that type to that side
- an average hitter (league tau and batter rates), first time through
- the fastball seen earlier in the PA is the previous pitch when it was one

For every cell the output gives predicted swing %, whiff % (per pitch), CSW %
and the deception lift: CSW with the pitch's look-based features (fastball-
prior miss, optical motion, ambiguity: P2-P4) as they are, minus CSW with
them set to the pitch type's league median. Context features (count
expectation, at-bat build-up) stay as they are, so the lift is about how this
pitch looks, not about the count it is thrown in.

Usage:
    python model/sequencing.py           # writes model/sequencing.json
    python model/sequencing.py --train-years 2025 --out model/sequencing_2025model.json
        # out-of-sample version for validate_sequencer.py: the model, the
        # previous-result lookup and batter medians come from 2025 only;
        # 2026 is used only for pitch shapes, locations and usage (no outcomes)
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests

import features as F
import perceptual as M
import speed_check as SC

OUT = Path(__file__).resolve().parent / "sequencing.json"
MIN_PITCHES = 1000
MIN_SIDE = 150
MIN_TYPE = 40
MIN_USAGE = 0.04
N_SAMPLES = 20
SEED = 7
COLS = M.CONTROLS + SC.SPEED + M.ALL_P
LOOK = M.P["P2 fastball-prior miss"] + M.P["P3 optical motion"] + M.P["P4 ambiguity"]
COUNTS = [f"{b}-{s}" for b in range(4) for s in range(3)]
CATS = ["pitch_type", "stand", "p_throws", "prev_pitch_type", "prev_desc"]
STATE = ["rx", "ry", "rz", "rvx", "rvy", "rvz", "ax_ft_s2", "ay_ft_s2", "az_ft_s2", "tf",
         "plate_x_ft", "plate_z_ft", "release_speed_mph"]

def load():
    df = pd.concat([SC.add_speed(M.season(y)) for y in (2025, 2026)], ignore_index=True)
    for c in CATS:
        df[c] = pd.Categorical(df[c].astype(object).fillna("none").astype(str).replace("nan", "none"))
    return df

def train(df):
    models = {}
    for y, data in [("swing", df), ("whiff", df[df.swing]), ("csw", df)]:
        models[y] = lgb.train(M.S.PARAMS, lgb.Dataset(data[COLS], data[y].astype(int)), M.S.ROUNDS)
        print("trained", y, flush=True)
    return models

def usage_tables(d26):
    """P(type | pitcher, side, count) and P(type | pitcher, side, previous type), smoothed as in perceptual.surprisal."""
    base = d26.groupby(M.PS + ["pitch_type"], observed=True).size() / d26.groupby(M.PS, observed=True).size()
    out = {}
    for name, by in [("count", "count"), ("prev", "prev_pitch_type")]:
        n = d26.groupby(M.PS + [by, "pitch_type"], observed=True).size()
        tot = d26.groupby(M.PS + [by], observed=True).size()
        out[name] = (n, tot)
    return base, out

def surprise(base, tables, name, pid, side, ctx, pt):
    n, tot = tables[name]
    b = base.get((pid, side, pt), 0.0)
    k = n.get((pid, side, ctx, pt), 0)
    t = tot.get((pid, side, ctx), 0)
    return -np.log2((k + M.USAGE_PRIOR * b) / (t + M.USAGE_PRIOR) + 1e-9)

def typical_prev_desc(df):
    """Most common previous-pitch result leading into each count."""
    d = df[df.prev_desc.astype(str) != "none"]
    return d.groupby("count", observed=True).prev_desc.agg(lambda x: x.astype(str).value_counts().index[0]).to_dict()

def names(ids):
    out = {}
    ids = list(ids)
    for i in range(0, len(ids), 100):
        r = requests.get("https://statsapi.mlb.com/api/v1/people",
                         params={"personIds": ",".join(map(str, ids[i:i + 100]))}, timeout=60)
        r.raise_for_status()
        out.update({p["id"]: p["fullName"] for p in r.json()["people"]})
    return out

def build_cells(sub, pid, side, types, base, tables, prev_desc_by_count, medians):
    """Rows for every count x previous type x candidate, from real candidate pitches."""
    rng = np.random.default_rng(SEED + int(pid))
    cands = {t: sub[sub.pitch_type == t].sample(min(N_SAMPLES, (sub.pitch_type == t).sum()), random_state=int(rng.integers(1 << 31)))
             for t in types}
    means = {t: sub[sub.pitch_type == t][STATE].mean() for t in types}
    frames = []
    for count in COUNTS:
        balls, strikes = map(int, count.split("-"))
        # 0-0 is the first pitch of the PA; every other count has a previous pitch.
        for prev in (["none"] if count == "0-0" else types):
            for t in types:
                r = cands[t].copy()
                r["count"], r["balls"], r["strikes"] = count, balls, strikes
                r["pitch_number"] = balls + strikes + 1
                r["n_priorpa_thisgame"] = 1
                r["tau_b"] = M.TAU0
                r["bat_whiff_rate"], r["bat_swing_rate"] = medians["bat_whiff_rate"], medians["bat_swing_rate"]
                if prev == "none":
                    for c in STATE:
                        r["prev_" + c] = np.nan
                    r["prev_pitch_type"], r["prev_desc"] = "none", "none"
                    r["prev_loc_x"] = r["prev_loc_z"] = np.nan
                else:
                    m = means[prev]
                    for c in STATE:
                        r["prev_" + c] = m[c]
                    r["prev_pitch_type"] = prev
                    r["prev_desc"] = prev_desc_by_count.get(count, "ball") if count != "0-0" else "none"
                    sgn = -1 if side == "L" else 1
                    r["prev_loc_x"] = m.plate_x_ft * sgn
                    r["prev_loc_z"] = (m.plate_z_ft - r.sz_bot_ft) / (r.sz_top_ft - r.sz_bot_ft)
                r["prev_dv"] = r.release_speed_mph - r.prev_release_speed_mph
                r["prev_dtf_ms"] = (r.tf - r.prev_tf) * 1000
                prev_fb = prev in F.FASTBALLS
                r["fb_seen_pa"] = int(prev_fb)
                for c in ["rx", "ry", "rz", "rvx", "rvy", "rvz", "ax_ft_s2", "ay_ft_s2", "az_ft_s2", "tf"]:
                    r["lfb_" + c] = means[prev][c] if prev_fb else np.nan
                r["surprise_count"] = surprise(base, tables, "count", pid, side, count, t)
                r["surprise_prev"] = surprise(base, tables, "prev", pid, side, prev, t) if prev != "none" else np.nan
                r["_count"], r["_prev"], r["_type"] = count, prev, t
                frames.append(r)
    cells = pd.concat(frames, ignore_index=True)
    cells = M.commit_features(cells)
    eye = M.eye_of(cells)
    tc = np.maximum(cells.tf.to_numpy() - cells.tau_b.to_numpy(), 0.05)
    has = cells.lfb_tf.notna().to_numpy()
    tl = np.minimum(tc, cells.lfb_tf.fillna(0).to_numpy())
    filled = cells.fillna({"lfb_" + c: 0 for c in ["rx", "ry", "rz", "rvx", "rvy", "rvz", "ax_ft_s2", "ay_ft_s2", "az_ft_s2", "tf"]})
    cells["sep_last_fb"] = np.where(has, F.angle_deg(F.position(cells, tc), F.position(filled, tl, "lfb_"), eye), np.nan)
    return cells

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-years", type=int, nargs="+", default=[2025, 2026])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    df = load()
    season = df.game_date.astype(str).str[:4].astype(int)
    fit_on = df[season.isin(args.train_years)]
    models = train(fit_on)
    d26 = df[season == 2026]
    base, tables = usage_tables(d26)
    prev_desc_by_count = typical_prev_desc(fit_on)
    medians = fit_on[["bat_whiff_rate", "bat_swing_rate"]].median()
    type_median = d26.groupby("pitch_type", observed=True)[LOOK].median()
    counts = d26.groupby("pitcher").size()
    pitchers = counts[counts >= MIN_PITCHES].index
    print(len(pitchers), "pitchers", flush=True)
    out = {"counts": COUNTS, "pitchers": {}}
    who = names(pitchers)
    for i, pid in enumerate(pitchers):
        p = d26[d26.pitcher == pid]
        entry = {"name": who.get(int(pid), str(pid)), "throws": str(p.p_throws.iloc[0]), "n": int(len(p)), "sides": {}}
        for side in ["R", "L"]:
            sub = p[p.stand.astype(str) == side]
            if len(sub) < MIN_SIDE:
                continue
            use = sub.pitch_type.astype(str).value_counts()
            types = [t for t in use.index if use[t] >= MIN_TYPE and use[t] / len(sub) >= MIN_USAGE]
            if not types:
                continue
            sub = sub.copy()
            sub["pitch_type"] = sub.pitch_type.astype(str)
            cells = build_cells(sub, pid, side, types, base, tables, prev_desc_by_count, medians)
            for c in CATS:
                cells[c] = pd.Categorical(cells[c].astype(str), categories=df[c].cat.categories)
            cf = cells.copy()
            cf[LOOK] = type_median.reindex(cells.pitch_type.astype(str)).to_numpy()
            X, Xc = cells[COLS], cf[COLS]
            cells["sw"] = models["swing"].predict(X)
            cells["wh"] = cells.sw * models["whiff"].predict(X)
            cells["csw"] = models["csw"].predict(X)
            cells["lift"] = cells.csw - models["csw"].predict(Xc)
            g = cells.groupby(["_count", "_prev", "_type"], sort=False)[["sw", "wh", "csw", "lift"]].mean()
            grid = {}
            for (count, prev, t), r in g.iterrows():
                grid.setdefault(count, {}).setdefault(prev, {})[t] = [round(r.sw * 100, 1), round(r.wh * 100, 1),
                                                                     round(r.csw * 100, 1), round(r.lift * 100, 1)]
            entry["sides"][side] = {"types": types, "usage": [round(use[t] / len(sub), 3) for t in types], "grid": grid}
        if entry["sides"]:
            out["pitchers"][str(int(pid))] = entry
        if i % 25 == 0:
            print(i, entry["name"], flush=True)
    args.out.write_text(json.dumps(out, separators=(",", ":")))
    print("wrote", args.out, args.out.stat().st_size // 1024, "KB")

if __name__ == "__main__":
    main()
