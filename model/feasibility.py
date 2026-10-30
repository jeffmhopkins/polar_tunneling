"""
Go/no-go checks for the sequence tunnel score, on 2025.

1. Noise floor / free variation: how big the commit-point separation is for
   back-to-back pitches of the same type vs different types, and how much of it
   is already determined by where the two pitches ended up.
2. Raw signal: whiff/swing, swing% and CSW by deciles of the separations.
3. Incremental lift with a permutation null: does adding the tunnel features to
   a stuff + location + count (+ previous pitch) model improve held-out
   log-loss (train Mar-Jul, test Aug-Oct) by more than shuffled features do?
4. Pitcher-level: is the spread of pitcher x pitch-type tunnel value bigger
   than the null's, and does one half of games predict residual whiffs in the
   other half?

Usage:
    python model/feasibility.py            # writes model/feasibility_report.md
"""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402

OUT = Path(__file__).resolve().parent / "feasibility_report.md"
SEED = 7
N_NULL = 5

BASE = ["pitch_type", "stand", "p_throws", "release_speed_mph", "mov_x", "pfx_z_ft",
        "release_spin_rate_rpm", "release_extension_ft", "release_pos_z_ft", "rel_x",
        "loc_x", "loc_z", "balls", "strikes", "platoon"]
PREV = ["prev_pitch_type", "prev_desc", "prev_loc_x", "prev_loc_z"]
TUNNEL = ["d_dec_deg", "d_track_in", "d_plate_in"]
CATS = ["pitch_type", "stand", "p_throws", "prev_pitch_type", "prev_desc"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=SEED)
ROUNDS = 400

lines = []
def say(s=""):
    print(s, flush=True)
    lines.append(s)

def table(df):
    say(df.to_markdown())
    say()

def prep():
    df = F.build(2025)
    df = df[df.prev_tf.notna()].copy()
    side = np.where(df.stand == "L", -1, 1)
    df["prev_loc_x"] = df.prev_plate_x_ft * side
    df["prev_loc_z"] = (df.prev_plate_z_ft - df.sz_bot_ft) / (df.sz_top_ft - df.sz_bot_ft)
    d = df.prev_description
    df["prev_desc"] = np.select(
        [d.isin(F.WHIFF), d == "called_strike", d.str.startswith("foul"), d.isin({"ball", "blocked_ball"})],
        ["whiff", "called", "foul", "ball"], "other")
    df["same_type"] = df.pitch_type == df.prev_pitch_type
    for c in CATS:
        df[c] = df[c].astype("category")
    return df.reset_index(drop=True)

def check_noise(df):
    say("## 1. Noise floor and free variation")
    say()
    g = df.groupby("same_type")[TUNNEL].quantile([.1, .5, .9]).unstack()
    g.index = ["different type", "same type"]
    g.columns = [f"{a} p{int(b * 100)}" for a, b in g.columns]
    table(g.round(2))
    # How much of d_dec is fixed by the two pitches' types and end locations?
    cross = df[~df.same_type]
    tr = cross.month <= 7
    say("Share of commit-point separation (different-type pairs) predictable from other inputs, held-out R²:")
    say()
    rows = {}
    for name, cols in [("types + both plate locations", ["pitch_type", "prev_pitch_type", "stand", "loc_x", "loc_z", "prev_loc_x", "prev_loc_z"]),
                       ("+ plate separation (d_plate_in)", ["pitch_type", "prev_pitch_type", "stand", "loc_x", "loc_z", "prev_loc_x", "prev_loc_z", "d_plate_in"])]:
        m = lgb.train(dict(PARAMS, objective="regression"), lgb.Dataset(cross.loc[tr, cols], cross.loc[tr, "d_dec_deg"]), ROUNDS)
        rows[name] = r2_score(cross.loc[~tr, "d_dec_deg"], m.predict(cross.loc[~tr, cols]))
    table(pd.Series(rows, name="R²").round(3).to_frame())

def check_raw(df):
    say("## 2. Raw relationship (different-type pairs, no model)")
    say()
    cross = df[~df.same_type].copy()
    for f in ["d_dec_deg", "d_track_in"]:
        cross["bin"] = pd.qcut(cross[f], 10)
        sw = cross[cross.swing]
        t = pd.DataFrame({
            "pitches": cross.groupby("bin", observed=True).size(),
            "swing%": cross.groupby("bin", observed=True).swing.mean() * 100,
            "whiff/swing%": sw.groupby("bin", observed=True).whiff.mean() * 100,
            "CSW%": cross.groupby("bin", observed=True).csw.mean() * 100,
        })
        say(f"**By decile of {f}**")
        say()
        table(t.round(1))
    cross["dec3"] = pd.qcut(cross.d_dec_deg, 3, labels=["early close", "early mid", "early far"])
    cross["late3"] = pd.qcut(cross.d_track_in, 3, labels=["late close", "late mid", "late far"])
    sw = cross[cross.swing]
    say("**Whiff/swing % by early (commit-point) x late (tracking-limit) separation terciles**")
    say()
    table((sw.pivot_table(index="dec3", columns="late3", values="whiff", aggfunc="mean", observed=True) * 100).round(1))

def fit(train, test, y, cols):
    m = lgb.train(PARAMS, lgb.Dataset(train[cols], train[y].astype(int)), ROUNDS)
    return m, log_loss(test[y].astype(int), m.predict(test[cols]))

def shuffled(df, rng):
    """Tunnel features permuted within pitcher x pitch type (keeps each pitch type's spread)."""
    out = df.copy()
    idx = out.groupby(["pitcher", "pitch_type"], observed=True).indices
    perm = np.arange(len(out))
    for rows in idx.values():
        perm[rows] = rng.permutation(rows)
    out[TUNNEL] = out[TUNNEL].to_numpy()[perm]
    return out

def check_lift(df):
    say("## 3. Held-out lift and permutation null (train Mar-Jul, test Aug-Oct)")
    say()
    say(f"Log-loss improvement x 1000 over the base model. Null = tunnel features shuffled within pitcher x pitch type, {N_NULL} draws.")
    say()
    rng = np.random.default_rng(SEED)
    rows, models = [], {}
    for y, data in [("whiff", df[df.swing]), ("swing", df), ("csw", df)]:
        tr, te = data[data.month <= 7], data[data.month >= 8]
        _, base = fit(tr, te, y, BASE)
        _, prev = fit(tr, te, y, BASE + PREV)
        m, full = fit(tr, te, y, BASE + PREV + TUNNEL)
        _, tun_only = fit(tr, te, y, BASE + TUNNEL)
        nulls = []
        for _ in range(N_NULL):
            s = shuffled(data, rng)
            nulls.append(fit(s[s.month <= 7], s[s.month >= 8], y, BASE + PREV + TUNNEL)[1])
        nulls = np.array(nulls)
        rows.append({
            "outcome": {"whiff": "whiff | swing", "swing": "swing", "csw": "CSW"}[y],
            "test n": len(te),
            "base log-loss": round(base, 4),
            "+tunnel (no prev)": round((base - tun_only) * 1000, 2),
            "+prev": round((base - prev) * 1000, 2),
            "+prev +tunnel": round((base - full) * 1000, 2),
            "null +prev +tunnel (mean, max)": f"{(base - nulls).mean() * 1000:.2f}, {(base - nulls).max() * 1000:.2f}",
        })
        models[y] = m
    table(pd.DataFrame(rows).set_index("outcome"))
    return models

def check_pitchers(df, model):
    say("## 4. Pitcher x pitch-type level (whiff | swing)")
    say()
    sw = df[df.swing].copy()
    cols = BASE + PREV + TUNNEL
    # Tunnel value of a swing: model with actual tunnel features minus the same
    # pitch with the features set to its pitch type's median.
    med = sw.groupby("pitch_type", observed=True)[TUNNEL].transform("median")
    cf = sw.copy()
    cf[TUNNEL] = med
    sw["tv"] = model.predict(sw[cols]) - model.predict(cf[cols])
    # Residual whiffs against a model with no tunnel features, fit on the other months.
    sw["resid"] = np.nan
    for a, b in [(sw.month <= 7, sw.month >= 8), (sw.month >= 8, sw.month <= 7)]:
        m = lgb.train(PARAMS, lgb.Dataset(sw.loc[a, BASE + PREV], sw.loc[a, "whiff"].astype(int)), ROUNDS)
        sw.loc[b, "resid"] = sw.loc[b, "whiff"] - m.predict(sw.loc[b, BASE + PREV])
    sw["half"] = np.where(sw.game_pk % 2 == 0, "A", "B")
    g = sw.groupby(["pitcher", "pitch_type", "half"], observed=True).agg(n=("tv", "size"), tv=("tv", "mean"), resid=("resid", "mean")).unstack("half")
    g = g[(g["n"]["A"] >= 75) & (g["n"]["B"] >= 75)]
    r_tv = g["tv"]["A"].corr(g["tv"]["B"])
    r_res = g["resid"]["A"].corr(g["resid"]["B"])
    cross = np.mean([g["tv"]["A"].corr(g["resid"]["B"]), g["tv"]["B"].corr(g["resid"]["A"])])
    sb = lambda r: 2 * r / (1 + r)
    allg = sw.groupby(["pitcher", "pitch_type"], observed=True).agg(n=("tv", "size"), tv=("tv", "mean"))
    allg = allg[allg.n >= 150]
    t = pd.Series({
        "pitcher x type groups (>=75 swings per half)": len(g),
        "SD of mean tunnel value, whiff pct points (>=150 swings)": round(allg.tv.std() * 100, 2),
        "split-half r, tunnel value (Spearman-Brown)": f"{r_tv:.2f} ({sb(r_tv):.2f})",
        "split-half r, residual whiff% (Spearman-Brown)": f"{r_res:.2f} ({sb(r_res):.2f})",
        "tunnel value in one half vs residual whiff% in the other, r": round(cross, 3),
    }, name="value")
    table(t.to_frame())
    say("**Largest pitcher x pitch-type tunnel values (>=150 swings), whiff pct points**")
    say()
    top = allg.sort_values("tv", ascending=False).head(10).copy()
    top["tv"] = (top.tv * 100).round(2)
    table(top)

def main():
    df = prep()
    say("# Tunnel score feasibility (2025, sequence pairs)")
    say()
    say(f"Regular-season pitches with a previous pitch in the same PA: {len(df):,}. "
        f"Commit point {F.TAU * 1000:.0f} ms before the current pitch reaches the plate; "
        f"tracking limit {F.TRACK * 1000:.0f} ms.")
    say()
    check_noise(df)
    check_raw(df)
    models = check_lift(df)
    check_pitchers(df, models["whiff"])
    OUT.write_text("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
