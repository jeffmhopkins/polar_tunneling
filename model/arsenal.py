"""
Arsenal tunnel check: does a pitch type that tunnels well off the pitcher's
primary fastball do better than its stuff, location and fastball differentials
predict? This is the explorer's comparison, tested across pitchers.

For each pitcher x pitch type x batter side, from training months (Mar-Jul) only:
- primary fastball = highest-usage FF/SI/FC to that side
- average trajectory of each type (mean 9-parameter state)
- a_dec_deg  angle from the eye between the type and the fastball at the
             commit point (TAU before the fastball arrives, same elapsed time)
- a_track_in path separation at the tracking limit (equal depth)
- a_plate_in separation at the plate
Fastball differentials (speed, movement, release) go in the controls.

Test months (Aug-Oct): held-out log-loss gain of +tunnel over the controls,
against a null that shuffles the tunnel features between groups of the same
pitch type.

Usage:
    python model/arsenal.py            # writes model/arsenal_report.md
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

import features as F
import feasibility as S

OUT = Path(__file__).resolve().parent / "arsenal_report.md"
N_NULL = 10
MIN_N = 50
STATE = ["rx", "ry", "rz", "rvx", "rvy", "rvz", "ax_ft_s2", "ay_ft_s2", "az_ft_s2", "tf",
         "plate_x_ft", "plate_z_ft", "release_speed_mph", "mov_x", "pfx_z_ft", "rel_x", "release_pos_z_ft"]
DIFF = ["fb_dv", "fb_dmx", "fb_dmz", "fb_drx", "fb_drz"]
ARS = ["a_dec_deg", "a_track_in", "a_plate_in"]

def group_table(train):
    keys = ["pitcher", "stand", "pitch_type"]
    g = train.groupby(keys, observed=True)[STATE].mean()
    g["n"] = train.groupby(keys, observed=True).size()
    g = g[g.n >= MIN_N].reset_index()
    g["usage"] = g.n / g.groupby(["pitcher", "stand"]).n.transform("sum")
    fb = g[g.pitch_type.isin(F.FASTBALLS)].sort_values("usage").groupby(["pitcher", "stand"]).tail(1)
    fb = fb[["pitcher", "stand", "pitch_type"] + STATE].rename(columns={c: "prev_" + c for c in STATE + ["pitch_type"]})
    g = g.merge(fb, on=["pitcher", "stand"])
    g = g[g.pitch_type != g.prev_pitch_type].reset_index(drop=True)
    # Reuse the sequence-feature code with the fastball as the "previous" pitch,
    # but on the fastball's clock: commit TAU before the fastball arrives.
    eye = np.stack([np.where(g.stand == "L", -F.EYE_X_RHH, F.EYE_X_RHH),
                    np.full(len(g), F.EYE_Y), np.full(len(g), F.EYE_Z)], axis=1)
    t = np.maximum(g.prev_tf.to_numpy() - F.TAU, 0)
    g["a_dec_deg"] = F.angle_deg(F.position(g, np.minimum(t, g.tf.to_numpy())), F.position(g, t, "prev_"), eye)
    a = F.position(g, g.tf.to_numpy() - F.TRACK)
    b = F.position(g, F.time_at_y(g, a[:, 1], "prev_"), "prev_")
    g["a_track_in"] = np.hypot(a[:, 0] - b[:, 0], a[:, 2] - b[:, 2]) * 12
    g["a_plate_in"] = np.hypot(g.plate_x_ft - g.prev_plate_x_ft, g.plate_z_ft - g.prev_plate_z_ft) * 12
    g["fb_dv"] = g.release_speed_mph - g.prev_release_speed_mph
    g["fb_dmx"] = g.mov_x - g.prev_mov_x
    g["fb_dmz"] = g.pfx_z_ft - g.prev_pfx_z_ft
    g["fb_drx"] = g.rel_x - g.prev_rel_x
    g["fb_drz"] = g.release_pos_z_ft - g.prev_release_pos_z_ft
    return g[["pitcher", "stand", "pitch_type", "n"] + ARS + DIFF]

def fit(tr, te, y, cols):
    m = lgb.train(S.PARAMS, lgb.Dataset(tr[cols], tr[y].astype(int)), S.ROUNDS)
    return log_loss(te[y].astype(int), m.predict(te[cols]))

def main():
    df = F.build(2025)
    df = df[df.release_spin_rate_rpm.notna()]
    g = group_table(df[df.month <= 7])
    for c in ["pitch_type", "stand"]:
        df[c] = df[c].astype(str)
        g[c] = g[c].astype(str)
    d = df.merge(g.drop(columns="n"), on=["pitcher", "stand", "pitch_type"])
    for c in ["pitch_type", "stand", "p_throws"]:
        d[c] = d[c].astype("category")
    lines = ["# Arsenal tunnel check (2025, vs primary fastball)", "",
             f"{len(g):,} pitcher x pitch type x side groups (>= {MIN_N} pitches Mar-Jul), "
             f"{len(d):,} pitches. Group features from Mar-Jul; test on Aug-Oct.", ""]
    rng = np.random.default_rng(S.SEED)
    rows = []
    for y, data in [("whiff", d[d.swing]), ("swing", d), ("csw", d)]:
        tr, te = data[data.month <= 7], data[data.month >= 8]
        base = fit(tr, te, y, S.BASE + DIFF)
        full = fit(tr, te, y, S.BASE + DIFF + ARS)
        nulls = []
        for _ in range(N_NULL):
            # Shuffle the tunnel features between groups of the same pitch type.
            gs = g.copy()
            for _, idx in gs.groupby("pitch_type").groups.items():
                gs.loc[idx, ARS] = gs.loc[rng.permutation(idx), ARS].to_numpy()
            s = data.drop(columns=ARS).merge(gs[["pitcher", "stand", "pitch_type"] + ARS].astype({"stand": "category", "pitch_type": "category"}),
                                             on=["pitcher", "stand", "pitch_type"])
            nulls.append(fit(s[s.month <= 7], s[s.month >= 8], y, S.BASE + DIFF + ARS))
        ng = (base - np.array(nulls)) * 1000
        gain = (base - full) * 1000
        rows.append({"outcome": {"whiff": "whiff | swing", "swing": "swing", "csw": "CSW"}[y],
                     "test n": len(te), "gain x1000": round(gain, 2), "null mean": round(ng.mean(), 2),
                     "null max": round(ng.max(), 2), "share of nulls beaten": f"{(gain > ng).mean():.0%}"})
        print(rows[-1], flush=True)
    lines += ["Log-loss gain x 1000 of +arsenal tunnel over stuff + location + count + fastball differentials:", "",
              pd.DataFrame(rows).to_markdown(index=False), ""]

    # Group level: residual whiff/swing in Aug-Oct vs Mar-Jul tunnel features.
    sw = d[d.swing]
    tr, te = sw[sw.month <= 7], sw[sw.month >= 8].copy()
    m = lgb.train(S.PARAMS, lgb.Dataset(tr[S.BASE + DIFF], tr.whiff.astype(int)), S.ROUNDS)
    te["resid"] = te.whiff - m.predict(te[S.BASE + DIFF])
    r = te.groupby(["pitcher", "stand", "pitch_type"], observed=True).agg(n=("resid", "size"), resid=("resid", "mean"))
    r = r[r.n >= 40].join(g.set_index(["pitcher", "stand", "pitch_type"])[ARS])
    corr = pd.Series({f: r.resid.corr(r[f], method="spearman") for f in ARS}, name="Spearman r")
    lines += [f"Group level ({len(r)} groups with >= 40 Aug-Oct swings): rank correlation of Mar-Jul tunnel features "
              "with Aug-Oct residual whiff/swing (tunneling predicts negative for a_dec_deg):", "",
              corr.round(3).to_frame().to_markdown(), ""]
    r["dec_q"] = pd.qcut(r.a_dec_deg, 4, labels=["closest", "2", "3", "farthest"])
    q = r.groupby("dec_q", observed=True).agg(groups=("resid", "size"), resid_whiff_pct=("resid", lambda x: x.mean() * 100))
    lines += ["Mean residual whiff/swing (pct points) by quartile of commit-point angle to the fastball:", "",
              q.round(2).to_markdown(), ""]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
