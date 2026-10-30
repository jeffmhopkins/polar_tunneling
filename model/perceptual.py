"""
Perceptual tunnel model, as pre-registered in PREREG.md: train on 2025, score
on 2026 once.

Usage:
    python model/perceptual.py           # writes model/perceptual_report.md
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

import features as F
import feasibility as S

OUT = Path(__file__).resolve().parent / "perceptual_report.md"
SEED = 7
N_NULL = 5
TAU0 = 0.175
FLOOR_DEG = 0.1
SWING_PRIOR = 50
BATTER_PRIOR = 100
USAGE_PRIOR = 5
MIN_CLOUD = 20
PS = ["pitcher", "stand"]

FB_STATE = ["rx", "ry", "rz", "rvx", "rvy", "rvz", "ax_ft_s2", "ay_ft_s2", "az_ft_s2", "tf",
            "release_speed_mph", "mov_x", "pfx_z_ft", "rel_x", "release_pos_z_ft"]
DIFF = ["fb_dv", "fb_dmx", "fb_dmz", "fb_drx", "fb_drz"]
CONTROLS = S.BASE + S.PREV + DIFF + ["arm_angle_deg", "bat_whiff_rate", "bat_swing_rate"]
P = {
    "P1 timing": ["tau_b"],
    "P2 fastball-prior miss": ["miss_x_in", "miss_z_in", "miss_t_ms"],
    "P3 optical motion": ["ang_speed", "ang_speed_vs_fb", "sep_rate_vs_fb"],
    "P4 ambiguity": ["post_fb_commit", "post_true_commit", "post_fb_release", "post_true_release"],
    "P5 expectation": ["surprise_count", "surprise_prev"],
    "P6 at-bat build-up": ["fb_seen_pa", "sep_last_fb", "n_priorpa_thisgame", "pitch_number"],
}
ALL_P = [c for cols in P.values() for c in cols]

def eye_of(df):
    return np.stack([np.where(df.stand == "L", -F.EYE_X_RHH, F.EYE_X_RHH),
                     np.full(len(df), F.EYE_Y), np.full(len(df), F.EYE_Z)], axis=1)

def velocity(df, t, prefix=""):
    g = lambda c: df[prefix + c].to_numpy()
    return np.stack([g("rvx") + g("ax_ft_s2") * t, g("rvy") + g("ay_ft_s2") * t, g("rvz") + g("az_ft_s2") * t], axis=1)

def angular_speed(p, v, eye):
    """Speed of the ball's direction from the eye, deg/s."""
    d = p - eye
    r = np.linalg.norm(d, axis=1)
    u = d / r[:, None]
    perp = v - (v * u).sum(1)[:, None] * u
    return np.degrees(np.linalg.norm(perp, axis=1) / r)

def az_el(p, eye):
    d = p - eye
    return np.degrees(np.arctan2(d[:, 0], d[:, 1])), np.degrees(np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1])))

# ---- season-level tables (trajectories and labels only, no outcomes) ----

def primary_fastball(df):
    fb = df[df.pitch_type.isin(F.FASTBALLS)]
    use = fb.groupby(PS + ["pitch_type"], observed=True).size().rename("n").reset_index()
    top = use.sort_values("n").groupby(PS).tail(1)[PS + ["pitch_type"]].rename(columns={"pitch_type": "fb_type"})
    fb = fb.merge(top, on=PS)
    fb = fb[fb.pitch_type == fb.fb_type]
    st = fb.groupby(PS, observed=True)[FB_STATE].mean().add_prefix("fb_").reset_index()
    return st.merge(top, on=PS)

def batter_tau(df):
    sw = df.dropna(subset=["bat_speed_mph", "swing_length_ft"])
    sw = sw[sw.bat_speed_mph > 40]
    ratio = sw.swing_length_ft / sw.bat_speed_mph
    league = ratio.mean()
    g = ratio.groupby(sw.batter).agg(["mean", "size"])
    shrunk = (g["mean"] * g["size"] + league * SWING_PRIOR) / (g["size"] + SWING_PRIOR)
    return (TAU0 * shrunk / league).clip(0.125, 0.225).rename("tau_b")

def batter_rates(df):
    """Batter whiff/swing and swing rates, leaving out the current game."""
    def logo(num, den, prior):
        tot_n = num.groupby(df.batter).transform("sum")
        tot_d = den.groupby(df.batter).transform("sum")
        g_n = num.groupby([df.batter, df.game_pk]).transform("sum")
        g_d = den.groupby([df.batter, df.game_pk]).transform("sum")
        league = num.sum() / den.sum()
        return ((tot_n - g_n) + league * prior) / ((tot_d - g_d) + prior)
    df["bat_whiff_rate"] = logo(df.whiff.astype(float), df.swing.astype(float), BATTER_PRIOR)
    df["bat_swing_rate"] = logo(df.swing.astype(float), pd.Series(1.0, index=df.index), BATTER_PRIOR)
    return df

def posteriors(df, a, e, prefix):
    """Gaussian-mixture posterior over each pitcher x side's pitch types, from angles (a, e) in deg."""
    df["_a"], df["_e"] = a, e
    post_fb = np.full(len(df), np.nan)
    post_true = np.full(len(df), np.nan)
    for _, idx in df.groupby(PS, observed=True).indices.items():
        sub = df.iloc[idx]
        types = sub.pitch_type.value_counts()
        types = types[types >= MIN_CLOUD]
        if len(types) < 2:
            continue
        X = sub[["_a", "_e"]].to_numpy()
        logp = []
        for t in types.index:
            Y = X[(sub.pitch_type == t).to_numpy()]
            mu, cov = Y.mean(0), np.cov(Y.T) + np.eye(2) * FLOOR_DEG**2
            d = X - mu
            inv = np.linalg.inv(cov)
            logp.append(-0.5 * np.einsum("ij,jk,ik->i", d, inv, d) - 0.5 * np.log(np.linalg.det(cov))
                        + np.log(types[t] / types.sum()))
        logp = np.array(logp)
        logp -= logp.max(0)
        post = np.exp(logp) / np.exp(logp).sum(0)
        names = list(types.index)
        fb = sub.fb_type.iloc[0]
        if fb in names:
            post_fb[idx] = post[names.index(fb)]
        own = np.array([names.index(t) if t in names else -1 for t in sub.pitch_type])
        ok = own >= 0
        post_true[idx[ok]] = post[own[ok], np.where(ok)[0]]
    df[f"post_fb_{prefix}"], df[f"post_true_{prefix}"] = post_fb, post_true
    return df.drop(columns=["_a", "_e"])

def surprisal(df, by, name):
    keys = PS + by
    n_type = df.groupby(keys + ["pitch_type"], observed=True).pitch_type.transform("size")
    n_all = df.groupby(keys, observed=True).pitch_type.transform("size")
    base = df.groupby(PS + ["pitch_type"], observed=True).pitch_type.transform("size") / df.groupby(PS, observed=True).pitch_type.transform("size")
    p = (n_type + USAGE_PRIOR * base) / (n_all + USAGE_PRIOR)
    df[name] = -np.log2(p)
    return df

# ---- per-season feature build ----

def commit_features(df):
    """P2 and P3: what the hitter sees at his commit point (needs tau_b and the fb_ columns)."""
    side = np.where(df.stand == "L", -1, 1)
    eye = eye_of(df)
    tc = np.maximum(df.tf.to_numpy() - df.tau_b.to_numpy(), 0.05)
    p, v = F.position(df, tc), velocity(df, tc)

    # P2: extrapolate from commit with the fastball's acceleration.
    a = df[["fb_ax_ft_s2", "fb_ay_ft_s2", "fb_az_ft_s2"]].to_numpy()
    qa, qb, qc = 0.5 * a[:, 1], v[:, 1], p[:, 1] - F.PLATE_FRONT_Y
    s = (-qb - np.sqrt(qb * qb - 4 * qa * qc)) / (2 * qa)
    px = p[:, 0] + v[:, 0] * s + 0.5 * a[:, 0] * s * s
    pz = p[:, 2] + v[:, 2] * s + 0.5 * a[:, 2] * s * s
    df["miss_x_in"] = (df.plate_x_ft - px) * side * 12
    df["miss_z_in"] = (df.plate_z_ft - pz) * 12
    df["miss_t_ms"] = (df.tau_b - s) * 1000

    # P3: optical motion vs the fastball's mean path at the same elapsed time.
    tfb = np.minimum(tc, df.fb_tf.fillna(0).to_numpy())
    pf, vf = F.position(df, tfb, "fb_"), velocity(df, tfb, "fb_")
    df["ang_speed"] = angular_speed(p, v, eye)
    df["ang_speed_vs_fb"] = df.ang_speed - angular_speed(pf, vf, eye)
    h = 0.005
    sep = lambda t, tf_: F.angle_deg(F.position(df, t), F.position(df, np.minimum(t, tf_), "fb_"), eye)
    fbtf = df.fb_tf.fillna(0).to_numpy()
    df["sep_rate_vs_fb"] = (sep(tc + h, fbtf) - sep(tc - h, fbtf)) / (2 * h)
    for c in ["miss_x_in", "miss_z_in", "miss_t_ms", "ang_speed_vs_fb", "sep_rate_vs_fb"]:
        df.loc[df.fb_tf.isna(), c] = np.nan

    return df

def season(year):
    df = F.build(year)
    df = df[df.release_spin_rate_rpm.notna()].copy()
    for c in ["pitch_type", "stand", "prev_pitch_type"]:
        df[c] = df[c].astype(str)
    df = df.merge(primary_fastball(df), on=PS, how="left")
    df = df.merge(batter_tau(df), left_on="batter", right_index=True, how="left")
    df["tau_b"] = df.tau_b.fillna(TAU0)
    df = batter_rates(df)
    df["fb_dv"] = df.release_speed_mph - df.fb_release_speed_mph
    df["fb_dmx"] = df.mov_x - df.fb_mov_x
    df["fb_dmz"] = df.pfx_z_ft - df.fb_pfx_z_ft
    df["fb_drx"] = df.rel_x - df.fb_rel_x
    df["fb_drz"] = df.release_pos_z_ft - df.fb_release_pos_z_ft
    side = np.where(df.stand == "L", -1, 1)
    df["prev_loc_x"] = df.prev_plate_x_ft * side
    df["prev_loc_z"] = (df.prev_plate_z_ft - df.sz_bot_ft) / (df.sz_top_ft - df.sz_bot_ft)
    d = df.prev_description.fillna("")
    df["prev_desc"] = np.select(
        [d.isin(F.WHIFF), d == "called_strike", d.str.startswith("foul"), d.isin({"ball", "blocked_ball"}), d == ""],
        ["whiff", "called", "foul", "ball", "none"], "other")
    df = commit_features(df)
    eye = eye_of(df)

    # P4: Bayesian observer at 175 ms before arrival and at release.
    df = posteriors(df, *az_el(F.position(df, df.tf.to_numpy() - TAU0), eye), "commit")
    df = posteriors(df, *az_el(df[["rx", "ry", "rz"]].to_numpy(), eye), "release")

    # P5: expectation.
    df = surprisal(df, ["count"], "surprise_count")
    df = surprisal(df, ["prev_pitch_type"], "surprise_prev")

    # P6: build-up within the plate appearance.
    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    pa = [df.game_pk, df.at_bat_number]
    is_fb = df.pitch_type.isin(F.FASTBALLS)
    df["fb_seen_pa"] = is_fb.astype(int).groupby(pa).cumsum() - is_fb.astype(int)
    st = ["rx", "ry", "rz", "rvx", "rvy", "rvz", "ax_ft_s2", "ay_ft_s2", "az_ft_s2", "tf"]
    last = df[st].where(is_fb).groupby(pa).shift(1)
    last = last.groupby(pa).ffill().add_prefix("lfb_")
    df = pd.concat([df, last], axis=1)
    tc = np.maximum(df.tf.to_numpy() - df.tau_b.to_numpy(), 0.05)
    tl = np.minimum(tc, df.lfb_tf.fillna(0).to_numpy())
    df["sep_last_fb"] = np.where(df.lfb_tf.notna(), F.angle_deg(F.position(df, tc), F.position(df.fillna({c: 0 for c in last.columns}), tl, "lfb_"), eye_of(df)), np.nan)
    return df

def fit(tr, te, y, cols):
    m = lgb.train(S.PARAMS, lgb.Dataset(tr[cols], tr[y].astype(int)), S.ROUNDS)
    return m, log_loss(te[y].astype(int), m.predict(te[cols]))

def shuffled(df, rng):
    out = df.copy()
    perm = np.arange(len(out))
    for rows in out.groupby(["pitcher", "pitch_type"], observed=True).indices.values():
        perm[rows] = rng.permutation(rows)
    out[ALL_P] = out[ALL_P].to_numpy()[perm]
    return out

def main():
    tr, te = season(2025), season(2026)
    cats = ["pitch_type", "stand", "p_throws", "prev_pitch_type", "prev_desc"]
    for c in cats:
        tr[c] = tr[c].astype(object).fillna("none").astype(str)
        te[c] = te[c].astype(object).fillna("none").astype(str)
        levels = sorted(set(tr[c]) | set(te[c]))
        tr[c] = pd.Categorical(tr[c], categories=levels)
        te[c] = pd.Categorical(te[c], categories=levels)
    lines = ["# Perceptual tunnel model: 2025 -> 2026 (pre-registered, single run)", "",
             f"Train {len(tr):,} pitches (2025), test {len(te):,} (2026).", "",
             "Feature coverage in 2026 (share non-missing): "
             + ", ".join(f"{c} {te[c].notna().mean():.2f}" for c in ALL_P), ""]
    rng = np.random.default_rng(SEED)
    rows, models = [], {}
    for y in ["whiff", "swing"]:
        a, b = (tr[tr.swing], te[te.swing]) if y == "whiff" else (tr, te)
        m0, l0 = fit(a, b, y, CONTROLS)
        m1, l1 = fit(a, b, y, CONTROLS + ALL_P)
        row = {"outcome": "whiff | swing" if y == "whiff" else "swing", "2026 n": len(b),
               "M0 log-loss": round(l0, 4), "M1 gain x1000": round((l0 - l1) * 1000, 2)}
        for name, cols in P.items():
            row[f"+{name}"] = round((l0 - fit(a, b, y, CONTROLS + cols)[1]) * 1000, 2)
        nulls = []
        for _ in range(N_NULL):
            sa, sb = shuffled(a, rng), shuffled(b, rng)
            nulls.append((l0 - fit(sa, sb, y, CONTROLS + ALL_P)[1]) * 1000)
        row["null gains"] = ", ".join(f"{n:.2f}" for n in nulls)
        row["pass"] = (l0 - l1) * 1000 >= 1.0 and (l0 - l1) * 1000 > max(nulls)
        rows.append(row)
        models[y] = (m0, m1)
        print(row, flush=True)
    t = pd.DataFrame(rows).set_index("outcome").T
    lines += ["## Primary endpoints and ablations (log-loss gain x 1000 over M0, 2026)", "", t.to_markdown(), ""]

    # Secondary: pitcher x pitch type P value in one half vs M0 residual in the other.
    m0, m1 = models["whiff"]
    sw = te[te.swing].copy()
    cf = sw.copy()
    cf[ALL_P] = sw.groupby("pitch_type", observed=True)[ALL_P].transform("median")
    sw["pv"] = m1.predict(sw[CONTROLS + ALL_P]) - m1.predict(cf[CONTROLS + ALL_P])
    sw["resid"] = sw.whiff - m0.predict(sw[CONTROLS])
    sw["half"] = np.where(sw.game_pk % 2 == 0, "A", "B")
    g = sw.groupby(["pitcher", "pitch_type", "half"], observed=True).agg(n=("pv", "size"), pv=("pv", "mean"), resid=("resid", "mean")).unstack("half")
    g = g[(g["n"]["A"] >= 75) & (g["n"]["B"] >= 75)]
    r = np.mean([g["pv"]["A"].corr(g["resid"]["B"]), g["pv"]["B"].corr(g["resid"]["A"])])
    lines += ["## Secondary endpoint", "",
              f"{len(g)} pitcher x pitch type groups with >= 75 swings per half. "
              f"Split-half r of P value: {g['pv']['A'].corr(g['pv']['B']):.2f}. "
              f"P value in one half vs M0 residual whiff/swing in the other: r = {r:.3f} (pass if >= 0.10).", ""]
    imp = pd.Series(m1.feature_importance("gain"), index=CONTROLS + ALL_P)
    imp = (imp / imp.sum() * 100).round(2)
    lines += ["## M1 (whiff | swing) gain importance, % of total", "",
              imp.loc[ALL_P].sort_values(ascending=False).to_frame("% gain").to_markdown(), "",
              f"All perceptual features together: {imp.loc[ALL_P].sum():.1f}% of total gain.", ""]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
