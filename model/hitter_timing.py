"""
Per-hitter decision point, as pre-registered in PREREG_hitter.md (Part 1):
fit each hitter's commit time on 2025, then test on 2026 once.

Usage:
    python model/hitter_timing.py        # writes model/hitter_timing_report.md
                                         # and model/batter_tau_2025.csv
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

import features as F
import perceptual as M
import speed_check as SC

HERE = Path(__file__).resolve().parent
OUT = HERE / "hitter_timing_report.md"
TAUS_OUT = HERE / "batter_tau_2025.csv"
GRID = np.array([0.125, 0.150, 0.175, 0.200, 0.225])
PRIOR_SD = 0.025
N_NULL = 5
SEED = 7
COLS = M.CONTROLS + SC.SPEED + M.ALL_P
CATS = ["pitch_type", "stand", "p_throws", "prev_pitch_type", "prev_desc"]
LFB = ["rx", "ry", "rz", "rvx", "rvy", "rvz", "ax_ft_s2", "ay_ft_s2", "az_ft_s2", "tf"]
TAU_COLS = ["tau_b", "miss_x_in", "miss_z_in", "miss_t_ms", "ang_speed", "ang_speed_vs_fb", "sep_rate_vs_fb",
            "post_fb_commit", "post_true_commit", "sep_last_fb"]

def tag(t):
    return f"c{int(round(t * 1000))}"

def add_grid(df):
    """Commit posteriors at every grid tau (pitch-type clouds as seen by a hitter committing at that tau)."""
    eye = M.eye_of(df)
    for t in GRID:
        df = M.posteriors(df, *M.az_el(F.position(df, df.tf.to_numpy() - t), eye), tag(t))
    return df

def load(year):
    df = add_grid(SC.add_speed(M.season(year)))
    df["bat_tau"] = df.tau_b  # bat-tracking estimate, kept for arm A and the report
    return df

def categorize(*dfs):
    for c in CATS:
        vals = [d[c].astype(object).fillna("none").astype(str).replace("nan", "none") for d in dfs]
        levels = sorted(set().union(*[set(v) for v in vals]))
        for d, v in zip(dfs, vals):
            d[c] = pd.Categorical(v, categories=levels)

def set_tau(df, tau):
    """Recompute every commit-time feature with hitter timing tau (scalar or per row)."""
    tau = np.broadcast_to(np.asarray(tau, dtype=float), (len(df),)).copy()
    df["tau_b"] = tau
    M.commit_features(df)
    t = np.clip(tau, GRID[0], GRID[-1])
    hi = np.clip(np.searchsorted(GRID, t), 1, len(GRID) - 1)
    w = (t - GRID[hi - 1]) / (GRID[hi] - GRID[hi - 1])
    for kind in ["fb", "true"]:
        grid = np.stack([df[f"post_{kind}_{tag(g)}"].to_numpy() for g in GRID], axis=1)
        rows = np.arange(len(df))
        df[f"post_{kind}_commit"] = grid[rows, hi - 1] * (1 - w) + grid[rows, hi] * w
    tc = np.maximum(df.tf.to_numpy() - tau, 0.05)
    lfb = df[["lfb_" + c for c in LFB]].fillna(0)
    tl = np.minimum(tc, lfb.lfb_tf.to_numpy())
    df["sep_last_fb"] = np.where(df.lfb_tf.notna(), F.angle_deg(F.position(df, tc), F.position(lfb, tl, "lfb_"), M.eye_of(df)), np.nan)
    return df

def train(d, y):
    return lgb.train(M.S.PARAMS, lgb.Dataset(d[COLS], d[y].astype(int)), M.S.ROUNDS)

def loglik(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return y * np.log(p) + (1 - y) * np.log(1 - p)

def oof_loglik(df):
    """Per pitch, per grid tau: centered out-of-fold log-likelihood (swing + whiff on swings),
    and the league total at each tau."""
    fold = (df.game_pk % 2).to_numpy()
    sw = df.swing.to_numpy()
    out = np.zeros((len(df), len(GRID)))
    league = []
    for j, t in enumerate(GRID):
        set_tau(df, t)
        ls, lw = np.zeros(len(df)), np.zeros(len(df))
        for k in (0, 1):
            tr, te = fold != k, fold == k
            m = train(df[tr], "swing")
            ls[te] = loglik(df.swing[te].to_numpy(), m.predict(df.loc[te, COLS]))
            m = train(df[tr & sw], "whiff")
            te_sw = te & sw
            lw[te_sw] = loglik(df.whiff[te_sw].to_numpy(), m.predict(df.loc[te_sw, COLS]))
        out[:, j] = (ls - ls.mean()) + np.where(sw, lw - lw[sw].mean(), 0)
        print(f"tau {t:.3f}: swing ll {ls.mean():.5f}, whiff ll {lw[sw].mean():.5f}", flush=True)
        league.append(ls.sum() + lw.sum())
    return out, league

def posterior_tau(c, batter, center):
    prior = -0.5 * ((GRID - center) / PRIOR_SD) ** 2
    s = pd.DataFrame(c, columns=GRID).groupby(batter.to_numpy()).sum()
    lp = s.to_numpy() + prior
    lp -= lp.max(1, keepdims=True)
    w = np.exp(lp)
    w /= w.sum(1, keepdims=True)
    return pd.Series(w @ GRID, index=s.index)

def fit(tr, te, y, cols):
    m = lgb.train(M.S.PARAMS, lgb.Dataset(tr[cols], tr[y].astype(int)), M.S.ROUNDS)
    return log_loss(te[y].astype(int), m.predict(te[cols]))

def score(tr, te):
    out = {}
    for y in ["whiff", "swing"]:
        a, b = (tr[tr.swing], te[te.swing]) if y == "whiff" else (tr, te)
        out[y] = fit(a, b, y, COLS)
    return out

def main():
    tr, te = load(2025), load(2026)
    categorize(tr, te)
    lines = ["# Per-hitter decision point (pre-registered, single run)", "",
             f"Train {len(tr):,} pitches (2025), test {len(te):,} (2026).", ""]

    # Arm A first, while the season() features are untouched.
    base = {"A": score(tr, te)}
    print("A", base["A"], flush=True)

    # Fit on 2025.
    c, league = oof_loglik(tr)
    tau_star = GRID[int(np.argmax(league))]
    fitted = posterior_tau(c, tr.batter, tau_star)
    n = tr.groupby("batter").size()
    odd = (tr.game_pk % 2 == 1).to_numpy()
    half = [posterior_tau(c[m], tr.batter[m], tau_star) for m in (odd, ~odd)]
    nh = [tr.batter[m].value_counts() for m in (odd, ~odd)]
    big = nh[0].index[(nh[0] >= 500)].intersection(nh[1].index[nh[1] >= 500])
    r_half = half[0].reindex(big).corr(half[1].reindex(big))
    bat = tr.groupby("batter").bat_tau.first()
    pd.DataFrame({"tau_fit": fitted, "n_2025": n, "tau_bat": bat}).rename_axis("batter").to_csv(TAUS_OUT)

    lines += ["## Fit on 2025", "",
              "League out-of-fold log-likelihood by tau (higher is better):", "",
              pd.DataFrame({"tau ms": (GRID * 1000).astype(int), "total log-lik": np.round(league, 1),
                            "vs best": np.round(np.array(league) - max(league), 1)}).to_markdown(index=False), "",
              f"League best tau*: {tau_star * 1000:.0f} ms.", "",
              f"Fitted tau across {len(fitted):,} hitters: median {fitted.median() * 1000:.0f} ms, "
              f"10th-90th pct {fitted.quantile(0.1) * 1000:.0f}-{fitted.quantile(0.9) * 1000:.0f} ms; "
              f"hitters with >= 1,000 pitches: {fitted[n >= 1000].quantile(0.1) * 1000:.0f}-"
              f"{fitted[n >= 1000].quantile(0.9) * 1000:.0f} ms.", "",
              f"Split-half reliability (odd vs even games, {len(big)} hitters with >= 500 pitches in each half): r = {r_half:.2f}.", "",
              f"Correlation with bat-tracking tau (hitters with >= 1,000 pitches): "
              f"r = {fitted[n >= 1000].corr(bat.reindex(fitted.index)[n >= 1000]):.2f}.", ""]

    # Arms B, C and nulls, all trained on 2025 and scored on 2026.
    tau_of = lambda d, m: d.batter.map(m).fillna(tau_star).to_numpy()
    set_tau(tr, tau_of(tr, fitted)), set_tau(te, tau_of(te, fitted))
    base["B"] = score(tr, te)
    print("B", base["B"], flush=True)
    set_tau(tr, tau_star), set_tau(te, tau_star)
    base["C"] = score(tr, te)
    print("C", base["C"], flush=True)
    rng = np.random.default_rng(SEED)
    nulls = []
    for _ in range(N_NULL):
        sh = pd.Series(rng.permutation(fitted.to_numpy()), index=fitted.index)
        set_tau(tr, tau_of(tr, sh)), set_tau(te, tau_of(te, sh))
        nulls.append(score(tr, te))
        print("null", nulls[-1], flush=True)

    rows, passes, worse = [], [], []
    for y in ["whiff", "swing"]:
        g = lambda arm: (base["A"][y] - arm[y]) * 1000
        ng = [g(x) for x in nulls]
        ok = g(base["B"]) >= 0.3 and g(base["B"]) > max(ng)
        passes.append(ok)
        worse.append(g(base["B"]) < -0.3)
        rows.append({"outcome": "whiff per swing" if y == "whiff" else "swing", "2026 log-loss A": round(base["A"][y], 5),
                     "B fitted gain x1000": round(g(base["B"]), 2), "C league tau* gain x1000": round(g(base["C"]), 2),
                     "null gains": ", ".join(f"{x:.2f}" for x in ng), "pass": ok})
    part1 = any(passes) and not any(worse)
    lines += ["## 2026 test (gain over arm A, current timing)", "", pd.DataFrame(rows).set_index("outcome").T.to_markdown(), "",
              f"**Part 1 passes: {part1}.**", ""]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
