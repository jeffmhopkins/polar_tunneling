"""
Group-reference read, as pre-registered in PREREG_group.md: train on 2025,
score on 2026 once.

Usage:
    python model/group_read.py           # writes model/group_report.md
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import features as F
import perceptual as PC

OUT = Path(__file__).resolve().parent / "group_report.md"
CACHE = Path("/tmp/group_read_seasons.pkl")
SEED = 11
N_NULL = 5
G = ["gmiss_x_in", "gmiss_z_in", "gmiss_t_ms", "gspread_in"]
P2 = PC.P["P2 fastball-prior miss"]

def group_features(df):
    """G: the miss from the posterior-weighted read of the pitcher x side's look-alike types."""
    eye = PC.eye_of(df)
    a, e = PC.az_el(F.position(df, df.tf.to_numpy() - PC.TAU0), eye)
    tc = np.maximum(df.tf.to_numpy() - df.tau_b.to_numpy(), 0.05)
    p, v = F.position(df, tc), PC.velocity(df, tc)
    X = np.stack([a, e], axis=1)
    acc = df[["ax_ft_s2", "ay_ft_s2", "az_ft_s2"]].to_numpy()
    out = np.full((len(df), 4), np.nan)
    for _, idx in df.groupby(PC.PS, observed=True).indices.items():
        types = df.pitch_type.iloc[idx]
        n = types.value_counts()
        n = n[n >= PC.MIN_CLOUD]
        if len(n) == 0:
            continue
        x = X[idx]
        logp, ak = [], []
        for t in n.index:
            m = (types == t).to_numpy()
            ak.append(acc[idx[m]].mean(0))
            if len(n) == 1:
                continue
            Y = x[m]
            mu, cov = Y.mean(0), np.cov(Y.T) + np.eye(2) * PC.FLOOR_DEG**2
            d = x - mu
            logp.append(-0.5 * np.einsum("ij,jk,ik->i", d, np.linalg.inv(cov), d)
                        - 0.5 * np.log(np.linalg.det(cov)) + np.log(n[t] / n.sum()))
        if len(n) == 1:
            w = np.ones((1, len(idx)))
        else:
            logp = np.array(logp)
            logp -= logp.max(0)
            w = np.exp(logp) / np.exp(logp).sum(0)
        ak = np.array(ak)                                  # K x 3
        pp, vv = p[idx], v[idx]
        qa = 0.5 * ak[:, 1][:, None]                       # K x n
        qb, qc = vv[:, 1][None, :], (pp[:, 1] - F.PLATE_FRONT_Y)[None, :]
        s = (-qb - np.sqrt(qb * qb - 4 * qa * qc)) / (2 * qa)
        xk = pp[:, 0][None] + vv[:, 0][None] * s + 0.5 * ak[:, 0][:, None] * s * s
        zk = pp[:, 2][None] + vv[:, 2][None] * s + 0.5 * ak[:, 2][:, None] * s * s
        xb, zb, sb = (w * xk).sum(0), (w * zk).sum(0), (w * s).sum(0)
        spread = np.sqrt((w * ((xk - xb) ** 2 + (zk - zb) ** 2)).sum(0))
        out[idx] = np.stack([xb, zb, sb, spread], axis=1)
    side = np.where(df.stand == "L", -1, 1)
    df["gmiss_x_in"] = (df.plate_x_ft - out[:, 0]) * side * 12
    df["gmiss_z_in"] = (df.plate_z_ft - out[:, 1]) * 12
    df["gmiss_t_ms"] = (df.tau_b - out[:, 2]) * 1000
    df["gspread_in"] = out[:, 3] * 12
    return df

def seasons():
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    tr, te = group_features(PC.season(2025)), group_features(PC.season(2026))
    cats = ["pitch_type", "stand", "p_throws", "prev_pitch_type", "prev_desc"]
    for c in cats:
        tr[c] = tr[c].astype(object).fillna("none").astype(str)
        te[c] = te[c].astype(object).fillna("none").astype(str)
        levels = sorted(set(tr[c]) | set(te[c]))
        tr[c] = pd.Categorical(tr[c], categories=levels)
        te[c] = pd.Categorical(te[c], categories=levels)
    CACHE.write_bytes(pickle.dumps((tr, te)))
    return tr, te

def shuffled(df, rng):
    out = df.copy()
    perm = np.arange(len(out))
    for rows in out.groupby(["pitcher", "pitch_type"], observed=True).indices.values():
        perm[rows] = rng.permutation(rows)
    out[G] = out[G].to_numpy()[perm]
    return out

def main():
    tr, te = seasons()
    C, M1 = PC.CONTROLS, PC.CONTROLS + PC.ALL_P
    swap = [c for c in M1 if c not in P2] + G
    lines = ["# Group-reference read: 2025 -> 2026 (pre-registered, single run)", "",
             f"Train {len(tr):,} pitches (2025), test {len(te):,} (2026).", "",
             "Feature coverage in 2026 (share non-missing): "
             + ", ".join(f"{c} {te[c].notna().mean():.2f}" for c in G + P2), "",
             "Median in 2026 by pitch type (swings), group read vs fastball read:", ""]
    sw = te[te.swing]
    med = sw.assign(abs_gmiss_in=np.hypot(sw.gmiss_x_in, sw.gmiss_z_in), abs_fbmiss_in=np.hypot(sw.miss_x_in, sw.miss_z_in))
    med = med.groupby("pitch_type", observed=True).agg(n=("whiff", "size"), abs_fbmiss_in=("abs_fbmiss_in", "median"),
                                                       abs_gmiss_in=("abs_gmiss_in", "median"), gspread_in=("gspread_in", "median"),
                                                       whiff_pct=("whiff", "mean"))
    med = med[med.n >= 1000]
    med["whiff_pct"] *= 100
    lines += [med.round(2).to_markdown(), ""]
    rng = np.random.default_rng(SEED)
    rows = []
    for y in ["whiff", "swing"]:
        a, b = (tr[tr.swing], te[te.swing]) if y == "whiff" else (tr, te)
        l1 = PC.fit(a, b, y, M1)[1]
        lg = PC.fit(a, b, y, M1 + G)[1]
        row = {"outcome": "whiff \\| swing" if y == "whiff" else "swing", "2026 n": len(b),
               "M1 log-loss": round(l1, 4), "M1 + G gain x1000": round((l1 - lg) * 1000, 2)}
        if y == "whiff":
            l0 = PC.fit(a, b, y, C)[1]
            row["M0 + P2 gain x1000 (over M0)"] = round((l0 - PC.fit(a, b, y, C + P2)[1]) * 1000, 2)
            row["M0 + G gain x1000 (over M0)"] = round((l0 - PC.fit(a, b, y, C + G)[1]) * 1000, 2)
            row["M1 with G for P2 gain x1000"] = round((l1 - PC.fit(a, b, y, swap)[1]) * 1000, 2)
        nulls = []
        for _ in range(N_NULL):
            sa, sb = shuffled(a, rng), shuffled(b, rng)
            nulls.append((l1 - PC.fit(sa, sb, y, M1 + G)[1]) * 1000)
        row["null gains"] = ", ".join(f"{n:.2f}" for n in nulls)
        row["pass"] = (l1 - lg) * 1000 >= 0.3 and (l1 - lg) * 1000 > max(nulls)
        rows.append(row)
        print(row, flush=True)
    t = pd.DataFrame(rows).set_index("outcome").T
    lines += ["## Endpoints (log-loss gain x 1000, 2026)", "",
              "Primary: whiff | swing, M1 + G over M1. Pass needs >= 0.3 and above all 5 nulls. "
              "The swing column and the head-to-head and swap rows are secondary.", "", t.to_markdown(), ""]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
