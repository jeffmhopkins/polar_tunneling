"""
Speed check of the perceptual signal, as pre-registered in PREREG_speed.md:
does the perceptual model's 2026 gain survive stronger speed and timing
controls? Train on 2025, score on 2026 once.

Usage:
    python model/speed_check.py          # writes model/speed_check_report.md
"""

from pathlib import Path

import numpy as np
import pandas as pd

import perceptual as M

OUT = Path(__file__).resolve().parent / "speed_check_report.md"
SPEED = ["plate_speed_mph", "flight_ms", "prev_dv", "prev_dtf_ms", "fb_dtf_ms"]
KIN = ["k_x", "k_y", "k_z", "k_vx", "k_vy", "k_vz"]
ORIGINAL = {"whiff": 1.83, "swing": 5.86}

def add_speed(df):
    v = M.velocity(df, df.tf.to_numpy())
    df["plate_speed_mph"] = np.linalg.norm(v, axis=1) / 1.46667
    df["flight_ms"] = df.tf * 1000
    pa = [df.game_pk, df.at_bat_number]  # season() leaves rows sorted by PA and pitch
    df["prev_dv"] = df.release_speed_mph - df.groupby(pa).release_speed_mph.shift(1)
    df["prev_dtf_ms"] = (df.tf - df.prev_tf) * 1000
    df["fb_dtf_ms"] = (df.tf - df.fb_tf) * 1000
    # Batter-frame kinematics at the commit point (exploratory block).
    tc = np.maximum(df.tf.to_numpy() - df.tau_b.to_numpy(), 0.05)
    p, v = M.F.position(df, tc), M.velocity(df, tc)
    side = np.where(df.stand == "L", -1, 1)
    df["k_x"], df["k_y"], df["k_z"] = p[:, 0] * side, p[:, 1], p[:, 2]
    df["k_vx"], df["k_vy"], df["k_vz"] = v[:, 0] * side, v[:, 1], v[:, 2]
    return df

def main():
    tr, te = add_speed(M.season(2025)), add_speed(M.season(2026))
    for c in ["pitch_type", "stand", "p_throws", "prev_pitch_type", "prev_desc"]:
        tr[c] = tr[c].astype(object).fillna("none").astype(str)
        te[c] = te[c].astype(object).fillna("none").astype(str)
        levels = sorted(set(tr[c]) | set(te[c]))
        tr[c] = pd.Categorical(tr[c], categories=levels)
        te[c] = pd.Categorical(te[c], categories=levels)
    base = M.CONTROLS + SPEED
    rng = np.random.default_rng(M.SEED)
    rows = []
    for y in ["whiff", "swing"]:
        a, b = (tr[tr.swing], te[te.swing]) if y == "whiff" else (tr, te)
        _, l0 = M.fit(a, b, y, base)
        _, l1 = M.fit(a, b, y, base + M.ALL_P)
        gain = (l0 - l1) * 1000
        row = {"outcome": "whiff | swing" if y == "whiff" else "swing", "2026 n": len(b),
               "M0s log-loss": round(l0, 4), "M1s gain x1000": round(gain, 2),
               "share of original gain": f"{gain / ORIGINAL[y]:.0%}"}
        for name, cols in M.P.items():
            row[f"+{name}"] = round((l0 - M.fit(a, b, y, base + cols)[1]) * 1000, 2)
        nulls = []
        for _ in range(M.N_NULL):
            nulls.append((l0 - M.fit(M.shuffled(a, rng), M.shuffled(b, rng), y, base + M.ALL_P)[1]) * 1000)
        row["null gains"] = ", ".join(f"{n:.2f}" for n in nulls)
        row["pass"] = gain >= 1.0 and gain > max(nulls)
        # Exploratory: raw commit-point kinematics in the baseline.
        _, lk = M.fit(a, b, y, base + KIN)
        _, lk1 = M.fit(a, b, y, base + KIN + M.ALL_P)
        row["(expl.) +kinematics over M0s"] = round((l0 - lk) * 1000, 2)
        row["(expl.) +P over M0s+kinematics"] = round((lk - lk1) * 1000, 2)
        _, lk3 = M.fit(a, b, y, base + KIN + M.P["P3 optical motion"])
        row["(expl.) +P3 over M0s+kinematics"] = round((lk - lk3) * 1000, 2)
        rows.append(row)
        print(row, flush=True)
    t = pd.DataFrame(rows).set_index("outcome").T
    OUT.write_text("# Speed check: perceptual signal with stronger speed controls (2025 -> 2026, single run)\n\n"
                   f"Speed controls added to M0: {', '.join(SPEED)}.\n\n"
                   "Log-loss gain x 1000 on 2026. Rows marked (expl.) are exploratory and not part of pass/fail.\n\n"
                   + t.to_markdown() + "\n")
    print(OUT.read_text())

if __name__ == "__main__":
    main()
