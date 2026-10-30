"""
Per-pitch tunnel features against the previous pitch in the same plate appearance.

Each pitch is rebuilt from Statcast's 9-parameter fit and viewed from the
batter's eye (polar_transformations/to_polar.py origins, by `stand`).

Clock: the hitter commits TAU seconds before the *current* pitch reaches the
front of the plate. The previous pitch is compared at the same elapsed time
since its own release (capped at its flight time).

Features (current pitch vs previous pitch):
- d_dec_deg   angular separation from the eye at the commit point
- d_track_in  physical (x, z) separation of the two paths at the depth the
              current pitch reaches TRACK s before arrival (the tracking limit)
- d_plate_in  physical separation at the front of the plate

Usage as a library: build(year, tau=0.175) -> DataFrame of all regular-season
pitches with outcomes, controls and (where a previous pitch exists) features.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "polar_transformations"))
from reconstruct_trajectory import PLATE_FRONT_Y, release_state_from_fit  # noqa: E402

IN = 1 / 12
EYE_Y, EYE_Z, EYE_X_RHH = -28.3 * IN, 65.0 * IN, -35.8 * IN  # RHH stands on -x
TAU = 0.175
TRACK = 0.100

SWING = {"swinging_strike", "swinging_strike_blocked", "foul_tip", "foul", "foul_bunt",
         "hit_into_play", "missed_bunt", "bunt_foul_tip"}
WHIFF = {"swinging_strike", "swinging_strike_blocked", "foul_tip", "missed_bunt"}
FASTBALLS = {"FF", "SI", "FC"}
EXCLUDE = {"PO", "EP", "FA", "CS", "SC", "UN"}  # pitchouts, eephus, position players, unknown

def load(year):
    files = sorted((ROOT / "datamining" / "outputs").glob(f"pitches_{year}_*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[(df.game_type == "R") & ~df.pitch_type.isin(EXCLUDE)]
    need = ["vx0_ft_s", "vy0_ft_s", "vz0_ft_s", "ax_ft_s2", "ay_ft_s2", "az_ft_s2",
            "release_pos_y_ft", "plate_x_ft", "plate_z_ft", "pitch_type", "stand"]
    return df.dropna(subset=need).reset_index(drop=True)

MAX_FLIGHT = 0.6  # s; slower "pitches" are lobs and bad fits

def add_state(df):
    """Release position/velocity (ft, ft/s) and flight time to the front of the plate."""
    (x, y, z), (vx, vy, vz), tf = release_state_from_fit(
        df.release_pos_y_ft.to_numpy(), df.vx0_ft_s.to_numpy(), df.vy0_ft_s.to_numpy(),
        df.vz0_ft_s.to_numpy(), df.ax_ft_s2.to_numpy(), df.ay_ft_s2.to_numpy(),
        df.az_ft_s2.to_numpy(), df.plate_x_ft.to_numpy(), df.plate_z_ft.to_numpy())
    for k, v in dict(rx=x, ry=y, rz=z, rvx=vx, rvy=vy, rvz=vz, tf=tf).items():
        df[k] = np.asarray(v, dtype=float)
    return df

def position(df, t, prefix=""):
    """(n, 3) ball positions at time t after release for the rows' own fits."""
    g = lambda c: df[prefix + c].to_numpy()
    return np.stack([g("rx") + g("rvx") * t + 0.5 * g("ax_ft_s2") * t * t,
                     g("ry") + g("rvy") * t + 0.5 * g("ay_ft_s2") * t * t,
                     g("rz") + g("rvz") * t + 0.5 * g("az_ft_s2") * t * t], axis=1)

def time_at_y(df, y, prefix=""):
    """Time after release at which the rows' fits reach depth y."""
    a = 0.5 * df[prefix + "ay_ft_s2"].to_numpy()
    b = df[prefix + "rvy"].to_numpy()
    c = df[prefix + "ry"].to_numpy() - y
    return (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)

def angle_deg(a, b, eye):
    a, b = a - eye, b - eye
    cos = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

def add_previous(df):
    """Join the previous pitch of the same plate appearance as prev_* columns."""
    keys = ["game_pk", "at_bat_number"]
    cols = ["pitch_number", "pitch_type", "description", "plate_x_ft", "plate_z_ft",
            "rx", "ry", "rz", "rvx", "rvy", "rvz", "ax_ft_s2", "ay_ft_s2", "az_ft_s2", "tf"]
    prev = df[keys + cols].copy()
    prev["pitch_number"] += 1
    prev = prev.rename(columns={c: "prev_" + c for c in cols if c != "pitch_number"})
    return df.merge(prev, on=keys + ["pitch_number"], how="left")

def add_features(df, tau=TAU, track=TRACK):
    has = df.prev_tf.notna().to_numpy()
    eye = np.stack([np.where(df.stand == "L", -EYE_X_RHH, EYE_X_RHH),
                    np.full(len(df), EYE_Y), np.full(len(df), EYE_Z)], axis=1)
    t = np.maximum(df.tf.to_numpy() - tau, 0.0)
    tp = np.minimum(t, df.prev_tf.fillna(0).to_numpy())
    df["d_dec_deg"] = np.where(has, angle_deg(position(df, t), position(df, tp, "prev_"), eye), np.nan)
    # Late separation is compared at equal depth (path, not timing).
    a = position(df, df.tf.to_numpy() - track)
    tp = time_at_y(df, a[:, 1], "prev_")
    b = position(df, np.nan_to_num(tp), "prev_")
    df["d_track_in"] = np.where(has, np.hypot(a[:, 0] - b[:, 0], a[:, 2] - b[:, 2]) * 12, np.nan)
    df["d_plate_in"] = np.hypot(df.plate_x_ft - df.prev_plate_x_ft, df.plate_z_ft - df.prev_plate_z_ft) * 12
    return df

def add_outcomes_and_controls(df):
    d = df.description
    df["swing"] = d.isin(SWING)
    df["whiff"] = d.isin(WHIFF)
    df["csw"] = d.isin(WHIFF | {"called_strike"})
    # Location relative to the batter: + is away from the batter, z scaled to his zone.
    side = np.where(df.stand == "L", -1, 1)
    df["loc_x"] = df.plate_x_ft * side
    df["loc_z"] = (df.plate_z_ft - df.sz_bot_ft) / (df.sz_top_ft - df.sz_bot_ft)
    # Movement mirrored so arm side is + for both hands.
    arm = np.where(df.p_throws == "L", -1, 1)
    df["mov_x"] = df.pfx_x_ft * arm
    df["rel_x"] = df.release_pos_x_ft * arm
    df["platoon"] = (df.stand == df.p_throws).astype(int)
    df["count"] = df.balls.astype(int).astype(str) + "-" + df.strikes.astype(int).astype(str)
    df["month"] = pd.to_datetime(df.game_date).dt.month
    return df

def build(year, tau=None):
    tau = TAU if tau is None else tau
    df = load(year)
    df = add_state(df)
    df = df[df.tf < MAX_FLIGHT].reset_index(drop=True)
    df = add_previous(df)
    df = add_features(df, tau=tau)
    return add_outcomes_and_controls(df)
