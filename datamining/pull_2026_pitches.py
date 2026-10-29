"""
Pull 2026 pitch-level Statcast data via pybaseball and write selected columns to CSV.

Intended as the evaluation / hold-out set for the polar tunneling model.
Includes batter, count and context columns for the tunnel score.

Columns (with units where applicable):
- pitcher (MLBAM ID)
- release_pos_x_ft, release_pos_y_ft, release_pos_z_ft
- ax_ft_s2, ay_ft_s2, az_ft_s2
- release_speed_mph
- plate_x_ft, plate_z_ft
- description (per-pitch outcome)
- events (plate-appearance outcome)
- batter, stand, count, catcher, zone, movement, spin, contact quality and
  run value (see add_context.py for the full list)

Usage:
    pip install pybaseball pandas
    python pull_2026_pitches.py
"""

import pandas as pd
from pybaseball import statcast, cache

# Enable caching so large queries can be resumed if interrupted
cache.enable()

# Date range for the 2026 regular season (approximate; adjust as needed)
START_DATE = "2026-03-20"
END_DATE = "2026-10-01"
OUTPUT_PATH = "datamining/pitches_2026.csv"

# Columns needed for trajectory work and identification
COLUMNS = [
    "pitcher",          # MLBAM ID of pitcher
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "ax",
    "ay",
    "az",
    "release_speed",
    "plate_x",
    "plate_z",
    "description",      # per-pitch outcome
    "events",           # plate-appearance outcome (null on non-terminal pitches)
    # Useful extras
    "pitch_type",
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "vx0",
    "vy0",
    "vz0",
    # Batter, count and context for the tunnel score (see add_context.py)
    "batter",
    "stand",
    "p_throws",
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "inning_topbot",
    "on_1b",
    "on_2b",
    "on_3b",
    "fielder_2",
    "game_type",
    "zone",
    "sz_top",
    "sz_bot",
    "pfx_x",
    "pfx_z",
    "release_spin_rate",
    "spin_axis",
    "release_extension",
    "launch_speed",
    "launch_angle",
    "estimated_woba_using_speedangle",
    "woba_value",
    "delta_run_exp",
]

# Rename to include units
RENAME = {
    "release_pos_x": "release_pos_x_ft",
    "release_pos_y": "release_pos_y_ft",
    "release_pos_z": "release_pos_z_ft",
    "ax": "ax_ft_s2",
    "ay": "ay_ft_s2",
    "az": "az_ft_s2",
    "release_speed": "release_speed_mph",
    "plate_x": "plate_x_ft",
    "plate_z": "plate_z_ft",
    "vx0": "vx0_ft_s",
    "vy0": "vy0_ft_s",
    "vz0": "vz0_ft_s",
    "fielder_2": "catcher",
    "sz_top": "sz_top_ft",
    "sz_bot": "sz_bot_ft",
    "pfx_x": "pfx_x_ft",
    "pfx_z": "pfx_z_ft",
    "release_spin_rate": "release_spin_rate_rpm",
    "spin_axis": "spin_axis_deg",
    "release_extension": "release_extension_ft",
    "launch_speed": "launch_speed_mph",
    "launch_angle": "launch_angle_deg",
    "estimated_woba_using_speedangle": "xwoba",
}

def main():
    print(f"Pulling Statcast data from {START_DATE} to {END_DATE}...")
    df = statcast(start_dt=START_DATE, end_dt=END_DATE, verbose=True, parallel=True)

    # Keep only the columns we care about (drop any that are missing)
    available = [c for c in COLUMNS if c in df.columns]
    df = df[available].copy()

    # Basic cleanup
    required = ["pitcher", "release_pos_x", "release_pos_y", "release_pos_z", "ax", "ay", "az"]
    df = df.dropna(subset=[c for c in required if c in df.columns])
    df["pitcher"] = df["pitcher"].astype(int)

    # Add units to column names
    df = df.rename(columns=RENAME)

    print(f"Writing {len(df):,} pitches to {OUTPUT_PATH}")
    df.to_csv(OUTPUT_PATH, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
