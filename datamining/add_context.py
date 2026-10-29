"""
Add the batter, count, zone, movement, contact-quality and run-value columns
that the tunnel score needs to the monthly pitch CSVs.

The tunnel score compares each pitch with the previous pitch to the same
batter and controls for batter, count, catcher and stuff, so it needs columns
the original pull left out. Same Savant pull and join as add_velocities.py.

Usage:
    python add_context.py --year 2025
"""

import argparse

from add_velocities import add_columns

# Savant column -> name in the monthly CSVs (units added where they apply)
CONTEXT = {
    "batter": "batter",
    "stand": "stand",
    "p_throws": "p_throws",
    "balls": "balls",
    "strikes": "strikes",
    "outs_when_up": "outs_when_up",
    "inning": "inning",
    "inning_topbot": "inning_topbot",
    "on_1b": "on_1b",
    "on_2b": "on_2b",
    "on_3b": "on_3b",
    "fielder_2": "catcher",
    "game_type": "game_type",
    "zone": "zone",
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
    "woba_value": "woba_value",
    "delta_run_exp": "delta_run_exp",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()
    add_columns(args.year, CONTEXT)

if __name__ == "__main__":
    main()
