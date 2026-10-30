"""
Add bat-tracking and pitcher arm-angle columns to the monthly pitch CSVs:
swing length and bat speed (hitter timing for the perceptual tunnel model),
arm angle, and times the batter has already faced the pitcher this game.

Usage:
    python datamining/add_bat_tracking.py --year 2025
"""

import argparse

from add_velocities import add_columns

COLS = {
    "bat_speed": "bat_speed_mph",
    "swing_length": "swing_length_ft",
    "arm_angle": "arm_angle_deg",
    "n_priorpa_thisgame_player_at_bat": "n_priorpa_thisgame",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()
    add_columns(args.year, COLS)

if __name__ == "__main__":
    main()
