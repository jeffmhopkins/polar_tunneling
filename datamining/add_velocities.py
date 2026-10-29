"""
Add Statcast's measured initial velocities (vx0, vy0, vz0) to the monthly
pitch CSVs that the chunked pull already wrote.

vx0/vy0/vz0 are the velocity terms of Statcast's 9-parameter constant-
acceleration fit, evaluated at y = 50 ft. With them (plus ax/ay/az and the
plate location) a trajectory can be rebuilt exactly instead of guessing the
flight time from release speed.

Pulls one day at a time from Baseball Savant's CSV endpoint and joins on
(game_pk, at_bat_number, pitch_number).

Usage:
    python add_velocities.py --year 2025
"""

import argparse
import csv
import datetime as dt
import io
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

URL = (
    "https://baseballsavant.mlb.com/statcast_search/csv?all=true&type=details"
    "&game_date_gt={d}&game_date_lt={d}&player_type=pitcher"
)
KEYS = ["game_pk", "at_bat_number", "pitch_number"]
VEL = {"vx0": "vx0_ft_s", "vy0": "vy0_ft_s", "vz0": "vz0_ft_s"}
OUTPUT_DIR = Path("datamining/outputs")

def pull_day(day):
    for attempt in range(4):
        try:
            r = requests.get(URL.format(d=day), timeout=180)
            r.raise_for_status()
            text = r.content.decode("utf-8-sig")
            if len(text) < 50:
                return None
            return pd.read_csv(io.StringIO(text), usecols=KEYS + list(VEL), dtype=str)
        except Exception as e:
            print(f"{day}: retry after {e}")
            time.sleep(2**attempt)
    raise RuntimeError(f"Could not pull {day}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()

    files = sorted(OUTPUT_DIR.glob(f"pitches_{args.year}_*.csv"))
    dates = pd.concat([pd.read_csv(f, usecols=["game_date"]) for f in files]).game_date
    start, end = dt.date.fromisoformat(dates.min()), dt.date.fromisoformat(dates.max())
    days = [str(start + dt.timedelta(i)) for i in range((end - start).days + 1)]

    with ThreadPoolExecutor(4) as ex:
        frames = [f for f in ex.map(pull_day, days) if f is not None]
    vel = pd.concat(frames).drop_duplicates(KEYS)
    lookup = {tuple(int(float(k)) for k in row[:3]): row[3:] for row in vel[KEYS + list(VEL)].itertuples(index=False)}
    print(f"Pulled velocities for {len(lookup):,} pitches")

    # Append columns as text so the existing columns stay byte-for-byte unchanged.
    for f in files:
        with open(f, newline="") as fh:
            rows = list(csv.reader(fh))
        header = rows[0]
        if VEL["vx0"] in header:
            print(f"{f} already has velocities")
            continue
        idx = [header.index(k) for k in KEYS]
        missing = 0
        out = [header + list(VEL.values())]
        for row in rows[1:]:
            v = lookup.get(tuple(int(float(row[i])) for i in idx))
            if v is None:
                missing += 1
                v = ("", "", "")
            out.append(row + list(v))
        with open(f, "w", newline="") as fh:
            csv.writer(fh, lineterminator="\n").writerows(out)
        print(f"{f}: {len(rows) - 1:,} pitches, {missing} without velocities")

if __name__ == "__main__":
    main()
