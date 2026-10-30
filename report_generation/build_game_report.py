"""
Build the game report page: inject report_generation/game_<pk>.json (from
model/game_report.py, plus game_<pk>_runs.json and game_<pk>_location.json
when present) into game_report_tpl.html.

Usage:
    python report_generation/build_game_report.py 823491 OUT.html
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def main():
    pk, out = sys.argv[1], Path(sys.argv[2])
    tpl = (ROOT / "report_generation" / "game_report_tpl.html").read_text()
    data = json.loads((ROOT / "report_generation" / f"game_{pk}.json").read_text())
    runs = ROOT / "report_generation" / f"game_{pk}_runs.json"
    if runs.exists():  # from model/runs_compare.py
        data["runs"] = json.loads(runs.read_text())
    loc = ROOT / "report_generation" / f"game_{pk}_location.json"
    if loc.exists():  # from model/location_compare.py
        data["location"] = json.loads(loc.read_text())
    raw = pd.concat([pd.read_csv(f, usecols=["game_pk", "at_bat_number", "pitch_number", "zone"])
                     for f in sorted((ROOT / "datamining" / "outputs").glob("pitches_*.csv"))])
    zone = raw[raw.game_pk == int(pk)].set_index(["at_bat_number", "pitch_number"]).zone
    for p in data["pitches"]:
        z = zone.get((p["ab"], p["n"]))
        p["zone"] = None if z is None or pd.isna(z) else int(z)
    data = json.dumps(data, separators=(",", ":"))
    out.write_text(tpl.replace("__DATA__", data))
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")

if __name__ == "__main__":
    main()
