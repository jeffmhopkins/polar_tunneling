"""
Public pitch-type grades for 2025 from FanGraphs: Stuff+ (sp_s_*), Pitching+
(sp_p_*) and PitchingBot overall (pb_o_*), keyed by MLBAM id.

Usage:
    python datamining/pull_fangraphs_grades.py   # writes outputs/fangraphs_grades_2025.csv
"""

from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).resolve().parent / "outputs" / "fangraphs_grades_2025.csv"
URL = "https://www.fangraphs.com/api/leaders/major-league/data"

def main():
    params = dict(age="", pos="all", stats="pit", lg="all", qual=0, season=2025, season1=2025, ind=0, type=36,
                  pageitems=3000, pagenum=1)
    d = pd.DataFrame(requests.get(URL, params=params, timeout=120).json()["data"])
    cols = ["xMLBAMID", "PlayerName", "Pitches"] + [c for c in d.columns if c[:5] in ("sp_s_", "sp_p_", "pb_o_")]
    d[cols].rename(columns={"xMLBAMID": "pitcher"}).to_csv(OUT, index=False)
    print("wrote", OUT, len(d), "pitchers")

if __name__ == "__main__":
    main()
