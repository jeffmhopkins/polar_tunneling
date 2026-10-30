"""
Build the pre-game report page: inject report_generation/pregame_<pk>_<pitcher>.json
(from model/pregame_report.py) into pregame_tpl.html.

Usage:
    python report_generation/build_pregame.py 823491 669432 OUT.html
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    pk, pitcher, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    tpl = (ROOT / "report_generation" / "pregame_tpl.html").read_text()
    data = json.loads((ROOT / "report_generation" / f"pregame_{pk}_{pitcher}.json").read_text())
    g = data["game"]["teams"]
    opp = g["home"] if g["away"]["starter"] == data["pitcher"]["id"] else g["away"]
    title = f"{data['pitcher']['name'].split()[-1]} vs {opp['name'].split()[-1]} Plan"
    page = tpl.replace("<title>Pre-Game Pitch Plan</title>", f"<title>{title}</title>")
    out.write_text(page.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")

if __name__ == "__main__":
    main()
