"""
Build the game report page: inject report_generation/game_<pk>.json (from
model/game_report.py) into game_report_tpl.html.

Usage:
    python report_generation/build_game_report.py 823491 OUT.html
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    pk, out = sys.argv[1], Path(sys.argv[2])
    tpl = (ROOT / "report_generation" / "game_report_tpl.html").read_text()
    data = (ROOT / "report_generation" / f"game_{pk}.json").read_text()
    out.write_text(tpl.replace("__DATA__", data))
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")

if __name__ == "__main__":
    main()
