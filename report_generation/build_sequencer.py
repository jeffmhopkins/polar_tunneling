"""
Build the Pitch Sequencer page: inject model/sequencing.json into
sequencer_tpl.html, and copy the hitter adjustment shards (model/batter_adj)
to adj/ next to it, where the page loads them on demand.

Usage:
    python report_generation/build_sequencer.py OUT.html
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    tpl = (ROOT / "report_generation" / "sequencer_tpl.html").read_text()
    data = (ROOT / "model" / "sequencing.json").read_text()
    out = Path(sys.argv[1])
    out.write_text(tpl.replace("/*DATA*/null", data))
    adj = out.parent / "adj"
    adj.mkdir(exist_ok=True)
    shards = sorted((ROOT / "model" / "batter_adj").glob("adj_*.json"))
    for f in shards:
        shutil.copy(f, adj / f.name)
    print(f"wrote {out} ({out.stat().st_size // 1024} KB) and {len(shards)} hitter shards")

if __name__ == "__main__":
    main()
