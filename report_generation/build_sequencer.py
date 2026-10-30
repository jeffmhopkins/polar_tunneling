"""
Build the Pitch Sequencer page: inject model/sequencing.json into
sequencer_tpl.html.

Usage:
    python report_generation/build_sequencer.py OUT.html
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    tpl = (ROOT / "report_generation" / "sequencer_tpl.html").read_text()
    data = (ROOT / "model" / "sequencing.json").read_text()
    out = Path(sys.argv[1])
    out.write_text(tpl.replace("/*DATA*/null", data))
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")

if __name__ == "__main__":
    main()
