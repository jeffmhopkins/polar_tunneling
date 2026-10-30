<div align="center">

# Polar Tunneling

Pitch tunneling measured from the hitter's eye: does a pitch that looks like another one early, then splits late, do better than its stuff and location say it should?

[![Python 3](https://img.shields.io/badge/Python-3-blue)](model/requirements.txt) [![Data: Statcast](https://img.shields.io/badge/data-Statcast%202025%20%7C%202026-blue)](https://baseballsavant.mlb.com/statcast_search) [![Model: LightGBM](https://img.shields.io/badge/model-LightGBM-blue)](https://lightgbm.readthedocs.io) [![Tests: pre-registered](https://img.shields.io/badge/tests-pre--registered-blue)](model/README.md) [![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

</div>

Every pitch is rebuilt from Statcast's 9-parameter trajectory fit and turned into angles as seen from the
batter's eye, in real time rather than normalized flight time. The models train on 2025 and are tested once
on 2026, and every test is written down and committed before its outcome is computed, failures included.

> [!NOTE]
> **Credit.** The idea, the polar transformation and the first data pipeline are by
> [ginsy-codes](https://github.com/ginsy-codes), from
> [ginsy-codes/polar_tunneling](https://github.com/ginsy-codes/polar_tunneling). This repository carries
> that history and builds the models, tests and report pages on top of it. The original outline is in
> [docs/original_outline.md](docs/original_outline.md).

## Try it

<table>
<tr>
<td align="center" valign="top"><a href="https://jeffmhopkins.github.io/polar_tunneling/explorer/"><img src="docs/img/explorer.png" width="400" alt="Pitch Tunnel Explorer"></a><br><b><a href="https://jeffmhopkins.github.io/polar_tunneling/explorer/">Pitch Tunnel Explorer</a></b><br>Every 2025 arsenal in 3D, from the hitter's eye</td>
<td align="center" valign="top"><a href="https://jeffmhopkins.github.io/polar_tunneling/sequencer/"><img src="docs/img/sequencer.png" width="400" alt="Pitch Sequencer"></a><br><b><a href="https://jeffmhopkins.github.io/polar_tunneling/sequencer/">Pitch Sequencer</a></b><br>Best next pitch by count, previous pitch and hitter</td>
</tr>
<tr>
<td align="center" valign="top"><a href="https://jeffmhopkins.github.io/polar_tunneling/game-report/"><img src="docs/img/game-report.png" width="400" alt="Game report"></a><br><b><a href="https://jeffmhopkins.github.io/polar_tunneling/game-report/">Game report</a></b><br>Orioles at Yankees, 2026-09-25, pitch by pitch against five pickers</td>
<td align="center" valign="top"><a href="https://jeffmhopkins.github.io/polar_tunneling/pregame/rogers.html"><img src="docs/img/pregame.png" width="400" alt="Pre-game plan"></a><br><b><a href="https://jeffmhopkins.github.io/polar_tunneling/pregame/rogers.html">Pre-game plan</a></b><br>Trevor Rogers vs the Yankees (also <a href="https://jeffmhopkins.github.io/polar_tunneling/pregame/young.html">Brandon Young</a>)</td>
</tr>
</table>

Every page runs in the browser with no server. [The index](https://jeffmhopkins.github.io/polar_tunneling/) lists them all, and
[Report pages](#report-pages) says how each one is built.

## For humans

> [!TIP]
> **Start with [What it found](#what-it-found)**, the one-table summary. Every row links to the report
> behind it, and [model/README.md](model/README.md) walks through each study in the order it was run.

- To see the results rather than read them, open the [pages above](#try-it): the Pitch Sequencer is the
  one to start with.
- To rerun a study, see [Running it](#running-it). The data is already in the repository.
- The project started from [the original outline](docs/original_outline.md), which is still the best
  explanation of why the hitter's-eye view matters.

## For agents

> [!IMPORTANT]
> **Read [CLAUDE.md](CLAUDE.md) first.** It is the jump start for a new session: what the project is,
> the layout, how to run the scripts and build the pages, and where the work was left.

The rules that matter most:

- Train on 2025 and test on 2026 **once**. A new variant gets a new test, never a rerun of an old one.
- Commit a `model/PREREG_<name>.md` before computing any outcome, and never edit it after the run.
- Report every result, failures included, in `model/<name>_report.md`, `model/README.md` and the table
  below.
- Run scripts from the repository root.

## What it found

| Question | Result | Write-up |
|---|---|---|
| Does a geometric tunnel score (pitch vs the previous pitch) add to stuff, location and count? | No. Pitch types and locations already pin down most of the separation | [feasibility](model/feasibility_report.md), [robustness](model/robustness_report.md) |
| Does an arsenal tunnel score (each pitch vs the fastball) predict whiffs? | Weak in 2025, did not replicate in 2026 | [2025](model/arsenal_report.md), [2026](model/arsenal_report_2026.md), [holdout](model/arsenal_holdout_report.md) |
| Do perceptual features (what the hitter sees and expects at the decision point) help? | **Yes**, pitch to pitch on 2026. No stable per-pitcher Tunnel+ | [perceptual](model/perceptual_report.md) |
| Is that just pitch speed? | No, the gain survives speed and flight-time controls | [speed check](model/speed_check_report.md) |
| Do the sequencer's top picks beat other pitches in real 2026 at-bats? | **Yes**, +4.5 CSW pts, about +3.6 after controlling for pitcher choice | [validation](model/validation_report.md), [choice](model/choice_report.md) |
| Does following it over a whole at-bat help? | **Yes**, through strikeouts (+1.2 K pts per +25 pts adherence), not weaker contact | [at-bat](model/atbat_report.md) |
| Does picking by runs instead of CSW help? | **Yes**, -0.27 runs per 100 pitches where the picks differ | [damage](model/damage_report.md) |
| Does a hitter-specific sequencer help? Does per-hitter timing? | Hitter-specific picks yes (+2.9 CSW pts where they differ). Fitted timing no | [hitter sequencer](model/hitter_sequencer_report.md), [hitter timing](model/hitter_timing_report.md) |
| Do the perceptual features change the *best* pitch, by CSW, runs or location? | No. They improve forecasts but rarely change the pick | [game](model/game_report.md), [runs](model/runs_report.md), [location](model/location_report.md) |

> [!IMPORTANT]
> **The short version.** A per-pitcher Tunnel+ doesn't hold up. A perceptual signal does, pitch to pitch,
> and a sequencer built on the whole model beats simple rules and FanGraphs' Stuff+, Pitching+ and
> PitchingBot grades on 2026. Most of that comes from modelling stuff, location and count well; the
> tunneling features sharpen the forecasts more than the picks. The gains are observational upper bounds,
> since pitchers and catchers scout too.

[model/README.md](model/README.md) has every study in order, with its numbers and caveats.

## Layout

| Folder | What's in it |
|---|---|
| [datamining](datamining) | Statcast pulls (pybaseball and Baseball Savant), the columns added to them later, FanGraphs grades, and the monthly pitch CSVs in `outputs/` |
| [polar_transformations](polar_transformations) | Trajectory rebuild from the 9-parameter fit, and the batter's-eye polar transform |
| [model](model) | Features, models, pre-registrations (`PREREG_*.md`), the scripts that run them and the reports they write |
| [report_generation](report_generation) | HTML templates and builders for the Pitch Sequencer, game report and pre-game plan, plus the [Pitch Tunnel Explorer](report_generation/explorer) |

## Running it

```sh
pip install -r datamining/requirements.txt -r model/requirements.txt -r report_generation/requirements.txt
```

> [!NOTE]
> The pitch data for both seasons is already in `datamining/outputs/` (about 470 MB), so the models run
> without pulling anything. Run scripts from the repository root: paths are relative to it.

Each model script writes its report next to it, for example:

```sh
python model/perceptual.py                 # writes model/perceptual_report.md
python model/damage_sequencer.py --production
python report_generation/build_sequencer.py sequencer.html
```

To pull the data again, run `pull_2025_chunked.py` and `pull_2026_pitches.py`, then `add_velocities.py`,
`add_context.py` and `add_bat_tracking.py` with `--year 2025` and `--year 2026`. Each script's docstring
says what it adds.

## Report pages

| Page | Built by | Data from |
|---|---|---|
| [Pitch Sequencer](https://jeffmhopkins.github.io/polar_tunneling/sequencer/) | `report_generation/build_sequencer.py OUT.html` | `model/sequencing.json`, `model/batter_adj/` |
| [Game report](https://jeffmhopkins.github.io/polar_tunneling/game-report/) | `report_generation/build_game_report.py 823491 OUT.html` | `model/game_report.py`, `runs_compare.py`, `location_compare.py` |
| [Pre-game plan](https://jeffmhopkins.github.io/polar_tunneling/pregame/rogers.html) | `report_generation/build_pregame.py 823491 669432 OUT.html` | `model/pregame_report.py` |
| [Pitch Tunnel Explorer](https://jeffmhopkins.github.io/polar_tunneling/explorer/) | [report_generation/explorer](report_generation/explorer) | 2025 Statcast trajectory fits |

Each page is one self-contained HTML file, except the sequencer, which loads hitter data from an `adj/`
folder copied next to it. The published copies live on the `gh-pages` branch.

## Data

Pitch data is from [Baseball Savant](https://baseballsavant.mlb.com) (MLB Advanced Media), pulled with
[pybaseball](https://github.com/jldbc/pybaseball). Stuff+, Pitching+ and PitchingBot grades are from
[FanGraphs](https://www.fangraphs.com). Both are subject to their owners' terms.

## License

[MIT](LICENSE). The pitch data in `datamining/outputs/` is not covered by it: it stays under Baseball
Savant's and FanGraphs' terms.
