# Session jump start

What a new session needs to pick up work on this repository without re-reading all of it.

## The project in one paragraph

Pitch tunneling measured from the batter's eye. Pitches are rebuilt from Statcast's 9-parameter fit
(`polar_transformations/reconstruct_trajectory.py`), viewed from a fixed eye position per batter side
(`polar_transformations/to_polar.py`), and compared at a commit point 175 ms before the pitch reaches the
plate (`TAU` in `model/features.py`). Geometric tunnel scores failed; perceptual features (what the hitter
sees and expects at the decision point) help pitch-level forecasts on 2026 but rarely change the best pick.
The useful product is the pitch sequencer, which ranks a pitcher's options by runs saved and CSW for a
count, batter side, previous pitch and hitter. [README.md](README.md) has the findings table and
[model/README.md](model/README.md) has every study with its numbers.

## Rules the work follows

- **Train on 2025, test on 2026 once.** Anything fitted (reference clouds, usage, priors, hitter rates) comes
  from the training data only. Don't rerun a 2026 test to try a variant: add a new pre-registration.
- **Pre-register before running.** A new test gets a `model/PREREG_<name>.md` (question, data, controls,
  primary and secondary tests, pass bar) committed before any outcome is computed. An amendment goes in the
  same file, marked "Amendment (before any run)". Never edit a pre-registration after its run.
- **Report everything.** The script writes `model/<name>_report.md`, including failures. Then add a section
  to `model/README.md` and a row to its study table, and to the findings table in `README.md` if it changes
  the story. Anything outside the pre-registration is labelled exploratory.
- **Say what the numbers can't.** Gains are observational: pitchers, catchers and scouting choose pitches
  too, so the sequencer's gains are upper bounds. Keep that caveat with any new result.
- **Writing style.** Plain sentences, numbers with their CI, "pts" for percentage points, CSW per pitch and
  runs per 100 pitches. No hype.

## Layout and data

- `datamining/outputs/pitches_<year>_<MM>.csv`: one file per month, 2025 and 2026, about 470 MB in all,
  with the columns the pull, `add_velocities.py`, `add_context.py` and `add_bat_tracking.py` wrote. Load
  them with `load(year)` in `model/features.py` instead of re-pulling.
- `datamining/outputs/fangraphs_grades_2025.csv`: FanGraphs Stuff+, Pitching+ and PitchingBot by pitch type.
- `model/`: one script per study, each with a usage docstring. `--production` on `hitter_sequencer.py` or
  `damage_sequencer.py` retrains on both seasons and writes the page data (`sequencing.json`,
  `batter_adj/`). The current page data comes from `damage_sequencer.py --production`.
- `report_generation/`: `*_tpl.html` templates and `build_*.py` scripts that inject JSON into them. The
  game and pre-game JSON live here, named by game pk (and pitcher id).

## Running things

- Run every script from the repository root. Paths like `datamining/outputs` are relative to it.
- Install with `pip install -r datamining/requirements.txt -r model/requirements.txt -r report_generation/requirements.txt`.
- Several model scripts cache fitted models or test tables in `/tmp` (`/tmp/game_report_cache.pkl`,
  `/tmp/location_models.pkl` and others). Delete the cache if inputs change.
- Full model runs load both seasons and take minutes, not seconds. Nothing here has a test suite. Check a
  change by rerunning the script it touches and comparing its report.

## Pages

| Page | Build |
|---|---|
| Pitch Sequencer | `python report_generation/build_sequencer.py OUT.html` (copies `adj/` next to it) |
| Game report | `python report_generation/build_game_report.py <game_pk> OUT.html` |
| Pre-game plan | `python model/pregame_report.py --game <game_pk> --pitcher <pitcher_id>`, then `python report_generation/build_pregame.py <game_pk> <pitcher_id> OUT.html` |
| Pitch Tunnel Explorer | the four steps in [report_generation/explorer/README.md](report_generation/explorer/README.md) |

## Where things were left

The open ideas, none started: pitch design suggestions (how a change in a pitch's movement or release
would change its deception against the rest of the arsenal), a drift check (train early 2026, test late
2026), and a short write-up of the method and results.
