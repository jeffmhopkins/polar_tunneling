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
- **Document the math.** Every `*_report.md` ends with a hand-written **How the model works** (the math,
  step by step, from the trajectory to the table) and **Variables** (name, meaning, units, source column)
  section, and every `PREREG_*.md` ends with **Variables**. The scripts don't write these sections: when a
  script rewrites its report, carry them over. A new report or pre-registration needs both, with
  definitions matching [model/README.md](model/README.md#variables).
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

The published copies are on the `gh-pages` branch, served at
https://jeffmhopkins.github.io/polar_tunneling/ (`index.html` there lists them). To update one, rebuild it,
copy it over the file on that branch (artifact-style pages need a `<!doctype html>` head added), and
refresh the screenshot in `docs/img/` on `main` if it changed.

## Where things were left

The open ideas, none started: pitch design suggestions (how a change in a pitch's movement or release
would change its deception against the rest of the arsenal), a drift check (train early 2026, test late
2026), and a short write-up of the method and results.

## Variables

| Variable | Meaning | Units | Source |
|---|---|---|---|
| `TAU` | Commit point: how long before the ball reaches the front of the plate the hitter must decide. Fixed at 0.175 s unless stated | s | `features.TAU` |
| `tau_b` | P1. Hitter's commit time from bat tracking: $`0.175 \times \tilde\rho_b/\bar\rho`$, clipped to 0.125-0.225 s, where $`\rho`$ = swing length / bat speed (swings over 40 mph), $`\bar\rho`$ the league mean and $`\tilde\rho_b`$ the batter's mean shrunk to it with 50 prior swings. 0.175 s if he has none | s | `perceptual.batter_tau` |
| `CSW` | Share of pitches that are a called strike or whiff (`csw` averaged) | % or pts | `csw` |
| `pts` | Percentage points (a rate difference x100) | pts | - |
| `runs per 100 pitches` | Run value per 100 pitches; negative means fewer runs allowed | runs | `damage_sequencer.py`, `runs_compare.py` |
| `log-loss` | $`-\frac1n\sum_i [y_i\log\hat p_i + (1-y_i)\log(1-\hat p_i)]`$ on held-out pitches; lower is better | nats per pitch | `sklearn.metrics.log_loss` |
| `gain x1000` | $`1000\,(L_{\text{without}} - L_{\text{with}})`$: how much adding the features lowers held-out log-loss. + helps | nats per 1000 pitches | computed in each script |
