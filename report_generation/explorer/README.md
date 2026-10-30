<div align="center">

# Pitch Tunnel Explorer

Every 2025 pitcher's arsenal in 3D, with each pitch's angle gap from the fastball as the hitter sees it.

[![Data: Statcast 2025](https://img.shields.io/badge/data-Statcast%202025-blue)](pull_velocities.py) [![Page: one HTML file](https://img.shields.io/badge/page-one%20HTML%20file-blue)](tpl.html)

</div>

## Building the page

1. `pull_velocities.py OUT.csv` pulls the 2025 Statcast trajectory fit (vx0..az) by day.
2. `fit.py OUT.csv REPO FIT.csv` runs each pitch's fit back to the release point.
3. `build.py FIT.csv REPO data.json` averages per pitcher x pitch type and writes `data.json`.
4. `inject.py tpl.html data.json explorer.html` fills the template.

> [!NOTE]
> `REPO` is the path to this repository's root. The result, `explorer.html`, is self-contained.

## What the hitter sees

The "What the hitter sees at the decision" panel computes, on the average pitch, the optical-motion
and fastball-prior-miss features defined in [model/perceptual.py](../../model/perceptual.py): angular
speed, speed vs. the fastball, separation rate, and where a fastball read from the decision point would
cross the plate.
