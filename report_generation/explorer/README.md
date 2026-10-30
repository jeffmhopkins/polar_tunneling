# Pitch Tunnel Explorer

Source for the Pitch Tunnel Explorer page.

1. `pull_velocities.py OUT.csv` pulls the 2025 Statcast trajectory fit (vx0..az) by day.
2. `fit.py OUT.csv REPO FIT.csv` runs each pitch's fit back to the release point.
3. `build.py FIT.csv REPO data.json` averages per pitcher x pitch type and writes `data.json`.
4. `inject.py tpl.html data.json explorer.html` fills the template.

The "What the hitter sees at the decision" panel computes, on the average pitch, the optical-motion
and fastball-prior-miss features defined in `model/perceptual.py` (angular speed, speed vs. the
fastball, separation rate, and where a fastball read from the decision point would cross the plate).
