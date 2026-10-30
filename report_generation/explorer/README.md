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

## How the page computes it

1. `pull_velocities.py` and `fit.py` rebuild every 2025 pitch from Statcast's 9-parameter fit, exactly as in
   `polar_transformations/reconstruct_trajectory.py`: the release position $`\mathbf r_0`$, release velocity
   $`\mathbf v_r`$, constant acceleration $`\mathbf a`$ and flight time $`T`$ to the front of the plate, so
   $`\mathbf p(t) = \mathbf r_0 + \mathbf v_r t + \tfrac12\mathbf a t^2`$.
2. `build.py` averages $`\mathbf r_0`$, $`\mathbf v_r`$, $`\mathbf a`$, release speed and plate location per pitcher x
   pitch type (both batter sides pooled; types with >= 25 pitches, pitchers with >= 300), recomputes $`T`$
   from the averaged state, and stores swing, whiff and CSW counts.
3. On the page, the primary fastball is the pitcher's highest-usage FF, SI or FC. With the decision point
   `dec` (ms before the fastball reaches the plate), the decision time is
   $`t_d = \max(0.02,\ T_{fb} - \text{dec})`$ s after release, and every pitch type is compared with the
   fastball at the same time since release (capped at its own $`T`$):
   ```math
   \text{aDeg} = \theta\big(\mathbf p(\min(t_d, T)),\ \mathbf p_{fb}(t_d)\big), \qquad
   \text{gapIn} = 12\,\lVert \mathbf p(\min(t_d, T)) - \mathbf p_{fb}(t_d) \rVert, \qquad
   \text{plateIn} = 12\sqrt{(x_p - x_{p,fb})^2 + (z_p - z_{p,fb})^2}
   ```
   with the eye $`\mathbf E`$ at 65 in high, 28.3 in behind the plate's back tip and 35.8 in off its center
   on the chosen batter's side, and $`\theta`$ the angle between the two directions from the eye.
4. The "What the hitter sees" panel evaluates the optical-motion and fastball-prior-miss features of
   [model/perceptual_report.md](../../model/perceptual_report.md#how-the-model-works) on the average pitch at
   the decision point: angular speed $`\lVert\mathbf v_\perp\rVert / r`$, its difference from the fastball's,
   the rate the angle to the fastball is growing, and where and when a ball read as a fastball from that
   point (the fastball's acceleration applied from the pitch's real position and velocity) would cross the
   plate.

## Variables

| Variable | Meaning | Units | Source |
|---|---|---|---|
| `pitcher` | MLBAM id of the pitcher | id | Statcast column `pitcher` (`datamining/outputs`) |
| `pitch_type` | Statcast pitch type code: FF four-seam, SI sinker, FC cutter, SL slider, ST sweeper, CU curve, KC knuckle curve, SV slurve, CH changeup, FS splitter, FO forkball, KN knuckleball. PO (pitchout), EP (eephus), FA (position players), CS (slow curve), SC (screwball) and UN (unknown) are dropped | category | Statcast column `pitch_type` (`datamining/outputs`) |
| `stand` | Batter side, R or L. Sets the eye position and the sign $`s_b`$ | category | Statcast column `stand` (`datamining/outputs`) |
| `usage` | Explorer: the type's share of the pitcher's 2025 pitches | share | `explorer/build.py` |
| `dec` | Explorer: the decision point, ms before the primary fastball reaches the plate (slider, 100-300 ms, default 175) | ms | `tpl.html` `state.dec` |
| `E` | Batter's eye: $`(s_b \cdot (-35.8),\ -28.3,\ 65.0)`$ in, i.e. 35.8 in off the plate's center line on the batter's side, 28.3 in behind the back of the plate, 65 in high | ft | `tpl.html` `EYE_*`, `to_polar.ORIGIN_*` |
| `theta(a,b)` | Angle between two ball positions as seen from the eye: $`\arccos\big((\mathbf a-\mathbf E)\cdot(\mathbf b-\mathbf E)/(\lVert\mathbf a-\mathbf E\rVert\,\lVert\mathbf b-\mathbf E\rVert)\big)`$ | deg | `features.angle_deg` |
| `p(t)` | Ball position $`t`$ seconds after release, $`\mathbf r_0 + \mathbf v_r t + \tfrac12 \mathbf a t^2`$ | ft | `features.position` |
| `tf` | Flight time $`T`$ from release to the front of the plate. Pitches with $`T \ge 0.6`$ s are dropped | s | `features.add_state` |
| `aDeg` | Explorer: angle from the eye between a pitch type's average path and the primary fastball's, at the decision point | deg | `tpl.html` `metrics()` |
| `gapIn` | Explorer: straight-line distance between the two average paths at the decision point | in | `tpl.html` `metrics()` |
| `plateIn` | Explorer: distance between the two average plate crossings | in | `tpl.html` `metrics()` |
| `ang_speed` | P3. Angular speed of the ball as seen from the eye at commit, $`\lVert\mathbf v_\perp\rVert/r`$ | deg/s | `perceptual.angular_speed` |
| `ang_speed_vs_fb` | P3. `ang_speed` minus the angular speed of the primary fastball's mean path at the same time since release | deg/s | `perceptual.commit_features` |
| `sep_rate_vs_fb` | P3. Rate of change of the angle between the pitch and the fastball's mean path at commit, central difference with h = 5 ms | deg/s | `perceptual.commit_features` |
| `miss_x_in` | P2. Horizontal error of a fastball read: actual plate crossing minus where the ball would cross if, from the commit point, it kept the primary fastball's acceleration. Batter-oriented ($`s_b`$), + is away | in | `perceptual.commit_features` |
| `miss_z_in` | P2. Vertical error of the same fastball read; + means the pitch finished higher than the read | in | `perceptual.commit_features` |
| `miss_t_ms` | P2. Arrival-time error: actual time left at commit ($`\tau_b`$) minus the fastball read's time to the plate. + means the pitch arrives later than a fastball would | ms | `perceptual.commit_features` |
| `swing` | 1 if the batter swung: swinging_strike(_blocked), foul_tip, foul, foul_bunt, hit_into_play, missed_bunt, bunt_foul_tip (`features.SWING`) | 0/1 | `features.add_outcomes_and_controls` |
| `whiff` | 1 if the batter swung and missed: swinging_strike(_blocked), foul_tip, missed_bunt (`features.WHIFF`). Modelled on swings only, so the model is P(whiff \| swing) | 0/1 | `features.add_outcomes_and_controls` |
| `csw` | Called strike or whiff: 1 if `whiff` or `description == called_strike` | 0/1 | `features.add_outcomes_and_controls` |
