# Perceptual tunnel model: pre-registration

Written and committed before any outcome is computed for these features. The
2026 test is run once. Every result is reported, including failures.

## Why

The geometric tunnel features (sequence pairs, arsenal vs fastball) add no
held-out value (`model/README.md`). Four ways the earlier model could have
missed real tunneling: a fixed 175 ms decision point, geometry instead of what
a hitter perceives or expects, unused pitch-by-pitch release consistency, and
no build-up over the at-bat. This model targets those four.

## Data

- Train: 2025 regular season (`game_type == "R"`), all months.
- Test: 2026 regular season, scored once.
- Reference clouds and usage rates are computed per season from trajectories
  and pitch labels only (no outcomes), so they are not leakage.
- New Savant columns: `bat_speed`, `swing_length`, `arm_angle`,
  `n_priorpa_thisgame_player_at_bat`.

## Controls (M0)

These are the same as feasibility.py: stuff, location, count, platoon, and the
previous pitch's type, location and result. Added to them:

- fastball differentials: speed, movement and release vs the pitcher's primary
  fastball to that side
- arm angle
- batter whiff/swing and swing rates, computed within season leaving out the
  current game

## Perceptual features (P), all at the hitter's commit point

**P1 Hitter timing.**
- tau_b = 175 ms x (batter's swing_length / bat_speed) / (league mean of the
  same).
- The batter's ratio is shrunk to the league mean with a prior of 50 swings,
  then clipped to 125-225 ms.
- tau_b replaces the fixed 175 ms everywhere below. It is also a feature
  itself.

**P2 Fastball-prior extrapolation error.**
- At commit, the ball's position and velocity are extrapolated to the front of
  the plate using the acceleration of the pitcher's primary fastball.
- Features: horizontal miss and vertical miss (in, batter-oriented), and
  arrival-time miss (ms).

**P3 Optical motion.**
- Features: angular speed of the ball from the eye at commit (deg/s), and its
  difference from the pitcher's primary-fastball mean path at the same moment.
- Also the rate at which the angular separation from that path is growing
  (deg/s).

**P4 Commit-time ambiguity (Bayesian observer).**
- Per pitcher x side x pitch type: a 2D Gaussian of eye-frame angles at
  175 ms before arrival. Covariance has 0.1 deg added per axis as a perceptual
  floor. The season usage rate is the prior.
- Features: posterior that the pitch is the primary fastball, and posterior of
  its true type.
- The same two features are computed from release-point (x, z) clouds.

**P5 Expectation.**
- -log2 P(type | pitcher, side, count) from season usage.
- -log2 P(type | pitcher, side, previous pitch type).

**P6 At-bat build-up.**
- Fastballs already seen in the plate appearance.
- Angular separation at commit to the most recent fastball in the PA (missing
  if none).
- Times through the order (n_priorpa_thisgame_player_at_bat).
- Pitch number.

## Models and endpoints

- LightGBM with the same parameters as feasibility.py. M0 = controls. M1 =
  controls + all of P. The P1-P6 ablations are reported but do not decide
  pass/fail.
- Primary endpoints: the 2026 held-out log-loss gain of M1 over M0 for
  whiff | swing, and for swing.
- Null: M1 retrained with the P features permuted within pitcher x pitch type
  in train and test, 5 draws.
- **Pass** if the gain is >= 1.0 per 1000 on whiff | swing or on swing, AND
  above all 5 null draws on that endpoint. (For scale: adding previous-pitch
  context gave 1.4 and 2.1 per 1000 in 2025.)
- Secondary: a pitcher x pitch type's mean P value in 2026. That is M1's
  prediction minus M1 with P set to the pitch type's median, on whiff | swing.
  It should predict M0 residual whiff/swing in the other half of 2026 games
  with r >= 0.10.
