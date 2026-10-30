# Speed check: pre-registration

Written and committed before running. Single 2025 -> 2026 run, same data,
LightGBM parameters and perceptual features as `PREREG.md`.

## Question

The perceptual features passed the primary test, mostly through optical motion
(P3). Is that gain really perception, or speed and timing information that the
controls missed?

## Speed controls (S), added to M0 to make M0s

- plate speed (mph): the fit's speed at the front of the plate
- flight time (ms) from release to the front of the plate
- speed differential vs the previous pitch in the PA (release speed, mph)
- flight-time differential vs the previous pitch (ms)
- flight-time differential vs the pitcher's primary fastball (ms); the
  release-speed differential to the fastball is already in M0
- release speed is already in M0; no change

## Endpoints

- **Primary:** the 2026 log-loss gain of M1s = M0s + all perceptual features
  over M0s, for whiff | swing and for swing.
- **Pass** (the signal survives) if the gain is >= 1.0 per 1000 on either
  endpoint AND above all 5 null draws on that endpoint. Nulls permute the
  perceptual features within pitcher x pitch type in train and test.
- Also reported:
  - the gain of each block P1-P6 alone over M0s
  - the share of the original M1 gain (1.83 / 5.86) that remains

## Exploratory (labeled as such, not part of pass/fail)

M0k = M0s + the ball's batter-frame position and velocity at the commit point
(6 numbers). Optical motion is a function of these plus the eye position, so
this asks whether P3 adds anything beyond raw kinematics at the decision
moment.
