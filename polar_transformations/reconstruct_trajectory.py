"""
Reconstruct a pitch trajectory under the constant-acceleration model.

Assumptions:
- Constant acceleration (ax, ay, az) for the full flight.
- t = 0 at release

When Statcast's measured initial velocities (vx0, vy0, vz0, given at y = 50 ft)
are available, the trajectory is Statcast's own 9-parameter fit: run back to
release_pos_y and forward to the front edge of the plate (y = 17/12 ft), where
plate_x / plate_z are reported. The fit reproduces Statcast's reported release
point to within about 0.05 in.

Without them, the older approximation is used: flight time is guessed from
release speed, velocity is back-solved to reach plate_x / plate_z at y = 0.
That guess is off by about 1.5 ms in flight time and about 1.7 in of ball
position 175 ms before the plate (median, 2025).

Can be used as a library or run as a CLI on a single pitch row.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def solve_initial_velocity(x0, y0, z0, ax, ay, az, xf, zf, t_flight):
    """
    Given constant acceleration and known start/end positions,
    recover initial velocity components for a given flight time.

    x(t) = x0 + vx0*t + 0.5*ax*t**2
    => vx0 = (xf - x0 - 0.5*ax*t**2) / t
    """
    vx0 = (xf - x0 - 0.5 * ax * t_flight**2) / t_flight
    vy0 = (0.0 - y0 - 0.5 * ay * t_flight**2) / t_flight
    vz0 = (zf - z0 - 0.5 * az * t_flight**2) / t_flight
    return vx0, vy0, vz0

def estimate_flight_time(release_speed_mph, y0):
    """
    Rough flight-time estimate from release speed and y-distance.
    release_speed is mph; convert to ft/s.
    """
    speed_ft_s = release_speed_mph * 1.46667  # mph to ft/s
    # Approximate average speed ~ 0.95 * release speed
    avg_speed = 0.95 * speed_ft_s
    return abs(y0) / avg_speed

FIT_Y = 50.0  # Statcast reports vx0/vy0/vz0 at y = 50 ft
PLATE_FRONT_Y = 17 / 12  # plate_x / plate_z are measured at the front edge of the plate

def _time_at_y(y, vy0, ay):
    """Time (from y = 50 ft) at which the fit reaches y; negative before y = 50."""
    a, b, c = 0.5 * ay, vy0, FIT_Y - y
    return (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)

def release_state_from_fit(release_pos_y, vx0, vy0, vz0, ax, ay, az, plate_x, plate_z):
    """
    Position and velocity at release, and flight time to the front of the plate,
    from Statcast's 9-parameter fit.

    The fit's x/z at y = 50 ft are not in the public CSV, so they are recovered
    from plate_x / plate_z, which are the fit evaluated at the front of the plate.
    """
    t_plate = _time_at_y(PLATE_FRONT_Y, vy0, ay)
    t_rel = _time_at_y(release_pos_y, vy0, ay)
    x50 = plate_x - vx0 * t_plate - 0.5 * ax * t_plate**2
    z50 = plate_z - vz0 * t_plate - 0.5 * az * t_plate**2
    pos = (
        x50 + vx0 * t_rel + 0.5 * ax * t_rel**2,
        release_pos_y,
        z50 + vz0 * t_rel + 0.5 * az * t_rel**2,
    )
    vel = (vx0 + ax * t_rel, vy0 + ay * t_rel, vz0 + az * t_rel)
    return pos, vel, t_plate - t_rel

def reconstruct_trajectory(
    release_pos_x, release_pos_y, release_pos_z,
    ax, ay, az,
    plate_x, plate_z,
    release_speed_mph,
    dt=0.01,
    vx0=None, vy0=None, vz0=None,
):
    """
    Return a DataFrame with columns: t, x, y, z
    t=0 at release, sampled every dt seconds until plate.

    Pass vx0/vy0/vz0 (Statcast, ft/s at y = 50 ft) to use the measured fit;
    otherwise flight time is estimated from release speed.
    """
    if vx0 is not None and vy0 is not None and vz0 is not None:
        (x0, y0, z0), (vxr, vyr, vzr), t_flight = release_state_from_fit(
            release_pos_y, vx0, vy0, vz0, ax, ay, az, plate_x, plate_z,
        )
        times = np.arange(0, t_flight, dt)
        times = np.append(times, t_flight)
        xs = x0 + vxr * times + 0.5 * ax * times**2
        ys = y0 + vyr * times + 0.5 * ay * times**2
        zs = z0 + vzr * times + 0.5 * az * times**2
        return pd.DataFrame({"t": times, "x": xs, "y": ys, "z": zs})

    t_flight = estimate_flight_time(release_speed_mph, release_pos_y)
    vx0, vy0, vz0 = solve_initial_velocity(
        release_pos_x, release_pos_y, release_pos_z,
        ax, ay, az,
        plate_x, plate_z,
        t_flight,
    )

    times = np.arange(0, t_flight + dt/2, dt)
    xs = release_pos_x + vx0 * times + 0.5 * ax * times**2
    ys = release_pos_y + vy0 * times + 0.5 * ay * times**2
    zs = release_pos_z + vz0 * times + 0.5 * az * times**2

    # Ensure last point is exactly at plate if close
    if ys[-1] > 0:
        times = np.append(times, t_flight)
        xs = np.append(xs, plate_x)
        ys = np.append(ys, 0.0)
        zs = np.append(zs, plate_z)

    return pd.DataFrame({"t": times, "x": xs, "y": ys, "z": zs})

def main():
    parser = argparse.ArgumentParser(description="Reconstruct a single pitch trajectory")
    parser.add_argument("--release_pos_x", type=float, required=True)
    parser.add_argument("--release_pos_y", type=float, required=True)
    parser.add_argument("--release_pos_z", type=float, required=True)
    parser.add_argument("--ax", type=float, required=True)
    parser.add_argument("--ay", type=float, required=True)
    parser.add_argument("--az", type=float, required=True)
    parser.add_argument("--plate_x", type=float, required=True)
    parser.add_argument("--plate_z", type=float, required=True)
    parser.add_argument("--release_speed", type=float, required=True)
    parser.add_argument("--vx0", type=float, default=None)
    parser.add_argument("--vy0", type=float, default=None)
    parser.add_argument("--vz0", type=float, default=None)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    traj = reconstruct_trajectory(
        args.release_pos_x, args.release_pos_y, args.release_pos_z,
        args.ax, args.ay, args.az,
        args.plate_x, args.plate_z,
        args.release_speed,
        dt=args.dt,
        vx0=args.vx0, vy0=args.vy0, vz0=args.vz0,
    )

    if args.output:
        traj.to_csv(args.output, index=False)
        print(f"Wrote {len(traj)} points to {args.output}")
    else:
        print(traj.to_string(index=False))

if __name__ == "__main__":
    main()
