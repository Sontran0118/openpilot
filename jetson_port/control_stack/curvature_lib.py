#!/usr/bin/env python3
"""Proper path -> curvature derivation, matching how openpilot's lateral planner
gets desiredCurvature from the model path — NOT the naive 2y/x^2.

openpilot fits the path y(x) and evaluates curvature from the fit near a
lookahead distance. Curvature of a planar curve y(x):
    k = y'' / (1 + y'^2)^1.5
For a road-following path y' is small, so k ~= y''. We fit a low-order
polynomial y(x) over the near path (robust to model noise) and take the 2nd
derivative at a speed-dependent lookahead.
"""
import numpy as np

# openpilot model time indices (33 pts) — path points are at these times
T_IDXS = np.array([0.0, 0.00976, 0.0398, 0.0898, 0.159, 0.248, 0.357, 0.485,
                   0.633, 0.801, 0.989, 1.197, 1.425, 1.673, 1.941, 2.229,
                   2.537, 2.865, 3.213, 3.581, 3.969, 4.377, 4.805, 5.253,
                   5.721, 6.209, 6.717, 7.245, 7.793, 8.361, 8.949, 9.557, 10.185])


def path_to_curvature(path_xyz, v_ego=20.0):
    """Derive desired curvature from the model path.
    path_xyz: (33,3) forward(x)/left(y)/up(z) meters.
    v_ego: speed for choosing a sensible lookahead distance.
    Returns curvature in 1/m (positive = left)."""
    p = np.asarray(path_xyz, dtype=float)
    if p.shape[0] < 6:
        return 0.0
    x = p[:, 0]
    y = p[:, 1]

    # Need enough forward extent to fit; if the path is basically static
    # (parked / no road), curvature is 0.
    if x[-1] - x[0] < 1.0:
        return 0.0

    # Lookahead distance ~ speed * time, clamped to the path we actually have.
    # openpilot uses a ~T seconds lookahead; use ~2.5s worth of distance.
    lookahead_d = float(np.clip(v_ego * 2.5, 5.0, x[-1] * 0.9))

    # Fit y as a function of x with a quadratic over the near path (up to
    # lookahead). Quadratic -> constant 2nd derivative = curvature estimate,
    # robust to per-point model noise. Weight nearer points more.
    mask = x <= lookahead_d
    if mask.sum() < 4:
        mask = np.ones_like(x, dtype=bool)
    xf, yf = x[mask], y[mask]

    # weights: emphasize near-field (where steering acts now)
    w = 1.0 / (1.0 + xf)
    # y = a*x^2 + b*x + c  ; curvature (small-angle) ~ y'' = 2a
    try:
        coeffs = np.polyfit(xf, yf, 2, w=w)
    except Exception:
        return 0.0
    a, b, c = coeffs
    yp = b          # y'(x=0) heading
    ypp = 2.0 * a   # y''
    # full planar curvature (not just small-angle)
    curv = ypp / (1.0 + yp * yp) ** 1.5
    # sanity clamp — physical road curvature is small
    return float(np.clip(curv, -0.2, 0.2))


if __name__ == "__main__":
    # self-test with known-curvature synthetic paths
    print("=== curvature derivation self-test ===")
    # a circular arc of radius R has curvature 1/R. Build a path on that arc.
    for R in (200.0, 100.0, 50.0, 1e9):
        k_true = 1.0 / R
        # points along arc: x forward, y = R - sqrt(R^2 - x^2) ~ x^2/(2R) for small x
        xs = np.linspace(0, 50, 33)
        ys = R - np.sqrt(np.maximum(R * R - xs * xs, 0.0))
        path = np.stack([xs, ys, np.zeros_like(xs)], axis=1)
        k_est = path_to_curvature(path, v_ego=20.0)
        print("  R=%8.0f  true k=%.5f  est k=%.5f  err=%.1e" %
              (R, k_true, k_est, abs(k_est - k_true)))
