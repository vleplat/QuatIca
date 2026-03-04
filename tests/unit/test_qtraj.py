import numpy as np
import quaternion

from quatica.qtraj import (
    enforce_sign_continuity,
    geodesic_distance_s3,
    interpolate_piecewise_slerp,
    interpolate_squad,
    log_euclidean_spline,
    slerp,
)


def _rand_unit_quat(rng: np.random.Generator) -> quaternion.quaternion:
    x = rng.normal(size=4)
    x = x / np.linalg.norm(x)
    return quaternion.quaternion(float(x[0]), float(x[1]), float(x[2]), float(x[3]))


def test_enforce_sign_continuity_flips_when_needed():
    rng = np.random.default_rng(0)
    q0 = _rand_unit_quat(rng)
    q1 = -q0
    qs = enforce_sign_continuity([q0, q1])
    assert geodesic_distance_s3(qs[0], qs[1]) < 1e-12


def test_slerp_endpoints_and_unit_norm():
    rng = np.random.default_rng(1)
    q0 = _rand_unit_quat(rng)
    q1 = _rand_unit_quat(rng)

    q_at_0 = slerp(q0, q1, 0.0)
    q_at_1 = slerp(q0, q1, 1.0)

    assert geodesic_distance_s3(q_at_0, q0) < 1e-12
    assert geodesic_distance_s3(q_at_1, q1) < 1e-12

    for t in [0.0, 0.2, 0.5, 0.9, 1.0]:
        qt = slerp(q0, q1, float(t))
        n = np.linalg.norm(quaternion.as_float_array(qt))
        assert abs(n - 1.0) < 1e-12


def test_interpolants_hit_keyframes_geodesically():
    rng = np.random.default_rng(2)
    K = 6
    ts = np.linspace(0.0, 1.0, K)

    # Create a "smooth-ish" sequence by increasing rotation angle.
    axes = rng.normal(size=(K, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    angles = np.linspace(0.0, 1.2, K)
    qs = [quaternion.from_rotation_vector(axes[i] * angles[i]) for i in range(K)]
    qs = enforce_sign_continuity(qs)

    # Piecewise SLERP and SQUAD are sampled at knots by construction.
    t_slerp, q_slerp = interpolate_piecewise_slerp(qs, ts, samples_per_seg=50)
    t_squad, q_squad = interpolate_squad(qs, ts, samples_per_seg=50)

    # For each key time, check nearest sample is very close.
    for tk, qk in zip(ts, qs):
        i1 = int(np.argmin(np.abs(t_slerp - tk)))
        i2 = int(np.argmin(np.abs(t_squad - tk)))
        assert geodesic_distance_s3(q_slerp[i1], qk) < 1e-10
        assert geodesic_distance_s3(q_squad[i2], qk) < 1e-10

    # Log-exp: evaluate exactly at knots.
    t_eval, q_eval = log_euclidean_spline(qs, ts, ts)
    assert np.allclose(t_eval, ts)
    for q_hat, qk in zip(q_eval, qs):
        assert geodesic_distance_s3(q_hat, qk) < 1e-10

