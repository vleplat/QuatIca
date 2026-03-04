
r"""
Quaternion trajectory interpolation on \(S^3\) (unit quaternions).

This module implements practical, efficient multi-keyframe interpolants for
orientation trajectories:

- **SLERP** (spherical linear interpolation) and **piecewise SLERP**
- **SQUAD** (Shoemake's quaternion spline)
- **Log–exp interpolation** (log-Euclidean spline on \(S^3\))

The main goal is to produce trajectories that (i) hit all keyframes exactly and
(ii) yield smooth transitions of angular velocity, compared to a piecewise SLERP
baseline which is typically only \(C^0\) at knot points.

Conventions
-----------
We work with unit quaternions represented by `numpy-quaternion` (`np.quaternion`).
For a differentiable trajectory \(q(t)\in S^3\), a common definition of body
angular velocity is:

\[
\omega(t) = 2\,\dot q(t)\,q(t)^{-1}\in\mathbb{R}^3
\]

In discrete time, a standard estimate uses the logarithm map:

\[
\omega(t_k)\approx \frac{2}{\Delta t}\,\log\!\big(q(t_{k+1})\,q(t_k)^{-1}\big)
\]

The quaternion logarithm/exponential used here are the \(S^3\) log/exp (not the
SO(3) rotation-vector conventions): if \(q = \cos(a) + u\sin(a)\) with
\(a\in[0,\pi]\), then \(\log(q) = u\,a\) (a pure quaternion) and
\(\exp(u\,a) = \cos(a) + u\sin(a)\).
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

import quaternion  # numpy-quaternion dtype

try:  # SciPy is a QuatIca dependency, but keep a small fallback for robustness.
    from scipy.interpolate import CubicSpline  # type: ignore
except Exception:  # pragma: no cover
    CubicSpline = None

_EPS = 1e-12

QuatLike = Union["quaternion.quaternion", np.quaternion]
ArrayQuat = np.ndarray

__all__ = [
    "enforce_sign_continuity",
    "unwrap_quaternion_signs",
    "slerp",
    "interpolate_piecewise_slerp",
    "squad_control_quaternions",
    "squad_segment",
    "interpolate_squad",
    "quat_log_unit",
    "quat_exp_pure",
    "log_euclidean_spline",
    "rotate_vector",
    "estimate_omega",
    "smoothness_energy",
    "velocity_jump_at_keyframes",
    "geodesic_distance_s3",
    "keyframe_errors_geodesic",
]

def _as_float4(q: QuatLike) -> np.ndarray:
    """Return the quaternion components as a float array `(w, x, y, z)`."""
    return np.array(quaternion.as_float_array(q), dtype=float).reshape(4)

def quat_dot(q1: QuatLike, q2: QuatLike) -> float:
    r"""Return the Euclidean dot product in \(\mathbb{R}^4\) between two quaternions."""
    return float(np.dot(_as_float4(q1), _as_float4(q2)))

def quat_norm(q: QuatLike) -> float:
    r"""Return the Euclidean norm in \(\mathbb{R}^4\)."""
    return float(np.linalg.norm(_as_float4(q)))

def quat_normalize(q: QuatLike) -> QuatLike:
    """Normalize a quaternion to unit length."""
    n = quat_norm(q)
    if n < _EPS:
        raise ValueError("Cannot normalize near-zero quaternion.")
    return q / n

def quat_conj(q: QuatLike) -> QuatLike:
    """Quaternion conjugate (for unit quaternions, this is the inverse)."""
    return np.conjugate(q)

def quat_inv_unit(q: QuatLike) -> QuatLike:
    """Inverse for a *unit* quaternion."""
    return quat_conj(q)

def enforce_sign_continuity(qs: Sequence[QuatLike]) -> list[QuatLike]:
    r"""
    Enforce sign continuity of a quaternion keyframe sequence.

    Because \(q\) and \(-q\) represent the same physical rotation (double cover
    \(S^3\to SO(3)\)), keyframes must be put on a consistent "sheet" to avoid
    artificial \(\pi\)-jumps. This routine flips \(q_i\) when needed so that
    successive dot products satisfy \(\langle q_{i-1}, q_i\rangle \ge 0\).

    Parameters
    ----------
    qs:
        Sequence of (approximately) unit quaternions.

    Returns
    -------
    list[quaternion]:
        New list with sign-consistent quaternions.
    """
    if len(qs) <= 1:
        return list(qs)
    out = [quat_normalize(qs[0])]
    for i in range(1, len(qs)):
        qi = quat_normalize(qs[i])
        if quat_dot(out[-1], qi) < 0.0:
            qi = -qi
        out.append(qi)
    return out


# Backward-compatible alias (older notebooks / drafts).
unwrap_quaternion_signs = enforce_sign_continuity

def slerp(q0: QuatLike, q1: QuatLike, t: float) -> QuatLike:
    r"""
    Spherical linear interpolation (SLERP) on \(S^3\).

    After enforcing the shortest-path sign convention, SLERP follows the
    shortest great-circle arc between unit quaternions \(q_0\) and \(q_1\).

    Parameters
    ----------
    q0, q1:
        Unit quaternions (will be normalized defensively).
    t:
        Interpolation parameter in \([0,1]\).

    Returns
    -------
    quaternion:
        Interpolated unit quaternion.
    """
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)

    # Shortest-path sign.
    if quat_dot(q0, q1) < 0.0:
        q1 = -q1

    dot = float(np.clip(quat_dot(q0, q1), -1.0, 1.0))
    theta = math.acos(dot)  # in [0, pi]
    if theta < 1e-8:
        # Small angle: linear interpolation + renormalize is stable.
        return quat_normalize((1.0 - t) * q0 + t * q1)

    s = math.sin(theta)
    w0 = math.sin((1.0 - t) * theta) / s
    w1 = math.sin(t * theta) / s
    return quat_normalize(w0 * q0 + w1 * q1)

def quat_log_unit(q: QuatLike) -> QuatLike:
    r"""
    Quaternion logarithm on \(S^3\) for a unit quaternion.

    For \(q = \cos(a) + u\sin(a)\) with \(a\in[0,\pi]\), returns
    \(\log(q) = u\,a\), encoded as a pure quaternion (zero real part).

    Parameters
    ----------
    q:
        Unit quaternion (will be normalized defensively).

    Returns
    -------
    quaternion:
        Pure quaternion representing the tangent vector in the Lie algebra.
    """
    q = quat_normalize(q)
    a = _as_float4(q)
    w = float(np.clip(a[0], -1.0, 1.0))
    v = np.array(a[1:], dtype=float)
    nv = float(np.linalg.norm(v))
    if nv < 1e-14:
        return quaternion.quaternion(0.0, 0.0, 0.0, 0.0)
    ang = math.atan2(nv, w)  # in [0, pi]
    u = v / nv
    vec = u * ang
    return quaternion.quaternion(0.0, vec[0], vec[1], vec[2])

def quat_exp_pure(p: QuatLike) -> QuatLike:
    r"""
    Quaternion exponential of a pure quaternion.

    If \(p = u\,a\) (pure quaternion, i.e. zero real part), returns
    \(\exp(p) = \cos(a) + u\sin(a)\).

    Parameters
    ----------
    p:
        Pure quaternion (real part ignored; vector part used).

    Returns
    -------
    quaternion:
        Unit quaternion on \(S^3\).
    """
    a = _as_float4(p)
    v = np.array(a[1:], dtype=float)
    ang = float(np.linalg.norm(v))
    if ang < 1e-14:
        return quaternion.quaternion(1.0, 0.0, 0.0, 0.0)
    u = v / ang
    w = math.cos(ang)
    s = math.sin(ang)
    vec = u * s
    return quaternion.quaternion(w, vec[0], vec[1], vec[2])

def squad_control_quaternions(qs):
    r"""
    Compute SQUAD control quaternions \(a_i\) from keyframes \(q_i\).

    For interior points \(i=1,\dots,n-1\), Shoemake's construction is:

    \[
    a_i
    = q_i\,\exp\!\left(
    -\frac14\Big(\log(q_i^{-1}q_{i-1})+\log(q_i^{-1}q_{i+1})\Big)
    \right)
    \]

    Endpoints use the common convention \(a_0=q_0\), \(a_n=q_n\).

    Parameters
    ----------
    qs:
        Keyframes as a sequence of unit quaternions.

    Returns
    -------
    list[quaternion]:
        Control quaternions of the same length as `qs`.
    """
    qs = enforce_sign_continuity(qs)
    n = len(qs) - 1
    if n < 1:
        return qs

    a = [None] * (n + 1)
    a[0] = qs[0]
    a[n] = qs[n]
    for i in range(1, n):
        qi = qs[i]
        qim1 = qs[i-1]
        qip1 = qs[i+1]
        term = quat_log_unit(quat_inv_unit(qi) * qim1) + quat_log_unit(quat_inv_unit(qi) * qip1)
        ai = qi * quat_exp_pure(-0.25 * term)
        a[i] = quat_normalize(ai)
    return a

def squad_segment(qi, qip1, ai, aip1, u: float):
    r"""
    Evaluate SQUAD on a single segment.

    Parameters
    ----------
    qi, qip1:
        Segment endpoint keyframes.
    ai, aip1:
        Precomputed SQUAD control quaternions for the endpoints.
    u:
        Local segment parameter in \([0,1]\).

    Returns
    -------
    quaternion:
        Interpolated unit quaternion.
    """
    s1 = slerp(qi, qip1, u)
    s2 = slerp(ai, aip1, u)
    return slerp(s1, s2, 2.0*u*(1.0-u))

def interpolate_piecewise_slerp(qs, ts, samples_per_seg=50):
    r"""
    Sample a piecewise SLERP trajectory through quaternion keyframes.

    Parameters
    ----------
    qs:
        Quaternion keyframes, length \(K\ge 2\).
    ts:
        Knot times (strictly increasing), same length as `qs`.
    samples_per_seg:
        Number of uniform samples per segment (including endpoints).

    Returns
    -------
    (t_path, q_path):
        - `t_path`: 1D float array of sample times in \([t_0,t_{K-1}]\)
        - `q_path`: 1D object array of quaternions at the sampled times
    """
    qs = enforce_sign_continuity(qs)
    ts = np.asarray(ts, dtype=float)
    out_t = []
    out_q = []
    nseg = len(qs) - 1
    for i in range(nseg):
        t0, t1 = ts[i], ts[i+1]
        for m in range(samples_per_seg + 1):
            # avoid duplicating knot samples except at the final endpoint
            if i < nseg - 1 and m == samples_per_seg:
                continue
            u = m / samples_per_seg
            out_t.append((1-u)*t0 + u*t1)
            out_q.append(slerp(qs[i], qs[i+1], u))
    return np.array(out_t), np.array(out_q, dtype=object)

def interpolate_squad(qs, ts, samples_per_seg=50):
    r"""
    Sample a SQUAD trajectory through quaternion keyframes.

    Parameters
    ----------
    qs:
        Quaternion keyframes, length \(K\ge 2\).
    ts:
        Knot times (strictly increasing), same length as `qs`.
    samples_per_seg:
        Number of uniform samples per segment (including endpoints).

    Returns
    -------
    (t_path, q_path):
        Sampled times and quaternions (same format as `interpolate_piecewise_slerp`).
    """
    qs = enforce_sign_continuity(qs)
    ts = np.asarray(ts, dtype=float)
    a = squad_control_quaternions(qs)
    out_t = []
    out_q = []
    nseg = len(qs) - 1
    for i in range(nseg):
        t0, t1 = ts[i], ts[i+1]
        for m in range(samples_per_seg + 1):
            if i < nseg - 1 and m == samples_per_seg:
                continue
            u = m / samples_per_seg
            out_t.append((1-u)*t0 + u*t1)
            out_q.append(squad_segment(qs[i], qs[i+1], a[i], a[i+1], u))
    return np.array(out_t), np.array(out_q, dtype=object)

# ----------------------
# Natural cubic spline in R^d (componentwise fallback)
# ----------------------
@dataclass
class NaturalCubicSpline1D:
    x: np.ndarray  # knots
    a: np.ndarray  # y values
    b: np.ndarray
    c: np.ndarray
    d: np.ndarray

    def eval(self, xq):
        xq = np.asarray(xq)
        yq = np.empty_like(xq, dtype=float)
        # find segment indices
        idx = np.searchsorted(self.x, xq, side='right') - 1
        idx = np.clip(idx, 0, len(self.x)-2)
        dx = xq - self.x[idx]
        yq = self.a[idx] + self.b[idx]*dx + self.c[idx]*dx**2 + self.d[idx]*dx**3
        return yq

def natural_cubic_spline_1d(x, y):
    """Natural cubic spline through knots (fallback when SciPy is unavailable)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x) - 1
    if n < 1:
        raise ValueError("Need at least two knots.")
    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("x must be strictly increasing.")

    # Solve tridiagonal for second derivatives m (natural: m0=mn=0)
    A = np.zeros((n+1, n+1), dtype=float)
    rhs = np.zeros(n+1, dtype=float)
    A[0,0] = 1.0
    A[n,n] = 1.0
    for i in range(1, n):
        A[i, i-1] = h[i-1]
        A[i, i]   = 2.0*(h[i-1] + h[i])
        A[i, i+1] = h[i]
        rhs[i] = 6.0*((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])
    m = np.linalg.solve(A, rhs)

    # Convert to polynomial pieces on each interval [x_i, x_{i+1}]
    a = y[:-1].copy()
    b = (y[1:]-y[:-1])/h - h*(2*m[:-1] + m[1:])/6.0
    c = m[:-1]/2.0
    d = (m[1:]-m[:-1])/(6.0*h)
    return NaturalCubicSpline1D(x=x, a=a, b=b, c=c, d=d)

@dataclass
class NaturalCubicSplineVec:
    splines: list  # list of 1D splines for each coordinate

    def eval(self, t):
        vals = np.stack([sp.eval(t) for sp in self.splines], axis=-1)
        return vals

def natural_cubic_spline_vec(t_knots, p_knots):
    p_knots = np.asarray(p_knots, dtype=float)
    d = p_knots.shape[1]
    spl = [natural_cubic_spline_1d(t_knots, p_knots[:,j]) for j in range(d)]
    return NaturalCubicSplineVec(spl)

def log_euclidean_spline(qs, ts, t_query):
    r"""
    Log–exp (log-Euclidean) spline interpolation on \(S^3\).

    Choose a reference quaternion \(q_\mathrm{ref}\) (here: `qs[0]`), map each
    keyframe into \(\mathbb{R}^3\) using the \(S^3\) logarithm:

    \[
    p_i := \log(q_\mathrm{ref}^{-1} q_i)\in\mathbb{R}^3,
    \]

    fit a cubic spline \(p(t)\) through \((t_i, p_i)\) in \(\mathbb{R}^3\), and
    map back via \(q(t)=q_\mathrm{ref}\exp(p(t))\).

    Parameters
    ----------
    qs:
        Quaternion keyframes (unit quaternions).
    ts:
        Knot times (strictly increasing), same length as `qs`.
    t_query:
        Times at which to evaluate the interpolant.

    Returns
    -------
    (t_query, q_path):
        `q_path` is an object array of unit quaternions of the same length as `t_query`.
    """
    qs = enforce_sign_continuity(qs)
    ts = np.asarray(ts, dtype=float)
    t_query = np.asarray(t_query, dtype=float)
    qref = qs[0]
    # map keyframes to R^3
    p = []
    for qi in qs:
        dq = quat_inv_unit(qref) * qi
        li = quat_log_unit(dq)
        a = _as_float4(li)
        p.append(a[1:])
    p = np.asarray(p, dtype=float)

    if CubicSpline is not None:
        spline = CubicSpline(ts, p, axis=0, bc_type="natural")
        p_t = spline(t_query)
    else:  # pragma: no cover
        spline = natural_cubic_spline_vec(ts, p)
        p_t = spline.eval(t_query)

    out = []
    for v in p_t:
        pure = quaternion.quaternion(0.0, v[0], v[1], v[2])
        out.append(quat_normalize(qref * quat_exp_pure(pure)))
    return np.asarray(t_query, dtype=float), np.array(out, dtype=object)

# ----------------------
# Metrics + visualization helpers
# ----------------------
def rotate_vector(q, v):
    r"""
    Rotate a 3D vector by a unit quaternion.

    Uses the standard formula \(v' = q(0,v)q^{-1}\).

    Parameters
    ----------
    q:
        Unit quaternion.
    v:
        3D vector.

    Returns
    -------
    np.ndarray:
        Rotated vector in \(\mathbb{R}^3\).
    """
    v = np.asarray(v, dtype=float).reshape(3,)
    pv = quaternion.quaternion(0.0, v[0], v[1], v[2])
    rv = q * pv * quat_inv_unit(q)
    a = _as_float4(rv)
    return a[1:]  # vector part

def estimate_omega(q_path, t_path):
    r"""
    Estimate body angular velocity from sampled quaternions.

    For samples \((t_k, q_k)\), we compute mid-point estimates:

    \[
    \omega(t_k)\approx \frac{2}{\Delta t}\log(q_{k+1}q_k^{-1})\in\mathbb{R}^3.
    \]

    Parameters
    ----------
    q_path:
        Sequence of unit quaternions.
    t_path:
        Strictly increasing sample times.

    Returns
    -------
    (t_mid, omega):
        - `t_mid`: times at midpoints of each sample interval
        - `omega`: array of shape `(len(q_path)-1, 3)`
    """
    q_path = list(q_path)
    t_path = np.asarray(t_path, dtype=float)
    omegas = []
    times = []
    for k in range(len(q_path)-1):
        dt = t_path[k+1] - t_path[k]
        dq = q_path[k+1] * quat_inv_unit(q_path[k])
        l = quat_log_unit(dq)
        a = _as_float4(l)[1:]
        omegas.append((2.0/dt) * a)
        times.append(0.5*(t_path[k+1] + t_path[k]))
    return np.asarray(times), np.asarray(omegas)

def smoothness_energy(q_path, t_path):
    r"""
    Heuristic smoothness energy based on discrete angular acceleration.

    With \(\omega_k\) from `estimate_omega`, we estimate:

    \[
    E \approx \sum_k \left\|\frac{\omega_{k+1}-\omega_k}{\Delta t_k}\right\|^2 \Delta t_k.
    \]
    """
    t_om, om = estimate_omega(q_path, t_path)
    if len(om) < 2:
        return 0.0
    dt = np.diff(t_om)
    dom = np.diff(om, axis=0)
    # use local dt (same length as dom)
    acc = dom / dt[:,None]
    return float(np.sum(np.sum(acc**2, axis=1) * dt))

def velocity_jump_at_keyframes(qs, ts, method_fn, samples_per_seg=200):
    r"""
    Estimate the maximum jump of angular velocity across knot points.

    This is mainly meaningful for piecewise constructions (e.g. piecewise SLERP),
    where the velocity is generally discontinuous at knots.

    Parameters
    ----------
    qs, ts:
        Keyframes and knot times.
    method_fn:
        Callable of signature `(qs, ts, samples_per_seg=...) -> (t_path, q_path)`.
    samples_per_seg:
        Sampling density per segment used to approximate left/right limits.

    Returns
    -------
    float:
        Estimated maximum \(\|\omega^+ - \omega^-\|\) across interior knots.
    """
    # sample densely
    t_path, q_path = method_fn(qs, ts, samples_per_seg=samples_per_seg)
    t_om, om = estimate_omega(q_path, t_path)

    jumps = []
    # knot times ts[1:-1]
    for knot in ts[1:-1]:
        # left omega: last omega with t_om < knot
        left_idx = np.where(t_om < knot)[0]
        right_idx = np.where(t_om > knot)[0]
        if len(left_idx)==0 or len(right_idx)==0:
            continue
        ol = om[left_idx[-1]]
        orr = om[right_idx[0]]
        jumps.append(np.linalg.norm(orr - ol))
    return float(np.max(jumps)) if jumps else 0.0


def geodesic_distance_s3(q0: QuatLike, q1: QuatLike) -> float:
    r"""
    Geodesic distance on \(S^3\) between unit quaternions.

    With the shortest-path sign convention, returns \(\theta\in[0,\pi]\) where
    \(\cos\theta = \langle q_0, q_1\rangle\) (dot in \(\mathbb{R}^4\)).
    """
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    d = quat_dot(q0, q1)
    d = abs(d)  # shortest sheet
    return float(2.0 * math.asin(min(1.0, math.sqrt(max(0.0, 0.5 * (1.0 - d))))))


def keyframe_errors_geodesic(
    t_path: np.ndarray,
    q_path: Sequence[QuatLike],
    qs_key: Sequence[QuatLike],
    ts_key: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute geodesic errors at keyframe times using nearest-sample evaluation.

    This is a lightweight diagnostic used in the demo scripts to verify that
    sampled trajectories hit keyframes (up to sampling resolution).

    Returns:
        errors_rad: 1D array of errors in radians, length `len(qs_key)`.
        indices: indices into `t_path` used for the nearest samples.
    """
    t_path = np.asarray(t_path, dtype=float)
    ts_key = np.asarray(ts_key, dtype=float)
    q_path = list(q_path)
    qs_key = list(qs_key)
    idx = np.array([int(np.argmin(np.abs(t_path - tk))) for tk in ts_key], dtype=int)
    errs = np.array(
        [geodesic_distance_s3(q_path[i], qs_key[k]) for k, i in enumerate(idx)],
        dtype=float,
    )
    return errs, idx



