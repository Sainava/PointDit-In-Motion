"""Joint estimate of each frame's depth scale and shift from parallax between metrically posed frames, with no GT.

Each frame k has depth_k = s_k * raw_z_k + t_k. Points of k on a regular pixel subsample are moved into every other
frame j with the known relative pose and compared with frame j's depth sampled bilinearly at their projection, as the
relative residual (z_in_j - depth_j) / depth_j. That residual alone is smallest for a flat scene far away, where
prediction noise and parallax both vanish relative to depth, so it is rescaled by depth_j / (s_j * IQR of raw_z_j),
which gives the raw prediction noise a size that no affine can shrink."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from pim.consistency.warp import relative_pose

# Weight of the ridge that pulls each frame's (log s_k, t_k / median depth) toward the window mean, against the data
# term, a mean of soft L1 losses that is about r**2 for a residual r. Fixed a priori, never tuned on results.
RIDGE_WEIGHT = 0.01
RIDGE_SENSITIVITY = (0.1, 10.0)
REJECTED = 1e3


@dataclass(frozen=True)
class JointScaleConfig:
    samples: int = 1500
    ridge_weight: float = RIDGE_WEIGHT
    loss_scale: float = 0.05
    """Residual, in units of the frame's interquartile depth range, where the soft L1 loss turns linear."""
    visibility: float = 0.5
    """A moved point is visible in frame j when its residual is at most this many interquartile depth ranges, the
    consistency metrics' 0.15 relative threshold at a typical indoor depth spread of 0.3; beyond it the loss is
    constant."""
    widening: tuple[float, ...] = (4.0, 2.0, 1.0)
    """Multiples of the visibility limit for the grid (the first) and successive refinements, wide to narrow, so
    that a shared start far from the per frame values still lies inside the basin."""
    margin: float = 0.1
    """Share of the width and height left out of the subsample on each side, so that whether a sample stays in
    view barely depends on scale."""
    grid_size: int = 15
    depth_range: tuple[float, float] = (0.3, 12.0)
    """Median metric depth bracket of the shared grid, log spaced, in metres."""
    near_ratio_range: tuple[float, float] = (0.05, 0.95)
    """Bracket of depth at the 1st percentile of raw z over median depth, which keeps the grid's depth positive."""
    min_baseline_ratio: float = 0.005
    rotation_dominance: float = 0.1
    """Low parallax when baseline over depth is below this fraction of the rotation angle in radians."""
    allow_low_parallax: bool = False
    """Refine per frame values even in low parallax windows."""
    max_iterations: int = 30
    tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if not self.widening or any(w <= 0 for w in self.widening):
            raise ValueError(f"widening must be a non-empty sequence of positive multiples, got {self.widening}")
        if self.samples < 1 or self.grid_size < 2 or self.max_iterations < 0:
            raise ValueError("samples must be positive, grid_size at least 2 and max_iterations non-negative")
        if not (self.ridge_weight >= 0 and self.loss_scale > 0 and self.visibility > 0 and 0 <= self.margin < 0.5):
            raise ValueError("ridge_weight must be non-negative, loss_scale and visibility positive and margin in "
                             "[0, 0.5)")
        low, high = self.depth_range
        if not 0 < low < high:
            raise ValueError(f"depth_range must be increasing and positive, got {self.depth_range}")
        low, high = self.near_ratio_range
        if not 0 < low < high < 1:
            raise ValueError(f"near_ratio_range must lie inside (0, 1), got {self.near_ratio_range}")


@dataclass(frozen=True)
class JointScaleResult:
    scale: np.ndarray
    shift: np.ndarray
    cost: float
    low_parallax: bool
    iterations: int
    baseline_ratio: float
    """Median over ordered pairs of baseline in metres over the shared median depth."""
    rotation_ratio: float
    """Median over pairs of baseline over depth divided by the rotation angle in radians; inf without rotation."""
    shared_scale: float
    shared_shift: float
    depth_unit: float
    """Median depth of the shared affine, the unit of the shift in the parameter vector."""
    valid_fraction: float
    per_frame: bool
    """False when the shared grid affine was returned for every frame."""

    def depth(self, k: int, raw_z: np.ndarray) -> np.ndarray:
        return (self.scale[k] * raw_z.astype(np.float64) + self.shift[k]).astype(np.float32)


def solve_joint_scale(raw_z: Sequence[np.ndarray], poses: Sequence[np.ndarray], K: np.ndarray,
                      config: JointScaleConfig = JointScaleConfig(), start: JointScaleResult | None = None
                      ) -> JointScaleResult:
    """start skips the grid and parallax test and refines from its values at the final visibility limit, with the
    same depth unit."""
    problem = _Problem(raw_z, poses, K, config)
    if start is not None:
        if len(start.scale) != problem.n:
            raise ValueError(f"start has {len(start.scale)} frames, expected {problem.n}")
        problem.unit = start.depth_unit
        theta, cost, iterations = problem.refine(problem.theta(start.scale, start.shift), (1.0,))
        return problem.result(theta, cost, iterations, start.low_parallax, start.baseline_ratio,
                              start.rotation_ratio, start.shared_scale, start.shared_shift, True)

    s, t = problem.grid_search()
    problem.unit = problem.shared_median_depth(s, t)
    baseline_ratio, rotation_ratio = problem.parallax(problem.unit)
    low = baseline_ratio < config.min_baseline_ratio or rotation_ratio < config.rotation_dominance
    theta = problem.theta(np.full(problem.n, s), np.full(problem.n, t))
    if low and not config.allow_low_parallax:
        return problem.result(theta, problem.cost(theta), 0, low, baseline_ratio, rotation_ratio, s, t, False)
    theta, cost, iterations = problem.refine(theta)
    return problem.result(theta, cost, iterations, low, baseline_ratio, rotation_ratio, s, t, True)


def ridge_sensitivity(raw_z: Sequence[np.ndarray], poses: Sequence[np.ndarray], K: np.ndarray,
                      result: JointScaleResult, config: JointScaleConfig = JointScaleConfig(),
                      factors: Sequence[float] = RIDGE_SENSITIVITY) -> dict[float, JointScaleResult]:
    """The solution refined from result with the ridge weight multiplied by each factor."""
    return {f: solve_joint_scale(raw_z, poses, K, replace(config, ridge_weight=config.ridge_weight * f), result)
            for f in factors}


def subsample(raw_z: np.ndarray, samples: int, margin: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """(v, u) of a regular grid of about samples pixels over the finite pixels of raw_z inside the margin."""
    height, width = raw_z.shape
    top, left = int(round(margin * height)), int(round(margin * width))
    finite = np.zeros(raw_z.shape, dtype=bool)
    finite[top:height - top, left:width - left] = np.isfinite(raw_z[top:height - top, left:width - left])
    count = int(finite.sum())
    if count == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    stride = max(1, int(round(math.sqrt(count / samples))))
    v, u = np.mgrid[top + stride // 2:height - top:stride, left + stride // 2:width - left:stride]
    keep = finite[v, u]
    return v[keep].ravel(), u[keep].ravel()


def median_pixel_depth_error(scale: float, shift: float, ref_scale: float, ref_shift: float,
                             raw_z: np.ndarray) -> float:
    """Signed relative error of the depth at the median finite raw z against the reference affine."""
    z = float(np.median(raw_z[np.isfinite(raw_z)]))
    return (scale * z + shift) / (ref_scale * z + ref_shift) - 1.0


def fit_affine(raw_z: np.ndarray, depth: np.ndarray) -> tuple[float, float]:
    """Least squares (s, t) of depth = s * raw_z + t over pixels finite in both and positive in depth."""
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(raw_z) & np.isfinite(depth) & (depth > 0)
    if valid.sum() < 2:
        return math.nan, math.nan
    z = raw_z[valid].astype(np.float64)
    design = np.stack([z, np.ones_like(z)], axis=1)
    s, t = np.linalg.lstsq(design, depth[valid].astype(np.float64), rcond=None)[0]
    return float(s), float(t)


class _Problem:
    def __init__(self, raw_z: Sequence[np.ndarray], poses: Sequence[np.ndarray], K: np.ndarray,
                 config: JointScaleConfig) -> None:
        if len(raw_z) != len(poses):
            raise ValueError(f"{len(raw_z)} raw z maps but {len(poses)} poses")
        if len(raw_z) < 2:
            raise ValueError(f"the joint scale solver needs at least two frames, got {len(raw_z)}")
        shape = raw_z[0].shape
        if any(z.ndim != 2 or z.shape != shape for z in raw_z):
            raise ValueError("raw z maps must all be (H, W) of one shape")
        if min(shape) < 2:
            raise ValueError(f"raw z maps must be at least 2x2, got {shape}")
        if any(np.asarray(T).shape != (4, 4) or not np.all(np.isfinite(T)) for T in poses):
            raise ValueError("poses must be finite 4x4")
        if K.shape != (3, 3) or not (K[0, 0] > 0 and K[1, 1] > 0):
            raise ValueError("K must be 3x3 with positive focal lengths")
        self.config, self.n, self.shape, self.K = config, len(raw_z), shape, K.astype(np.float64)
        self.raw = [np.asarray(z, dtype=np.float64) for z in raw_z]
        self.samples_z, self.rays = [], []
        for z in self.raw:
            v, u = subsample(z, config.samples, config.margin)
            if v.size == 0:
                raise ValueError("a raw z map has no finite pixel")
            self.samples_z.append(z[v, u])
            self.rays.append(np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones(v.size)], axis=1))
        spreads = [np.subtract(*np.percentile(z, [75, 25])) for z in self.samples_z]
        if not all(s > 0 for s in spreads):
            raise ValueError("every raw z map needs positive interquartile spread over its subsample")
        self.spreads = np.array(spreads)
        self.pairs = [(k, j) for k in range(self.n) for j in range(self.n) if k != j]
        self.relative = [relative_pose(np.asarray(poses[k], np.float64), np.asarray(poses[j], np.float64))
                         for k, j in self.pairs]
        # Every pair's samples in one flat array, so that a cost is a handful of vector operations.
        sizes = [self.samples_z[k].size for k, _ in self.pairs]
        self.src = np.repeat([k for k, _ in self.pairs], sizes)
        self.dst = np.repeat([j for _, j in self.pairs], sizes)
        self.z = np.concatenate([self.samples_z[k] for k, _ in self.pairs])
        self.ray = np.concatenate([self.rays[k] for k, _ in self.pairs])
        self.R = np.repeat(np.stack([T[:3, :3] for T in self.relative]), sizes, axis=0)
        self.t = np.repeat(np.stack([T[:3, 3] for T in self.relative]), sizes, axis=0)
        self.stack = np.stack(self.raw)
        self.n_data = int(self.z.size)
        self.touching = [np.flatnonzero((self.src == f) | (self.dst == f)) for f in range(self.n)]
        self.unit = 1.0
        self.limit = config.visibility

    def theta(self, scale: np.ndarray, shift: np.ndarray) -> np.ndarray:
        return np.stack([np.log(scale), np.asarray(shift) / self.unit], axis=1).ravel()

    def affine(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pairs = theta.reshape(self.n, 2)
        return np.exp(pairs[:, 0]), pairs[:, 1] * self.unit

    def _rho(self, r: np.ndarray) -> np.ndarray:
        d = self.config.loss_scale
        return 2 * d * d * (np.sqrt(1 + (r / d) ** 2) - 1)

    def _elements(self, index: np.ndarray | slice, scale: np.ndarray, shift: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
        """(residual, counted) of the flat samples at index. Samples on missing raw z of frame j are not counted.
        A sample that lands outside frame j, behind it or on non-positive depth is not seen and gets a residual
        beyond the visibility limit."""
        src, dst = self.src[index], self.dst[index]
        depth_k = scale[src] * self.z[index] + shift[src]
        p = np.einsum("nij,nj->ni", self.R[index], depth_k[:, None] * self.ray[index]) + self.t[index]
        z = p[:, 2]
        height, width = self.shape
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.K[0, 0] * p[:, 0] / z + self.K[0, 2]
            v = self.K[1, 1] * p[:, 1] / z + self.K[1, 2]
            inside = (depth_k > 0) & (z > 1e-9) & (u >= 0) & (u <= width - 1) & (v >= 0) & (v <= height - 1)
        u, v = np.where(inside, u, 0.0), np.where(inside, v, 0.0)
        u0 = np.minimum(np.floor(u).astype(np.int64), width - 2)
        v0 = np.minimum(np.floor(v).astype(np.int64), height - 2)
        fu, fv = u - u0, v - v0
        raw = self.stack
        sampled = ((1 - fu) * (1 - fv) * raw[dst, v0, u0] + fu * (1 - fv) * raw[dst, v0, u0 + 1]
                   + (1 - fu) * fv * raw[dst, v0 + 1, u0] + fu * fv * raw[dst, v0 + 1, u0 + 1])
        scale_j = scale[dst]
        depth_j = scale_j * sampled + shift[dst]
        counted = ~inside | np.isfinite(depth_j)
        with np.errstate(invalid="ignore"):
            seen = inside & counted & (depth_j > 0)
        # The relative residual (z - depth_j) / depth_j times depth_j / (s_j * IQR_j).
        r = np.where(seen, (z - depth_j) / (scale_j * self.spreads[dst]), 2 * self.limit)
        return np.where(counted, r, 0.0), counted

    def _errors(self, r: np.ndarray, counted: np.ndarray) -> np.ndarray:
        # Mean loss over counted samples. A sample beyond the visibility limit is occluded, meets a discontinuity or
        # is not seen at all, and is charged the loss at the limit.
        count = int(counted.sum())
        if count == 0:
            return np.full(r.size, REJECTED / math.sqrt(self.n_data))
        clipped = np.where(counted, np.clip(r, -self.limit, self.limit), 0.0)
        return np.sign(clipped) * np.sqrt(self._rho(clipped)) / math.sqrt(count)

    def _ridge(self, theta: np.ndarray) -> np.ndarray:
        pairs = theta.reshape(self.n, 2)
        return math.sqrt(self.config.ridge_weight) * (pairs - pairs.mean(axis=0)).ravel()

    def _data(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._elements(slice(None), *self.affine(theta))

    def residuals(self, theta: np.ndarray, data: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
        r, counted = self._data(theta) if data is None else data
        return np.concatenate([self._errors(r, counted), self._ridge(theta)])

    def cost(self, theta: np.ndarray) -> float:
        return float(np.sum(self.residuals(theta) ** 2))

    def valid_fraction(self, theta: np.ndarray) -> float:
        """Share of all samples that are counted and visible at the final limit."""
        r, counted = self._data(theta)
        return float((counted & (np.abs(r) <= self.config.visibility)).mean())

    def jacobian(self, theta: np.ndarray, data: tuple[np.ndarray, np.ndarray], e: np.ndarray,
                 step: float = 1e-5) -> np.ndarray:
        # A frame's parameters move only the samples of pairs that touch it.
        r, counted = data
        J = np.zeros((e.size, theta.size))
        for p in range(theta.size):
            moved = theta.copy()
            moved[p] += step
            index = self.touching[p // 2]
            r_moved, counted_moved = r.copy(), counted.copy()
            r_moved[index], counted_moved[index] = self._elements(index, *self.affine(moved))
            J[:, p] = (self.residuals(moved, (r_moved, counted_moved)) - e) / step
        return J

    def grid_search(self) -> tuple[float, float]:
        pooled = np.concatenate(self.samples_z)
        z_near, z_median = np.percentile(pooled, [1, 50])
        if not z_median > z_near:
            raise ValueError("raw z has no spread between its 1st percentile and median")
        best, best_cost = (math.nan, math.nan), math.inf
        size = self.config.grid_size
        self.limit = self.config.visibility * self.config.widening[0]
        for depth in np.geomspace(*self.config.depth_range, size):
            for ratio in np.linspace(*self.config.near_ratio_range, size):
                s = depth * (1 - ratio) / (z_median - z_near)
                t = depth - s * z_median
                cost = self.cost(self.theta(np.full(self.n, s), np.full(self.n, t)))
                if cost < best_cost:
                    best, best_cost = (float(s), float(t)), cost
        self.limit = self.config.visibility
        return best

    def shared_median_depth(self, s: float, t: float) -> float:
        return float(np.median(s * np.concatenate(self.samples_z) + t))

    def parallax(self, depth: float) -> tuple[float, float]:
        baselines = np.array([np.linalg.norm(T[:3, 3]) for T in self.relative])
        angles = np.array([math.acos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1.0, 1.0)) for T in self.relative])
        ratio = baselines / depth
        rotation = np.full(ratio.shape, math.inf)
        turning = angles > 1e-12
        rotation[turning] = ratio[turning] / angles[turning]
        return float(np.median(ratio)), float(np.median(rotation))

    def refine(self, theta: np.ndarray, widening: Sequence[float] | None = None) -> tuple[np.ndarray, float, int]:
        """Refinement at each widening in turn, by default the configured ones; the cost is at the final limit."""
        total = 0
        for widening in self.config.widening if widening is None else widening:
            self.limit = self.config.visibility * widening
            theta, cost, iterations = self._refine_at(theta)
            total += iterations
        self.limit = self.config.visibility
        return theta, self.cost(theta), total

    def _refine_at(self, theta: np.ndarray) -> tuple[np.ndarray, float, int]:
        data = self._data(theta)
        e = self.residuals(theta, data)
        cost = float(e @ e)
        damping, iterations = 1e-3, 0
        while iterations < self.config.max_iterations:
            J = self.jacobian(theta, data, e)
            A, g = J.T @ J, J.T @ e
            diagonal = np.diag(A) + 1e-12
            accepted = None
            while damping < 1e10:
                step = np.linalg.solve(A + damping * np.diag(diagonal), -g)
                trial = theta + step
                data_trial = self._data(trial)
                e_trial = self.residuals(trial, data_trial)
                cost_trial = float(e_trial @ e_trial)
                if cost_trial < cost:
                    accepted = trial, data_trial, e_trial, cost_trial
                    damping = max(damping / 3, 1e-9)
                    break
                damping *= 4
            if accepted is None:
                break
            iterations += 1
            gain = cost - accepted[3]
            theta, data, e, cost = accepted
            if gain <= self.config.tolerance * cost or np.max(np.abs(step)) < 1e-9:
                break
        return theta, cost, iterations

    def result(self, theta: np.ndarray, cost: float, iterations: int, low: bool, baseline_ratio: float,
               rotation_ratio: float, s: float, t: float, per_frame: bool) -> JointScaleResult:
        scale, shift = self.affine(theta)
        return JointScaleResult(scale, shift, cost, low, iterations, baseline_ratio, rotation_ratio, s, t,
                                self.unit, self.valid_fraction(theta), per_frame)
