from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - fallback path only matters when numba is absent
    njit = None
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=True)
    def _distance_matrix_numba(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        out = np.empty((x.shape[0], y.shape[0]), dtype=np.float64)
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                accum = 0.0
                for d in range(x.shape[1]):
                    diff = x[i, d] - y[j, d]
                    accum += diff * diff
                out[i, j] = np.sqrt(max(accum, 0.0))
        return out


    @njit(cache=True, fastmath=True)
    def _phs_kernel_numba(r: np.ndarray, degree: int) -> np.ndarray:
        out = np.empty_like(r)
        even_degree = (degree % 2) == 0
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                rij = r[i, j]
                if even_degree:
                    out[i, j] = (rij**degree) * np.log(rij + 2.0e-16) if rij > 0.0 else 0.0
                else:
                    out[i, j] = rij**degree
        return out


    @njit(cache=True, fastmath=True)
    def _phs_dr_over_r_numba(r: np.ndarray, degree: int) -> np.ndarray:
        out = np.empty_like(r)
        even_degree = (degree % 2) == 0
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                rij = r[i, j]
                if even_degree:
                    value = (rij ** (degree - 2)) * (degree * np.log(rij + 2.0e-16) + 1.0)
                else:
                    value = degree * (rij ** (degree - 2))
                out[i, j] = value if np.isfinite(value) else 0.0
        return out


    @njit(cache=True, fastmath=True)
    def _phs_lap_numba(r: np.ndarray, degree: int, dim: int) -> np.ndarray:
        out = np.empty_like(r)
        even_degree = (degree % 2) == 0
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                rij = r[i, j]
                if even_degree:
                    logt = np.log(rij + 2.0e-16)
                    value = (rij ** (degree - 2)) * (
                        dim + 2.0 * degree + degree * degree * logt - 2.0 * degree * logt + dim * degree * logt - 2.0
                    )
                else:
                    value = degree * (dim + degree - 2.0) * (rij ** (degree - 2))
                out[i, j] = value if np.isfinite(value) else 0.0
        return out

else:
    _distance_matrix_numba = None
    _phs_kernel_numba = None
    _phs_dr_over_r_numba = None
    _phs_lap_numba = None


def dense_distance_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return np.zeros((x.shape[0], y.shape[0]), dtype=float)
    if NUMBA_AVAILABLE:
        return _distance_matrix_numba(x, y)
    x_sq = np.sum(x * x, axis=1, keepdims=True)
    y_sq = np.sum(y * y, axis=1, keepdims=True).T
    sq_dist = np.maximum(x_sq + y_sq - 2.0 * (x @ y.T), 0.0)
    return np.sqrt(sq_dist)


def phs_kernel_matrix(r: np.ndarray, degree: int) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    if r.size == 0:
        return np.zeros_like(r)
    if NUMBA_AVAILABLE:
        return _phs_kernel_numba(r, int(degree))
    if degree % 2 == 0:
        return np.where(r > 0, r**degree * np.log(r + 2e-16), 0.0)
    return r**degree


def phs_dr_over_r_matrix(r: np.ndarray, degree: int) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    if r.size == 0:
        return np.zeros_like(r)
    if NUMBA_AVAILABLE:
        return _phs_dr_over_r_numba(r, int(degree))
    if degree % 2 == 0:
        d = r ** (degree - 2) * (degree * np.log(r + 2e-16) + 1)
    else:
        d = degree * r ** (degree - 2)
    d[~np.isfinite(d)] = 0.0
    return d


def phs_lap_matrix(r: np.ndarray, degree: int, dim: int) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    if r.size == 0:
        return np.zeros_like(r)
    if NUMBA_AVAILABLE:
        return _phs_lap_numba(r, int(degree), int(dim))
    if degree % 2 == 0:
        logt = np.log(r + 2e-16)
        l = r ** (degree - 2) * (dim + 2 * degree + degree**2 * logt - 2 * degree * logt + dim * degree * logt - 2)
    else:
        l = degree * (dim + degree - 2) * r ** (degree - 2)
    l[~np.isfinite(l)] = 0.0
    return l


if NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=True)
    def _normalize_stencil_points_numba(x: np.ndarray, width_floor: float) -> tuple[np.ndarray, np.ndarray, float]:
        xm = np.zeros(x.shape[1], dtype=np.float64)
        for i in range(x.shape[0]):
            for d in range(x.shape[1]):
                xm[d] += x[i, d]
        for d in range(x.shape[1]):
            xm[d] /= x.shape[0]

        width = 0.0
        for i in range(x.shape[0]):
            accum = 0.0
            for d in range(x.shape[1]):
                diff = x[i, d] - xm[d]
                accum += diff * diff
            dist = np.sqrt(accum)
            if dist > width:
                width = dist
        if width < width_floor:
            width = width_floor

        xc = np.empty_like(x)
        inv_width = 1.0 / width
        for i in range(x.shape[0]):
            for d in range(x.shape[1]):
                xc[i, d] = (x[i, d] - xm[d]) * inv_width
        return xm, xc, width


    @njit(cache=True, fastmath=True)
    def _build_augmented_rbf_lhs_numba(kernel: np.ndarray, poly: np.ndarray) -> np.ndarray:
        n = kernel.shape[0]
        npoly = poly.shape[1]
        out = np.zeros((n + npoly, n + npoly), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                out[i, j] = kernel[i, j]
        for i in range(n):
            for j in range(npoly):
                value = poly[i, j]
                out[i, n + j] = value
                out[n + j, i] = value
        return out

else:
    _normalize_stencil_points_numba = None
    _build_augmented_rbf_lhs_numba = None


if NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=True)
    def _legendre_recurrence_numba(n: int) -> tuple[np.ndarray, np.ndarray]:
        a = np.zeros(n, dtype=np.float64)
        b = np.ones(n, dtype=np.float64)
        if n > 0:
            b[0] = 2.0
        if n > 1:
            b[1] = 1.0 / 3.0
        for q in range(2, n):
            qq = float(q)
            b[q] = (qq * qq) / ((2.0 * qq + 1.0) * (2.0 * qq - 1.0))
        return a, b


    @njit(cache=True, fastmath=True)
    def _legendre_eval_numba(x: np.ndarray, n: int, d: int) -> np.ndarray:
        a, b = _legendre_recurrence_numba(n + 1)
        out = np.zeros((x.size, n + 1), dtype=np.float64)
        out[:, 0] = 1.0 / np.sqrt(b[0])
        if n > 0:
            out[:, 1] = ((x - a[0]) * out[:, 0]) / np.sqrt(b[1])
        for q in range(1, n):
            out[:, q + 1] = ((x - a[q]) * out[:, q] - np.sqrt(b[q]) * out[:, q - 1]) / np.sqrt(b[q + 1])

        if d == 0:
            return out

        cur = out
        for qd in range(1, d + 1):
            nxt = np.zeros_like(cur)
            for q in range(qd, n + 1):
                if q == qd:
                    denom = 1.0
                    for j in range(q + 1):
                        denom *= b[j]
                    const = 1.0
                    for j in range(2, qd + 1):
                        const *= float(j)
                    nxt[:, q] = const / np.sqrt(denom)
                else:
                    nxt[:, q] = ((x - a[q - 1]) * nxt[:, q - 1] - np.sqrt(b[q - 1]) * nxt[:, q - 2] + qd * cur[:, q - 1]) / np.sqrt(b[q])
            cur = nxt
        return cur


    @njit(cache=True, fastmath=True)
    def _legendre_tensor_evaluate_numba(x: np.ndarray, alpha: np.ndarray, d: np.ndarray) -> np.ndarray:
        max_alpha = 0
        for i in range(alpha.shape[0]):
            for j in range(alpha.shape[1]):
                if alpha[i, j] > max_alpha:
                    max_alpha = alpha[i, j]
        a0 = np.sqrt(2.0)
        out = np.ones((x.shape[0], alpha.shape[0], d.shape[0]), dtype=np.float64) / (a0 ** x.shape[1])
        for qd in range(d.shape[0]):
            for qdim in range(x.shape[1]):
                local_max = 0
                for i in range(alpha.shape[0]):
                    if alpha[i, qdim] > local_max:
                        local_max = alpha[i, qdim]
                temp = _legendre_eval_numba(x[:, qdim], local_max, int(d[qd, qdim]))
                for i in range(alpha.shape[0]):
                    degree = alpha[i, qdim]
                    if d[qd, qdim] == 0 and degree == 0:
                        continue
                    out[:, i, qd] *= temp[:, degree] * a0
        return out

else:
    _legendre_tensor_evaluate_numba = None


def legendre_tensor_evaluate(x: np.ndarray, alpha: np.ndarray, d: np.ndarray) -> np.ndarray | None:
    if not NUMBA_AVAILABLE:
        return None
    x = np.asarray(x, dtype=float)
    alpha = np.asarray(alpha, dtype=np.int64)
    d = np.asarray(d, dtype=np.int64)
    return _legendre_tensor_evaluate_numba(x, alpha, d)


def normalize_stencil_points(x: np.ndarray, width_floor: float = 1.0) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.zeros(x.shape[1], dtype=float), np.zeros_like(x), float(width_floor)
    if NUMBA_AVAILABLE:
        return _normalize_stencil_points_numba(x, float(width_floor))
    xm = x.mean(axis=0)
    width = max(float(np.linalg.norm(x - xm, axis=1).max(initial=0.0)), float(width_floor))
    xc = (x - xm) / width
    return xm, xc, width


def build_augmented_rbf_lhs(kernel: np.ndarray, poly: np.ndarray) -> np.ndarray:
    kernel = np.asarray(kernel, dtype=float)
    poly = np.asarray(poly, dtype=float)
    if NUMBA_AVAILABLE:
        return _build_augmented_rbf_lhs_numba(kernel, poly)
    n = kernel.shape[0]
    npoly = poly.shape[1]
    out = np.zeros((n + npoly, n + npoly), dtype=float)
    out[:n, :n] = kernel
    out[:n, n:] = poly
    out[n:, :n] = poly.T
    return out


if NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=True)
    def _divfree_gram_matrix_numba(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
        dim = x.shape[1]
        nx = x.shape[0]
        ny = y.shape[0]
        eps = np.finfo(np.float64).eps
        out = np.empty((dim * nx, dim * ny), dtype=np.float64)
        for i in range(nx):
            for j in range(ny):
                diff0 = x[i, 0] - y[j, 0]
                diff1 = x[i, 1] - y[j, 1]
                r2 = diff0 * diff0 + diff1 * diff1
                diff2 = 0.0
                if dim == 3:
                    diff2 = x[i, 2] - y[j, 2]
                    r2 += diff2 * diff2
                re = np.sqrt(r2) + eps
                re_m2 = re ** (degree - 2)
                cross_coeff = degree * (degree - 2) * (re ** (degree - 4))
                full_diag = -degree * (degree + dim - 2) * re_m2
                if dim == 2:
                    value00 = cross_coeff * diff0 * diff0 + degree * re_m2 + full_diag
                    value01 = cross_coeff * diff0 * diff1
                    value10 = value01
                    value11 = cross_coeff * diff1 * diff1 + degree * re_m2 + full_diag
                    out[i, j] = value00
                    out[i, ny + j] = value01
                    out[nx + i, j] = value10
                    out[nx + i, ny + j] = value11
                else:
                    value00 = cross_coeff * diff0 * diff0 + degree * re_m2 + full_diag
                    value01 = cross_coeff * diff0 * diff1
                    value02 = cross_coeff * diff0 * diff2
                    value10 = value01
                    value11 = cross_coeff * diff1 * diff1 + degree * re_m2 + full_diag
                    value12 = cross_coeff * diff1 * diff2
                    value20 = value02
                    value21 = value12
                    value22 = cross_coeff * diff2 * diff2 + degree * re_m2 + full_diag
                    out[i, j] = value00
                    out[i, ny + j] = value01
                    out[i, 2 * ny + j] = value02
                    out[nx + i, j] = value10
                    out[nx + i, ny + j] = value11
                    out[nx + i, 2 * ny + j] = value12
                    out[2 * nx + i, j] = value20
                    out[2 * nx + i, ny + j] = value21
                    out[2 * nx + i, 2 * ny + j] = value22
        return out

else:
    _divfree_gram_matrix_numba = None


def divfree_gram_matrix(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("x and y must be 2D arrays with matching column counts")
    if x.shape[1] not in {2, 3}:
        raise ValueError("divergence-free kernels are only supported in 2D and 3D")
    if x.size == 0 or y.size == 0:
        return np.zeros((x.shape[1] * x.shape[0], y.shape[1] * y.shape[0]), dtype=float)
    if NUMBA_AVAILABLE:
        return _divfree_gram_matrix_numba(x, y, int(degree))

    dim = x.shape[1]
    nx = x.shape[0]
    ny = y.shape[0]
    out = np.empty((dim * nx, dim * ny), dtype=float)
    diff = x[:, None, :] - y[None, :, :]
    re = np.sqrt(np.maximum(np.sum(diff * diff, axis=2), 0.0)) + np.finfo(float).eps
    re_m2 = re ** (degree - 2)
    cross_coeff = degree * (degree - 2) * (re ** (degree - 4))
    full_diag = -degree * (degree + dim - 2) * re_m2
    for a in range(dim):
        for b in range(dim):
            block = cross_coeff * diff[:, :, a] * diff[:, :, b]
            if a == b:
                block = block + degree * re_m2 + full_diag
            out[a * nx : (a + 1) * nx, b * ny : (b + 1) * ny] = block
    return out


if NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=True)
    def _surface_tangent_frame_numba(normal: np.ndarray) -> np.ndarray:
        eps = np.finfo(np.float64).eps
        norm = np.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
        unit = normal / max(norm, eps)
        dominant = 0
        if abs(unit[1]) > abs(unit[dominant]):
            dominant = 1
        if abs(unit[2]) > abs(unit[dominant]):
            dominant = 2
        first = np.zeros(3, dtype=np.float64)
        second = np.zeros(3, dtype=np.float64)
        if dominant == 0:
            first[1] = 1.0
            second[2] = 1.0
        elif dominant == 1:
            first[0] = 1.0
            second[2] = 1.0
        else:
            first[0] = 1.0
            second[1] = 1.0
        projection = unit[0] * first[0] + unit[1] * first[1] + unit[2] * first[2]
        tangent_one = first - projection * unit
        tangent_one /= max(np.sqrt(np.sum(tangent_one * tangent_one)), eps)
        projection_normal = unit[0] * second[0] + unit[1] * second[1] + unit[2] * second[2]
        projection_tangent = (
            tangent_one[0] * second[0]
            + tangent_one[1] * second[1]
            + tangent_one[2] * second[2]
        )
        tangent_two = second - projection_normal * unit - projection_tangent * tangent_one
        tangent_two /= max(np.sqrt(np.sum(tangent_two * tangent_two)), eps)
        frame = np.empty((3, 2), dtype=np.float64)
        frame[:, 0] = tangent_one
        frame[:, 1] = tangent_two
        return frame


    @njit(cache=True, fastmath=True, parallel=True)
    def _surface_local_systems_numba(
        points: np.ndarray,
        normals: np.ndarray,
        neighbors: np.ndarray,
        index_set: np.ndarray,
        phs_degree: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        row_count = points.shape[0]
        stencil_size = neighbors.shape[1]
        polynomial_count = index_set.shape[0]
        system_size = stencil_size + polynomial_count
        lhs_all = np.zeros((row_count, system_size, system_size), dtype=np.float64)
        rhs_all = np.zeros((row_count, system_size, 3), dtype=np.float64)
        frames = np.empty((row_count, 3, 2), dtype=np.float64)
        derivative_orders = np.array([[2, 0], [0, 2], [1, 0], [0, 1]], dtype=np.int64)
        zero_derivative = np.zeros((1, 2), dtype=np.int64)
        eps = np.finfo(np.float64).eps
        for row in prange(row_count):
            frame = _surface_tangent_frame_numba(normals[row])
            frames[row] = frame
            center = points[neighbors[row, 0]]
            scaled = np.empty((stencil_size, 2), dtype=np.float64)
            width = 0.0
            for i in range(stencil_size):
                point = points[neighbors[row, i]] - center
                for axis in range(2):
                    value = point[0] * frame[0, axis] + point[1] * frame[1, axis] + point[2] * frame[2, axis]
                    scaled[i, axis] = value
                    width = max(width, abs(value))
            width = max(width, eps)
            scaled /= width
            polynomial = _legendre_tensor_evaluate_numba(
                scaled, index_set, zero_derivative
            )[:, :, 0]
            for i in range(stencil_size):
                for j in range(stencil_size):
                    dx = scaled[i, 0] - scaled[j, 0]
                    dy = scaled[i, 1] - scaled[j, 1]
                    radius = np.sqrt(dx * dx + dy * dy)
                    lhs_all[row, i, j] = (radius + eps) ** phs_degree
                for j in range(polynomial_count):
                    value = polynomial[i, j]
                    lhs_all[row, i, stencil_size + j] = value
                    lhs_all[row, stencil_size + j, i] = value
                radius = np.sqrt(scaled[i, 0] ** 2 + scaled[i, 1] ** 2)
                first_over_radius = phs_degree * (radius + eps) ** (phs_degree - 2)
                second = phs_degree * (phs_degree - 1) * (radius + eps) ** (phs_degree - 2)
                rhs_all[row, i, 0] = (second + first_over_radius) / (width * width)
                rhs_all[row, i, 1] = -scaled[i, 0] * first_over_radius / width
                rhs_all[row, i, 2] = -scaled[i, 1] * first_over_radius / width
            derivatives = _legendre_tensor_evaluate_numba(
                scaled[:1], index_set, derivative_orders
            )[0]
            for j in range(polynomial_count):
                rhs_all[row, stencil_size + j, 0] = (
                    derivatives[j, 0] + derivatives[j, 1]
                ) / (width * width)
                rhs_all[row, stencil_size + j, 1] = derivatives[j, 2] / width
                rhs_all[row, stencil_size + j, 2] = derivatives[j, 3] / width
        return lhs_all, rhs_all, frames


    @njit(cache=True, fastmath=True, parallel=True)
    def _surface_interpolation_systems_numba(
        source_points: np.ndarray,
        target_points: np.ndarray,
        target_normals: np.ndarray,
        neighbors: np.ndarray,
        index_set: np.ndarray,
        phs_degree: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        row_count = target_points.shape[0]
        stencil_size = neighbors.shape[1]
        polynomial_count = index_set.shape[0]
        system_size = stencil_size + polynomial_count
        lhs_all = np.zeros((row_count, system_size, system_size), dtype=np.float64)
        rhs_all = np.zeros((row_count, system_size), dtype=np.float64)
        zero_derivative = np.zeros((1, 2), dtype=np.int64)
        query = np.zeros((1, 2), dtype=np.float64)
        query_polynomial = _legendre_tensor_evaluate_numba(
            query, index_set, zero_derivative
        )[0, :, 0]
        eps = np.finfo(np.float64).eps
        for row in prange(row_count):
            frame = _surface_tangent_frame_numba(target_normals[row])
            scaled = np.empty((stencil_size, 2), dtype=np.float64)
            width = 0.0
            target = target_points[row]
            for i in range(stencil_size):
                point = source_points[neighbors[row, i]] - target
                for axis in range(2):
                    value = point[0] * frame[0, axis] + point[1] * frame[1, axis] + point[2] * frame[2, axis]
                    scaled[i, axis] = value
                    width = max(width, abs(value))
            width = max(width, eps)
            scaled /= width
            polynomial = _legendre_tensor_evaluate_numba(
                scaled, index_set, zero_derivative
            )[:, :, 0]
            for i in range(stencil_size):
                for j in range(stencil_size):
                    dx = scaled[i, 0] - scaled[j, 0]
                    dy = scaled[i, 1] - scaled[j, 1]
                    radius = np.sqrt(dx * dx + dy * dy)
                    lhs_all[row, i, j] = (radius + eps) ** phs_degree
                for j in range(polynomial_count):
                    value = polynomial[i, j]
                    lhs_all[row, i, stencil_size + j] = value
                    lhs_all[row, stencil_size + j, i] = value
                radius = np.sqrt(scaled[i, 0] ** 2 + scaled[i, 1] ** 2)
                rhs_all[row, i] = (radius + eps) ** phs_degree
            rhs_all[row, stencil_size:] = query_polynomial
        return lhs_all, rhs_all


    @njit(cache=True, fastmath=True, parallel=True)
    def _surface_apply_scalar_numba(
        weights: np.ndarray, neighbors: np.ndarray, values: np.ndarray
    ) -> np.ndarray:
        out = np.empty(weights.shape[0], dtype=values.dtype)
        for row in prange(weights.shape[0]):
            total = 0.0
            for column in range(weights.shape[1]):
                total += weights[row, column] * values[neighbors[row, column]]
            out[row] = total
        return out


    @njit(cache=True, fastmath=True, parallel=True)
    def _surface_apply_matrix_numba(
        weights: np.ndarray, neighbors: np.ndarray, values: np.ndarray
    ) -> np.ndarray:
        out = np.empty((weights.shape[0], values.shape[1]), dtype=values.dtype)
        for row in prange(weights.shape[0]):
            for component in range(values.shape[1]):
                total = 0.0
                for column in range(weights.shape[1]):
                    total += weights[row, column] * values[neighbors[row, column], component]
                out[row, component] = total
        return out


    @njit(cache=True, fastmath=True, parallel=True)
    def _sbf_basis_derivatives_numba(
        queries: np.ndarray, centers: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        basis = np.empty((queries.shape[0], centers.shape[0]), dtype=np.float64)
        derivative_one = np.empty_like(basis)
        derivative_two = np.empty_like(basis)
        eps = np.finfo(np.float64).eps
        for row in prange(queries.shape[0]):
            point = queries[row]
            reference = np.array([1.0, 0.0, 0.0]) if abs(point[2]) > 0.9 else np.array([0.0, 0.0, 1.0])
            tangent_one = np.array([
                reference[1] * point[2] - reference[2] * point[1],
                reference[2] * point[0] - reference[0] * point[2],
                reference[0] * point[1] - reference[1] * point[0],
            ])
            tangent_one /= max(np.sqrt(np.sum(tangent_one * tangent_one)), eps)
            tangent_two = np.array([
                point[1] * tangent_one[2] - point[2] * tangent_one[1],
                point[2] * tangent_one[0] - point[0] * tangent_one[2],
                point[0] * tangent_one[1] - point[1] * tangent_one[0],
            ])
            tangent_two /= max(np.sqrt(np.sum(tangent_two * tangent_two)), eps)
            for column in range(centers.shape[0]):
                difference = point - centers[column]
                radius = np.sqrt(np.sum(difference * difference))
                basis[row, column] = radius**8 * np.log(radius + eps)
                radial = radius**6 * (8.0 * np.log(radius + eps) + 1.0)
                derivative_one[row, column] = radial * np.sum(difference * tangent_one)
                derivative_two[row, column] = radial * np.sum(difference * tangent_two)
        return basis, derivative_one, derivative_two


    @njit(cache=True, fastmath=True, parallel=True)
    def _sbf_basis_numba(
        queries: np.ndarray, centers: np.ndarray
    ) -> np.ndarray:
        basis = np.empty((queries.shape[0], centers.shape[0]), dtype=np.float64)
        eps = np.finfo(np.float64).eps
        for row in prange(queries.shape[0]):
            for column in range(centers.shape[0]):
                difference = queries[row] - centers[column]
                radius = np.sqrt(np.sum(difference * difference))
                basis[row, column] = radius**8 * np.log(radius + eps)
        return basis


    @njit(cache=True, fastmath=True)
    def _farthest_point_subset_numba(sites: np.ndarray, count: int) -> np.ndarray:
        selected = np.empty(count, dtype=np.int64)
        center = np.zeros(sites.shape[1], dtype=np.float64)
        for row in range(sites.shape[0]):
            center += sites[row]
        center /= sites.shape[0]
        first = 0
        first_distance = -1.0
        for row in range(sites.shape[0]):
            value = np.sum((sites[row] - center) ** 2)
            if value > first_distance:
                first = row
                first_distance = value
        selected[0] = first
        distance = np.empty(sites.shape[0], dtype=np.float64)
        for row in range(sites.shape[0]):
            distance[row] = np.sum((sites[row] - sites[first]) ** 2)
        distance[first] = -np.inf
        for index in range(1, count):
            next_index = np.argmax(distance)
            selected[index] = next_index
            for row in range(sites.shape[0]):
                candidate = np.sum((sites[row] - sites[next_index]) ** 2)
                distance[row] = min(distance[row], candidate)
            distance[next_index] = -np.inf
        return np.sort(selected)

else:
    _surface_local_systems_numba = None
    _surface_interpolation_systems_numba = None
    _surface_apply_scalar_numba = None
    _surface_apply_matrix_numba = None
    _sbf_basis_derivatives_numba = None
    _sbf_basis_numba = None
    _farthest_point_subset_numba = None


def surface_local_systems(
    points: np.ndarray,
    normals: np.ndarray,
    neighbors: np.ndarray,
    index_set: np.ndarray,
    phs_degree: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not NUMBA_AVAILABLE:
        return None
    return _surface_local_systems_numba(
        np.asarray(points, dtype=float),
        np.asarray(normals, dtype=float),
        np.asarray(neighbors, dtype=np.int64),
        np.asarray(index_set, dtype=np.int64),
        int(phs_degree),
    )


def surface_interpolation_systems(
    source_points: np.ndarray,
    target_points: np.ndarray,
    target_normals: np.ndarray,
    neighbors: np.ndarray,
    index_set: np.ndarray,
    phs_degree: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not NUMBA_AVAILABLE:
        return None
    return _surface_interpolation_systems_numba(
        np.asarray(source_points, dtype=float),
        np.asarray(target_points, dtype=float),
        np.asarray(target_normals, dtype=float),
        np.asarray(neighbors, dtype=np.int64),
        np.asarray(index_set, dtype=np.int64),
        int(phs_degree),
    )


def surface_apply(weights: np.ndarray, neighbors: np.ndarray, values: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    neighbors = np.asarray(neighbors, dtype=np.int64)
    values = np.asarray(values)
    if NUMBA_AVAILABLE and values.ndim == 1:
        return _surface_apply_scalar_numba(weights, neighbors, values)
    if NUMBA_AVAILABLE and values.ndim == 2:
        return _surface_apply_matrix_numba(weights, neighbors, values)
    return np.einsum("nk,nk...->n...", weights, values[neighbors])


def sbf_basis_derivatives(
    queries: np.ndarray, centers: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not NUMBA_AVAILABLE:
        return None
    return _sbf_basis_derivatives_numba(
        np.asarray(queries, dtype=float), np.asarray(centers, dtype=float)
    )


def sbf_basis(queries: np.ndarray, centers: np.ndarray) -> np.ndarray | None:
    if not NUMBA_AVAILABLE:
        return None
    return _sbf_basis_numba(
        np.asarray(queries, dtype=float), np.asarray(centers, dtype=float)
    )


def farthest_point_subset_numba(sites: np.ndarray, count: int) -> np.ndarray | None:
    if not NUMBA_AVAILABLE:
        return None
    return _farthest_point_subset_numba(np.asarray(sites, dtype=float), int(count))
