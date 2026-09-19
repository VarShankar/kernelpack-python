from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg
from scipy.spatial import cKDTree

from kernelpack._numba import surface_apply, surface_local_systems
from kernelpack.poly import PolynomialBasis


@dataclass
class SurfaceOperators:
    neighbors: np.ndarray
    laplacian_weights: np.ndarray
    gradient_weights: np.ndarray


@dataclass
class SurfaceOperatorCache:
    neighbors: np.ndarray
    lhs: np.ndarray
    lu: np.ndarray
    pivots: np.ndarray


@dataclass
class SurfaceUpdateInfo:
    relative_residual: np.ndarray
    defect_iterations: np.ndarray
    refactored: np.ndarray


def surface_polynomial_degree(xi: int, theta: int = 2) -> int:
    if xi < 1:
        raise ValueError("xi must be positive")
    return int(xi + theta - 1)


def surface_phs_degree(ell: int) -> int:
    degree = ell if ell % 2 else ell - 1
    return min(max(int(degree), 5), 11)


def surface_stencil_size(ell: int) -> int:
    polynomial_count = (ell + 1) * (ell + 2) // 2
    return 2 * polynomial_count + 1


def build_surface_stencil_graph(points: np.ndarray, stencil_size: int) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("moving-surface points must have shape (N, 3)")
    if stencil_size < 1 or stencil_size > points.shape[0]:
        raise ValueError("invalid surface stencil size")
    return np.asarray(cKDTree(points).query(points, k=stencil_size)[1], dtype=int)


def build_cross_surface_stencil_graph(
    query_points: np.ndarray,
    source_points: np.ndarray,
    stencil_size: int,
) -> np.ndarray:
    query_points = np.asarray(query_points, dtype=float)
    source_points = np.asarray(source_points, dtype=float)
    if query_points.ndim != 2 or source_points.ndim != 2:
        raise ValueError("query_points and source_points must be matrices")
    if query_points.shape[1] != 3 or source_points.shape[1] != 3:
        raise ValueError("surface transfer requires three-dimensional points")
    return np.asarray(cKDTree(source_points).query(query_points, k=stencil_size)[1], dtype=int)


def estimate_pca_normals(
    points: np.ndarray,
    *,
    previous_normals: np.ndarray | None = None,
    stencil_size: int = 20,
) -> np.ndarray:
    """Estimate consistently oriented point-cloud normals by local PCA."""

    points = np.asarray(points, dtype=float)
    neighbors = build_surface_stencil_graph(points, min(int(stencil_size), points.shape[0]))
    clouds = points[neighbors]
    clouds = clouds - np.mean(clouds, axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", clouds, clouds)
    _, eigenvectors = np.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0]
    reference = points - np.mean(points, axis=0) if previous_normals is None else np.asarray(previous_normals)
    normals[np.sum(normals * reference, axis=1) < 0.0] *= -1.0
    return normals / np.maximum(
        np.linalg.norm(normals, axis=1, keepdims=True), np.finfo(float).eps
    )


def _normalize(vector: np.ndarray) -> np.ndarray:
    return vector / max(np.linalg.norm(vector), np.finfo(float).eps)


def _tangent_frame(normal: np.ndarray) -> np.ndarray:
    normal = _normalize(normal)
    dominant = int(np.argmax(np.abs(normal)))
    first_references = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    second_references = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    tangent_one = _normalize(first_references[dominant] - np.dot(normal, first_references[dominant]) * normal)
    second = second_references[dominant]
    tangent_two = _normalize(second - np.dot(normal, second) * normal - np.dot(tangent_one, second) * tangent_one)
    return np.column_stack((tangent_one, tangent_two))


def _local_tangent_system(
    stencil_points: np.ndarray,
    center_normal: np.ndarray,
    basis: PolynomialBasis,
    phs_degree: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = _tangent_frame(center_normal)
    local_points = (stencil_points - stencil_points[0]) @ frame
    width = max(float(np.max(np.abs(local_points))), np.finfo(float).eps)
    scaled_points = local_points / width
    pairwise_distance = np.linalg.norm(
        scaled_points[:, None, :] - scaled_points[None, :, :], axis=2
    )
    epsilon = np.finfo(float).eps
    polynomial = basis.evaluate(scaled_points, assume_normalized=True)
    polynomial_count = basis.index_set.shape[0]
    lhs = np.block(
        [
            [(pairwise_distance + epsilon) ** phs_degree, polynomial],
            [polynomial.T, np.zeros((polynomial_count, polynomial_count))],
        ]
    )

    center_distance = pairwise_distance[:, 0]
    first_over_radius = phs_degree * (center_distance + epsilon) ** (phs_degree - 2)
    second = phs_degree * (phs_degree - 1) * (center_distance + epsilon) ** (phs_degree - 2)
    rbf_laplacian = (second + first_over_radius) / width**2
    rbf_gradient = -scaled_points * first_over_radius[:, None] / width
    derivative_orders = np.array([[2, 0], [0, 2], [1, 0], [0, 1]], dtype=int)
    polynomial_derivatives = basis.evaluate(
        scaled_points[:1], derivative_orders, assume_normalized=True
    )[0]
    polynomial_laplacian = (
        polynomial_derivatives[:, 0] + polynomial_derivatives[:, 1]
    ) / width**2
    polynomial_gradient = polynomial_derivatives[:, 2:4] / width
    rhs = np.concatenate(
        (
            np.column_stack((rbf_laplacian, rbf_gradient)),
            np.column_stack((polynomial_laplacian, polynomial_gradient)),
        ),
        axis=0,
    )
    return lhs, rhs, frame


def _operators_from_solution(
    neighbors: np.ndarray,
    solution: np.ndarray,
    frames: np.ndarray,
) -> SurfaceOperators:
    stencil_size = neighbors.shape[1]
    laplacian = solution[:, :stencil_size, 0]
    local_gradient = solution[:, :stencil_size, 1:3]
    ambient_gradient = np.einsum("nka,nda->nkd", local_gradient, frames)
    return SurfaceOperators(neighbors, laplacian, ambient_gradient)


def assemble_tangent_plane_operators(
    points: np.ndarray,
    normals: np.ndarray,
    neighbors: np.ndarray,
    *,
    xi: int,
    theta: int = 2,
) -> tuple[SurfaceOperators, SurfaceOperatorCache]:
    points = np.asarray(points, dtype=float)
    normals = np.asarray(normals, dtype=float)
    neighbors = np.asarray(neighbors, dtype=int)
    ell = surface_polynomial_degree(xi, theta)
    basis = PolynomialBasis.from_total_degree(2, ell, family="legendre")
    phs_degree = surface_phs_degree(ell)
    system_size = neighbors.shape[1] + basis.index_set.shape[0]
    systems = surface_local_systems(
        points, normals, neighbors, basis.index_set, phs_degree
    )
    if systems is None:
        lhs_all = np.empty((points.shape[0], system_size, system_size))
        rhs_all = np.empty((points.shape[0], system_size, 3))
        frames = np.empty((points.shape[0], 3, 2))
        for row in range(points.shape[0]):
            lhs_all[row], rhs_all[row], frames[row] = _local_tangent_system(
                points[neighbors[row]], normals[row], basis, phs_degree
            )
    else:
        lhs_all, rhs_all, frames = systems
    lu_all, pivots_all = linalg.lu_factor(lhs_all)
    solution = linalg.lu_solve((lu_all, pivots_all), rhs_all)
    operators = _operators_from_solution(neighbors, solution, frames)
    return operators, SurfaceOperatorCache(neighbors.copy(), lhs_all, lu_all, pivots_all)


def update_tangent_plane_operators(
    points: np.ndarray,
    normals: np.ndarray,
    neighbors: np.ndarray,
    cache: SurfaceOperatorCache,
    *,
    xi: int,
    theta: int = 2,
    tolerance: float = 1.0e-10,
    max_defect_iterations: int = 4,
) -> tuple[SurfaceOperators, SurfaceOperatorCache, SurfaceUpdateInfo]:
    points = np.asarray(points, dtype=float)
    normals = np.asarray(normals, dtype=float)
    neighbors = np.asarray(neighbors, dtype=int)
    if not np.array_equal(neighbors, cache.neighbors):
        operators, updated = assemble_tangent_plane_operators(
            points, normals, neighbors, xi=xi, theta=theta
        )
        info = SurfaceUpdateInfo(
            np.zeros(points.shape[0]), np.zeros(points.shape[0], dtype=int), np.ones(points.shape[0], dtype=bool)
        )
        return operators, updated, info

    ell = surface_polynomial_degree(xi, theta)
    basis = PolynomialBasis.from_total_degree(2, ell, family="legendre")
    phs_degree = surface_phs_degree(ell)
    systems = surface_local_systems(
        points, normals, neighbors, basis.index_set, phs_degree
    )
    if systems is None:
        lhs_current = np.empty_like(cache.lhs)
        rhs = np.empty((points.shape[0], cache.lhs.shape[1], 3))
        frames = np.empty((points.shape[0], 3, 2))
        for row in range(points.shape[0]):
            lhs_current[row], rhs[row], frames[row] = _local_tangent_system(
                points[neighbors[row]], normals[row], basis, phs_degree
            )
    else:
        lhs_current, rhs, frames = systems
    candidate = linalg.lu_solve((cache.lu, cache.pivots), rhs)
    denominator = np.maximum(
        np.linalg.norm(rhs, axis=(1, 2)), np.finfo(float).eps
    )
    iterations = np.zeros(points.shape[0], dtype=int)
    for _ in range(max_defect_iterations):
        residual = rhs - np.einsum("nij,njk->nik", lhs_current, candidate)
        active = np.linalg.norm(residual, axis=(1, 2)) / denominator > tolerance
        if not np.any(active):
            break
        correction = linalg.lu_solve((cache.lu, cache.pivots), residual)
        candidate += np.where(active[:, None, None], correction, 0.0)
        iterations += active.astype(int)
    residual = rhs - np.einsum("nij,njk->nik", lhs_current, candidate)
    residuals = np.linalg.norm(residual, axis=(1, 2)) / denominator
    refactored = (~np.all(np.isfinite(candidate), axis=(1, 2))) | (
        residuals > tolerance
    )
    lhs_all = cache.lhs.copy()
    lu_all = cache.lu.copy()
    pivots_all = cache.pivots.copy()
    if np.any(refactored):
        bad_lu, bad_pivots = linalg.lu_factor(lhs_current[refactored])
        candidate[refactored] = linalg.lu_solve(
            (bad_lu, bad_pivots), rhs[refactored]
        )
        lhs_all[refactored] = lhs_current[refactored]
        lu_all[refactored] = bad_lu
        pivots_all[refactored] = bad_pivots
        corrected_residual = rhs[refactored] - np.einsum(
            "nij,njk->nik", lhs_current[refactored], candidate[refactored]
        )
        residuals[refactored] = np.linalg.norm(
            corrected_residual, axis=(1, 2)
        ) / denominator[refactored]
    operators = _operators_from_solution(neighbors, candidate, frames)
    updated = SurfaceOperatorCache(neighbors.copy(), lhs_all, lu_all, pivots_all)
    return operators, updated, SurfaceUpdateInfo(residuals, iterations, refactored)


def apply_surface_operator(weights: np.ndarray, neighbors: np.ndarray, values: np.ndarray) -> np.ndarray:
    return surface_apply(weights, neighbors, values)


def surface_gradient(operators: SurfaceOperators, values: np.ndarray) -> np.ndarray:
    return np.einsum("nkd,nk...->nd...", operators.gradient_weights, np.asarray(values)[operators.neighbors])


def surface_divergence(operators: SurfaceOperators, vector_field: np.ndarray) -> np.ndarray:
    differentiated = np.einsum(
        "nkd,nkc->ndc", operators.gradient_weights, np.asarray(vector_field)[operators.neighbors]
    )
    return np.trace(differentiated, axis1=1, axis2=2)


def apply_surface_laplacian_power(
    operators: SurfaceOperators, values: np.ndarray, *, power: int
) -> np.ndarray:
    result = np.asarray(values)
    for _ in range(power):
        result = apply_surface_operator(operators.laplacian_weights, operators.neighbors, result)
    return result


def hyperviscosity_divergence_correction(
    operators: SurfaceOperators,
    velocity: np.ndarray,
    gamma: np.ndarray,
    *,
    power: int,
) -> np.ndarray:
    return apply_surface_laplacian_power(operators, velocity, power=power) @ np.asarray(gamma)


def surface_point_quality(points: np.ndarray, two_nearest_neighbors: np.ndarray) -> float:
    distances = np.linalg.norm(
        np.asarray(points)[two_nearest_neighbors] - np.asarray(points)[:, None, :], axis=2
    )
    nearest_nonself = np.max(distances, axis=1)
    return float(np.max(nearest_nonself) / max(np.min(nearest_nonself), np.finfo(float).eps))


def predict_surface_quality_crossing(
    previous_quality: float,
    current_quality: float,
    threshold: float,
    lookahead_steps: float,
) -> tuple[bool, float]:
    predicted = current_quality + max(0.0, lookahead_steps) * max(0.0, current_quality - previous_quality)
    return predicted >= threshold, predicted
