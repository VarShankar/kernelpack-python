from __future__ import annotations

import numpy as np

from kernelpack._numba import (
    farthest_point_subset_numba,
    surface_interpolation_systems,
)
from kernelpack.poly import PolynomialBasis

from .core import _tangent_frame, surface_phs_degree, surface_polynomial_degree


def local_tangent_interpolation_weights(
    source_points: np.ndarray,
    target_points: np.ndarray,
    target_normals: np.ndarray,
    cross_neighbors: np.ndarray,
    *,
    xi: int,
    theta: int = 2,
) -> np.ndarray:
    ell = surface_polynomial_degree(xi, theta)
    degree = surface_phs_degree(ell)
    basis = PolynomialBasis.from_total_degree(2, ell, family="legendre")
    source_points = np.asarray(source_points, dtype=float)
    target_points = np.asarray(target_points, dtype=float)
    target_normals = np.asarray(target_normals, dtype=float)
    cross_neighbors = np.asarray(cross_neighbors, dtype=int)
    systems = surface_interpolation_systems(
        source_points,
        target_points,
        target_normals,
        cross_neighbors,
        basis.index_set,
        degree,
    )
    if systems is not None:
        lhs, rhs = systems
        return np.linalg.solve(lhs, rhs[..., None])[:, : cross_neighbors.shape[1], 0]

    weights = np.empty(cross_neighbors.shape, dtype=float)
    epsilon = np.finfo(float).eps
    for row, indices in enumerate(cross_neighbors):
        target = target_points[row]
        frame = _tangent_frame(target_normals[row])
        local = (source_points[indices] - target) @ frame
        width = max(float(np.max(np.abs(local))), epsilon)
        scaled = local / width
        distance = np.linalg.norm(scaled[:, None, :] - scaled[None, :, :], axis=2)
        polynomial = basis.evaluate(scaled, assume_normalized=True)
        lhs = np.block(
            [
                [(distance + epsilon) ** degree, polynomial],
                [polynomial.T, np.zeros((polynomial.shape[1], polynomial.shape[1]))],
            ]
        )
        radius = np.linalg.norm(scaled, axis=1)
        query_polynomial = basis.evaluate(
            np.zeros((1, 2)), assume_normalized=True
        )[0]
        rhs = np.concatenate(((radius + epsilon) ** degree, query_polynomial))
        weights[row] = np.linalg.solve(lhs, rhs)[: indices.size]
    return weights


def apply_local_tangent_interpolant(
    weights: np.ndarray,
    cross_neighbors: np.ndarray,
    source_values: np.ndarray,
) -> np.ndarray:
    return np.einsum(
        "qk,qk...->q...", np.asarray(weights), np.asarray(source_values)[cross_neighbors]
    )


def farthest_point_subset(parameter_sites: np.ndarray, *, count: int) -> np.ndarray:
    sites = np.array(parameter_sites, dtype=float, copy=True)
    sites /= np.maximum(np.linalg.norm(sites, axis=1, keepdims=True), np.finfo(float).eps)
    selected_numba = farthest_point_subset_numba(sites, count)
    if selected_numba is not None:
        return selected_numba
    selected = np.empty(count, dtype=int)
    selected[0] = int(np.argmax(np.sum((sites - np.mean(sites, axis=0)) ** 2, axis=1)))
    distance = np.sum((sites - sites[selected[0]]) ** 2, axis=1)
    distance[selected[0]] = -np.inf
    for index in range(1, count):
        selected[index] = int(np.argmax(distance))
        candidate = np.sum((sites - sites[selected[index]]) ** 2, axis=1)
        distance = np.minimum(distance, candidate)
        distance[selected[index]] = -np.inf
    return np.sort(selected)


def spherical_sbf_transfer(
    source_parameter_sites: np.ndarray,
    source_values: np.ndarray,
    target_parameter_sites: np.ndarray,
    control_indices: np.ndarray,
    *,
    degree: int = 7,
) -> np.ndarray:
    normalize = lambda sites: sites / np.maximum(
        np.linalg.norm(sites, axis=1, keepdims=True), np.finfo(float).eps
    )
    source = normalize(np.asarray(source_parameter_sites, dtype=float))
    target = normalize(np.asarray(target_parameter_sites, dtype=float))
    centers = source[np.asarray(control_indices, dtype=int)]
    radius = np.sqrt(np.maximum(2.0 - 2.0 * centers @ centers.T, 0.0))
    kernel = (radius + np.finfo(float).eps) ** degree
    regularization = 1.0e-12 * max(1.0, float(np.max(np.abs(kernel))))
    coefficients = np.linalg.solve(
        kernel + regularization * np.eye(kernel.shape[0]),
        np.asarray(source_values)[control_indices],
    )
    target_radius = np.sqrt(np.maximum(2.0 - 2.0 * target @ centers.T, 0.0))
    return (target_radius + np.finfo(float).eps) ** degree @ coefficients
