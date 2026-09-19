from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg

from kernelpack._numba import sbf_basis, sbf_basis_derivatives


@dataclass
class SphericalSBFModel:
    parameter_sites: np.ndarray
    evaluation_sites: np.ndarray
    base_quadrature_weights: np.ndarray
    lu: np.ndarray
    pivots: np.ndarray


@dataclass
class SurfaceGeometry:
    points: np.ndarray
    normals: np.ndarray
    quadrature_weights: np.ndarray


@dataclass
class ToroidalRBFModel:
    parameter_sites: np.ndarray
    evaluation_sites: np.ndarray
    base_quadrature_weights: np.ndarray
    lu: np.ndarray
    pivots: np.ndarray
    degree: int


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(
        np.linalg.norm(values, axis=1, keepdims=True), np.finfo(float).eps
    )


def _sbf_parameter_matrix(parameter_sites: np.ndarray) -> np.ndarray:
    radius = np.sqrt(np.maximum(2.0 * (1.0 - parameter_sites @ parameter_sites.T), 0.0))
    return radius**8 * np.log(radius + np.finfo(float).eps)


def build_spherical_sbf_model(
    control_points: np.ndarray,
    evaluation_sites: np.ndarray | None = None,
    base_quadrature_weights: np.ndarray | None = None,
    parameter_sites: np.ndarray | None = None,
) -> SphericalSBFModel:
    control_points = np.asarray(control_points, dtype=float)
    if control_points.ndim != 2 or control_points.shape[1] != 3:
        raise ValueError("control_points must have shape (M, 3)")
    if parameter_sites is None:
        parameter_sites = _normalize_rows(control_points - np.mean(control_points, axis=0))
    else:
        parameter_sites = _normalize_rows(np.asarray(parameter_sites, dtype=float))
        if parameter_sites.shape != control_points.shape:
            raise ValueError("parameter_sites must match control_points")
    if evaluation_sites is None:
        evaluation_sites = parameter_sites
    evaluation_sites = _normalize_rows(np.asarray(evaluation_sites, dtype=float))
    if base_quadrature_weights is None:
        base_quadrature_weights = np.full(
            evaluation_sites.shape[0], 4.0 * np.pi / evaluation_sites.shape[0]
        )
    weights = np.asarray(base_quadrature_weights, dtype=float).reshape(-1)
    if weights.shape != (evaluation_sites.shape[0],):
        raise ValueError("base_quadrature_weights must match evaluation_sites")
    lu, pivots = linalg.lu_factor(_sbf_parameter_matrix(parameter_sites))
    return SphericalSBFModel(parameter_sites, evaluation_sites, weights, lu, pivots)


def _sphere_tangent_frame(point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.array([1.0, 0.0, 0.0]) if abs(point[2]) > 0.9 else np.array([0.0, 0.0, 1.0])
    tangent_one = np.cross(reference, point)
    tangent_one /= max(np.linalg.norm(tangent_one), np.finfo(float).eps)
    tangent_two = np.cross(point, tangent_one)
    tangent_two /= max(np.linalg.norm(tangent_two), np.finfo(float).eps)
    return tangent_one, tangent_two


def _sbf_basis_row(
    query: np.ndarray, parameter_sites: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    difference = query - parameter_sites
    radius = np.linalg.norm(difference, axis=1)
    epsilon = np.finfo(float).eps
    basis = radius**8 * np.log(radius + epsilon)
    radial_derivative = radius**6 * (8.0 * np.log(radius + epsilon) + 1.0)
    tangent_one, tangent_two = _sphere_tangent_frame(query)
    return (
        basis,
        radial_derivative * (difference @ tangent_one),
        radial_derivative * (difference @ tangent_two),
    )


def evaluate_spherical_sbf_geometry(
    model: SphericalSBFModel, control_points: np.ndarray
) -> SurfaceGeometry:
    coefficients = linalg.lu_solve((model.lu, model.pivots), np.asarray(control_points))
    basis_data = sbf_basis_derivatives(
        model.evaluation_sites, model.parameter_sites
    )
    if basis_data is None:
        rows = [
            _sbf_basis_row(query, model.parameter_sites)
            for query in model.evaluation_sites
        ]
        basis = np.asarray([row[0] for row in rows])
        derivative_one = np.asarray([row[1] for row in rows])
        derivative_two = np.asarray([row[2] for row in rows])
    else:
        basis, derivative_one, derivative_two = basis_data
    points = basis @ coefficients
    tangent_one = derivative_one @ coefficients
    tangent_two = derivative_two @ coefficients
    cross_product = np.cross(tangent_one, tangent_two)
    area_jacobian = np.linalg.norm(cross_product, axis=1)
    normals = _normalize_rows(cross_product)
    if np.mean(np.sum((points - np.mean(points, axis=0)) * normals, axis=1)) < 0.0:
        normals = -normals
    return SurfaceGeometry(
        points, normals, model.base_quadrature_weights * area_jacobian
    )


def evaluate_spherical_sbf_field(
    model: SphericalSBFModel,
    control_values: np.ndarray,
    query_sites: np.ndarray,
) -> np.ndarray:
    coefficients = linalg.lu_solve((model.lu, model.pivots), np.asarray(control_values))
    queries = _normalize_rows(query_sites)
    basis = sbf_basis(queries, model.parameter_sites)
    if basis is None:
        basis = np.asarray(
            [_sbf_basis_row(query, model.parameter_sites)[0] for query in queries]
        )
    return basis @ coefficients


def _periodic_parameter_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    difference = first[:, None, :] - second[None, :, :]
    return np.sqrt(np.maximum(np.sum(2.0 - 2.0 * np.cos(difference), axis=2), 0.0))


def build_toroidal_rbf_model(
    control_points: np.ndarray,
    parameter_sites: np.ndarray,
    evaluation_sites: np.ndarray | None = None,
    base_quadrature_weights: np.ndarray | None = None,
    *,
    degree: int = 7,
) -> ToroidalRBFModel:
    control_points = np.asarray(control_points, dtype=float)
    parameter_sites = np.asarray(parameter_sites, dtype=float)
    if parameter_sites.shape != (control_points.shape[0], 2):
        raise ValueError("parameter_sites must have shape (M, 2)")
    if evaluation_sites is None:
        evaluation_sites = parameter_sites
    evaluation_sites = np.asarray(evaluation_sites, dtype=float)
    if base_quadrature_weights is None:
        base_quadrature_weights = np.full(
            evaluation_sites.shape[0], (2.0 * np.pi) ** 2 / evaluation_sites.shape[0]
        )
    radius = _periodic_parameter_distance(parameter_sites, parameter_sites)
    kernel = (radius + np.finfo(float).eps) ** degree
    regularization = 1.0e-12 * max(1.0, float(np.max(np.abs(kernel))))
    lu, pivots = linalg.lu_factor(kernel + regularization * np.eye(kernel.shape[0]))
    return ToroidalRBFModel(
        parameter_sites,
        evaluation_sites,
        np.asarray(base_quadrature_weights),
        lu,
        pivots,
        int(degree),
    )


def evaluate_toroidal_rbf_geometry(
    model: ToroidalRBFModel, control_points: np.ndarray
) -> SurfaceGeometry:
    coefficients = linalg.lu_solve((model.lu, model.pivots), np.asarray(control_points))
    difference = model.evaluation_sites[:, None, :] - model.parameter_sites[None, :, :]
    radius = np.sqrt(np.maximum(np.sum(2.0 - 2.0 * np.cos(difference), axis=2), 0.0))
    epsilon = np.finfo(float).eps
    basis = (radius + epsilon) ** model.degree
    radial_factor = model.degree * (radius + epsilon) ** (model.degree - 2)
    tangent_theta = (radial_factor * np.sin(difference[:, :, 0])) @ coefficients
    tangent_phi = (radial_factor * np.sin(difference[:, :, 1])) @ coefficients
    points = basis @ coefficients
    cross_product = np.cross(tangent_theta, tangent_phi)
    area_jacobian = np.linalg.norm(cross_product, axis=1)
    normals = _normalize_rows(cross_product)
    if np.mean(np.sum((points - np.mean(points, axis=0)) * normals, axis=1)) < 0.0:
        normals = -normals
    return SurfaceGeometry(
        points, normals, model.base_quadrature_weights * area_jacobian
    )
