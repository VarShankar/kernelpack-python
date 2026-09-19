from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from kernelpack.manifold import (
    SurfaceOperators,
    apply_surface_operator,
    hyperviscosity_divergence_correction,
    surface_divergence,
)


VelocityCallback = Callable[[float, np.ndarray], np.ndarray]


@dataclass
class MovingSurfaceHistory:
    concentration: np.ndarray
    points: np.ndarray


@dataclass
class MovingSurfaceStepInfo:
    linear_info: int
    relative_residual: float
    mass_before_projection: float
    mass_after_projection: float
    mass_shift: float


def initialize_moving_surface_history(
    points: np.ndarray, concentration: np.ndarray
) -> MovingSurfaceHistory:
    points = np.asarray(points, dtype=float)
    concentration = np.asarray(concentration, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if concentration.shape != (points.shape[0],):
        raise ValueError("concentration must contain one scalar per surface node")
    return MovingSurfaceHistory(
        concentration=np.broadcast_to(concentration, (3, concentration.size)).copy(),
        points=np.broadcast_to(points, (3, *points.shape)).copy(),
    )


def rk3_material_step(
    time: float,
    points: np.ndarray,
    dt: float,
    velocity: VelocityCallback,
) -> np.ndarray:
    first = velocity(time, points)
    second = velocity(time + 0.5 * dt, points + 0.5 * dt * first)
    third = velocity(time + 0.75 * dt, points + 0.75 * dt * second)
    return points + dt * (2.0 * first + 3.0 * second + 4.0 * third) / 9.0


def semi_lagrangian_backfill_points(
    arrival_points: np.ndarray,
    arrival_time: float,
    dt: float,
    velocity: VelocityCallback,
    *,
    history_levels: int = 2,
) -> np.ndarray:
    points = np.asarray(arrival_points, dtype=float)
    departures = []
    for level in range(history_levels):
        points = rk3_material_step(arrival_time - level * dt, points, -dt, velocity)
        departures.append(points)
    return np.asarray(departures)


def _bdf_data(order: int) -> tuple[np.ndarray, np.ndarray, float]:
    if order == 1:
        return np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 1.0
    if order == 2:
        return np.array([4.0 / 3.0, -1.0 / 3.0, 0.0]), np.array([2.0, -1.0, 0.0]), 2.0 / 3.0
    if order == 3:
        return np.array([18.0 / 11.0, -9.0 / 11.0, 2.0 / 11.0]), np.array([3.0, -3.0, 1.0]), 6.0 / 11.0
    raise ValueError("BDF order must be 1, 2, or 3")


def bdf_material_velocity(
    new_points: np.ndarray,
    point_history: np.ndarray,
    dt: float,
    *,
    order: int,
) -> np.ndarray:
    if order == 1:
        return (new_points - point_history[0]) / dt
    if order == 2:
        return (3.0 * new_points - 4.0 * point_history[0] + point_history[1]) / (2.0 * dt)
    if order == 3:
        return (
            11.0 * new_points
            - 18.0 * point_history[0]
            + 9.0 * point_history[1]
            - 2.0 * point_history[2]
        ) / (6.0 * dt)
    raise ValueError("BDF order must be 1, 2, or 3")


def _laplacian_diagonal(operators: SurfaceOperators) -> np.ndarray:
    rows = np.arange(operators.neighbors.shape[0])[:, None]
    return np.sum(
        np.where(operators.neighbors == rows, operators.laplacian_weights, 0.0), axis=1
    )


def moving_surface_adr_step(
    history: MovingSurfaceHistory,
    new_points: np.ndarray,
    operators: SurfaceOperators,
    forcing: np.ndarray,
    quadrature_weights: np.ndarray,
    target_mass: float,
    dt: float,
    diffusivity: float,
    hyperviscosity_gamma: np.ndarray,
    linear_tolerance: float,
    *,
    order: int,
    hyperviscosity_power: int,
    reaction_history: np.ndarray | None = None,
    gmres_restart: int = 20,
    gmres_max_iterations: int = 200,
    project_mass: bool = True,
) -> tuple[MovingSurfaceHistory, MovingSurfaceStepInfo]:
    history_coefficients, extrapolation_coefficients, implicit_scale = _bdf_data(order)
    previous_combination = history_coefficients @ history.concentration
    extrapolated = extrapolation_coefficients @ history.concentration
    extrapolated_reaction = (
        np.zeros_like(extrapolated)
        if reaction_history is None
        else extrapolation_coefficients @ np.asarray(reaction_history)
    )
    material_velocity = bdf_material_velocity(new_points, history.points, dt, order=order)
    dilation = surface_divergence(operators, material_velocity)
    if hyperviscosity_power > 0:
        dilation += hyperviscosity_divergence_correction(
            operators,
            material_velocity,
            hyperviscosity_gamma,
            power=hyperviscosity_power,
        )
    rhs = (
        previous_combination
        + implicit_scale * dt * (np.asarray(forcing) + extrapolated_reaction)
        - implicit_scale * dt * extrapolated * dilation
    )

    def matvec(values: np.ndarray) -> np.ndarray:
        laplacian = apply_surface_operator(
            operators.laplacian_weights, operators.neighbors, values
        )
        return values - implicit_scale * dt * diffusivity * laplacian

    node_count = rhs.size
    system = LinearOperator((node_count, node_count), matvec=matvec, dtype=float)
    diagonal = 1.0 - implicit_scale * dt * diffusivity * _laplacian_diagonal(operators)
    inverse_diagonal = 1.0 / np.where(
        np.abs(diagonal) > np.finfo(float).eps, diagonal, 1.0
    )
    preconditioner = LinearOperator(
        (node_count, node_count), matvec=lambda values: inverse_diagonal * values, dtype=float
    )
    solution, linear_info = gmres(
        system,
        rhs,
        x0=extrapolated,
        rtol=linear_tolerance,
        atol=0.0,
        restart=gmres_restart,
        maxiter=gmres_max_iterations,
        M=preconditioner,
    )
    relative_residual = np.linalg.norm(matvec(solution) - rhs) / max(
        np.linalg.norm(rhs), np.finfo(float).eps
    )
    quadrature_weights = np.asarray(quadrature_weights, dtype=float)
    mass_before = float(quadrature_weights @ solution)
    mass_shift = (
        float((target_mass - mass_before) / np.sum(quadrature_weights))
        if project_mass
        else 0.0
    )
    solution = solution + mass_shift
    mass_after = float(quadrature_weights @ solution)
    updated = MovingSurfaceHistory(
        concentration=np.concatenate((solution[None, :], history.concentration[:2]), axis=0),
        points=np.concatenate((np.asarray(new_points)[None, :, :], history.points[:2]), axis=0),
    )
    return updated, MovingSurfaceStepInfo(
        int(linear_info),
        float(relative_residual),
        mass_before,
        mass_after,
        mass_shift,
    )
