from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs

from .core import SurfaceOperators, apply_surface_laplacian_power, apply_surface_operator


@dataclass
class HyperviscosityCalibration:
    gamma: np.ndarray
    power: int
    tau: np.ndarray
    growth_exponents: np.ndarray
    eta_mean: float
    raw_components: np.ndarray
    curvature_components: np.ndarray


def estimate_surface_gradient_spectrum(
    operators: SurfaceOperators,
    *,
    tolerance: float = 1.0e-2,
    maximum_iterations: int | None = None,
) -> np.ndarray:
    node_count = operators.neighbors.shape[0]
    estimates = np.empty(3)
    for component in range(3):
        weights = operators.gradient_weights[:, :, component]
        operator = LinearOperator(
            (node_count, node_count),
            matvec=lambda values, w=weights: apply_surface_operator(
                w, operators.neighbors, values
            ),
            dtype=float,
        )
        try:
            value = eigs(
                operator,
                k=1,
                which="LR",
                tol=tolerance,
                maxiter=maximum_iterations,
                return_eigenvectors=False,
            )[0]
            estimates[component] = max(0.0, float(np.real(value)))
        except Exception:
            estimates[component] = 0.0
    return estimates


def estimate_surface_gradient_growth(
    operators: SurfaceOperators,
    points: np.ndarray,
    normals: np.ndarray,
    h: float,
    tau: np.ndarray,
) -> np.ndarray:
    wave_number = 2.0 / h
    oscillation = np.exp(1j * wave_number * np.sum(points, axis=1))
    ambient = 1j * wave_number * oscillation[:, None] * np.ones((1, 3))
    projected = ambient - normals * np.sum(normals * ambient, axis=1)[:, None]
    numerical = np.column_stack(
        [
            apply_surface_operator(
                operators.gradient_weights[:, :, component],
                operators.neighbors,
                oscillation,
            )
            for component in range(3)
        ]
    )
    error = np.linalg.norm(numerical - projected, axis=0)
    safe_tau = np.maximum(np.abs(tau), np.finfo(float).eps)
    return (
        np.log(np.maximum(error, np.finfo(float).eps))
        - np.log(safe_tau)
        - np.log(np.linalg.norm(oscillation))
    ) / np.log(wave_number)


def surface_curvature_components(
    operators: SurfaceOperators, points: np.ndarray
) -> np.ndarray:
    coordinates = apply_surface_operator(
        operators.laplacian_weights, operators.neighbors, points
    )
    return np.sqrt(np.mean(coordinates**2, axis=0))


def _fixed_power_calibration(
    operators: SurfaceOperators,
    points: np.ndarray,
    h: float,
    tau: np.ndarray,
    growth_exponents: np.ndarray,
    power: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    raw = tau * 2.0 ** (growth_exponents - 2 * power) * h ** (
        2 * power - growth_exponents
    )
    wave_number = 2.0 / h
    oscillation = np.exp(1j * wave_number * np.sum(points, axis=1))
    real_eigenvalue = (-1.0) ** power * 3.0**power * wave_number ** (2 * power)
    eta = apply_surface_laplacian_power(
        operators, oscillation, power=power
    ) / (real_eigenvalue * oscillation)
    eta_mean = float(np.mean(np.abs(np.real(eta))))
    prefactor = 3.0 ** (-power) * (-1.0) ** (1 - power)
    return prefactor * raw / eta_mean, eta_mean, raw


def calibrate_surface_hyperviscosity(
    operators: SurfaceOperators,
    points: np.ndarray,
    normals: np.ndarray,
    h: float,
    *,
    target_order: int,
    power: int | None = None,
    minimum_power: int = 1,
    spectrum_tolerance: float = 1.0e-2,
) -> HyperviscosityCalibration:
    points = np.asarray(points, dtype=float)
    normals = np.asarray(normals, dtype=float)
    tau = estimate_surface_gradient_spectrum(
        operators, tolerance=spectrum_tolerance
    )
    growth = estimate_surface_gradient_growth(
        operators, points, normals, h, tau
    )
    selected = max(minimum_power, int(np.ceil((target_order + np.max(growth)) / 2.0))) if power is None else int(power)
    gamma, eta_mean, raw = _fixed_power_calibration(
        operators, points, h, tau, growth, selected
    )
    return HyperviscosityCalibration(
        gamma,
        selected,
        tau,
        growth,
        eta_mean,
        raw,
        surface_curvature_components(operators, points),
    )


def predict_surface_hyperviscosity(
    operators: SurfaceOperators,
    points: np.ndarray,
    h: float,
    calibration: HyperviscosityCalibration,
) -> tuple[HyperviscosityCalibration, float]:
    curvature = surface_curvature_components(operators, points)
    scale = calibration.curvature_components / np.maximum(
        curvature, np.finfo(float).eps
    )
    exponent = 2 * calibration.power - calibration.growth_exponents - 1.0
    raw = calibration.raw_components * scale**exponent
    gamma, eta_mean, _ = _fixed_power_calibration(
        operators,
        np.asarray(points),
        h,
        calibration.tau,
        calibration.growth_exponents,
        calibration.power,
    )
    prefactor = 3.0 ** (-calibration.power) * (-1.0) ** (1 - calibration.power)
    gamma = prefactor * raw / eta_mean
    drift = np.linalg.norm(gamma - calibration.gamma) / max(
        np.linalg.norm(calibration.gamma), np.finfo(float).eps
    )
    return HyperviscosityCalibration(
        gamma,
        calibration.power,
        calibration.tau,
        calibration.growth_exponents,
        eta_mean,
        raw,
        curvature,
    ), float(drift)
