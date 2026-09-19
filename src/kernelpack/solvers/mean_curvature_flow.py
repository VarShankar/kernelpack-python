from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from kernelpack.manifold import SurfaceOperators, apply_surface_operator


@dataclass
class MeanCurvatureFlowStepInfo:
    linear_info: int
    relative_residual: float
    velocity: np.ndarray


def mean_curvature_flow_step(
    points: np.ndarray,
    normals: np.ndarray,
    operators: SurfaceOperators,
    dt: float,
    *,
    mode: str = "semiimplicit",
    project_normal: bool = True,
    linear_tolerance: float = 1.0e-8,
    gmres_restart: int = 20,
    gmres_max_iterations: int = 200,
) -> tuple[np.ndarray, MeanCurvatureFlowStepInfo]:
    points = np.asarray(points, dtype=float)
    normals = np.asarray(normals, dtype=float)
    if mode == "explicit":
        velocity = apply_surface_operator(
            operators.laplacian_weights, operators.neighbors, points
        )
        if project_normal:
            velocity = np.sum(velocity * normals, axis=1, keepdims=True) * normals
        return points + dt * velocity, MeanCurvatureFlowStepInfo(0, 0.0, velocity)
    if mode != "semiimplicit":
        raise ValueError("mode must be 'explicit' or 'semiimplicit'")

    node_count = points.shape[0]

    def matvec(values: np.ndarray) -> np.ndarray:
        return values - dt * apply_surface_operator(
            operators.laplacian_weights, operators.neighbors, values
        )

    system = LinearOperator((node_count, node_count), matvec=matvec, dtype=float)
    solved = np.empty_like(points)
    linear_info = 0
    for component in range(3):
        solved[:, component], info = gmres(
            system,
            points[:, component],
            x0=points[:, component],
            rtol=linear_tolerance,
            atol=0.0,
            restart=gmres_restart,
            maxiter=gmres_max_iterations,
        )
        linear_info = max(linear_info, int(info))
    raw_step = solved - points
    if project_normal:
        raw_step = np.sum(raw_step * normals, axis=1, keepdims=True) * normals
    residual = np.column_stack(
        [matvec(solved[:, component]) - points[:, component] for component in range(3)]
    )
    relative_residual = np.linalg.norm(residual) / max(
        np.linalg.norm(points), np.finfo(float).eps
    )
    return points + raw_step, MeanCurvatureFlowStepInfo(
        linear_info, float(relative_residual), raw_step / dt
    )
