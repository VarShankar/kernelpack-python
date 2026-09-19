"""Shared domain and manufactured fields for the public PDE examples."""

import numpy as np

from kernelpack.domain import DomainDescriptor


def build_square_domain(grid_size: int = 11) -> DomainDescriptor:
    coordinates = np.linspace(-1.0, 1.0, grid_size)
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel()))
    boundary_mask = np.any(np.isclose(np.abs(points), 1.0), axis=1)
    boundary = points[boundary_mask]
    interior = points[~boundary_mask]
    normals = np.where(np.isclose(np.abs(boundary), 1.0), np.sign(boundary), 0.0)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    ghosts = boundary + 0.25 * normals

    domain = DomainDescriptor()
    domain.set_nodes(interior, boundary, ghosts)
    domain.set_normals(normals)
    domain.set_sep_rad(float(coordinates[1] - coordinates[0]))
    domain.build_structs()
    return domain


def exact_solution(time: float, points: np.ndarray) -> np.ndarray:
    return np.exp(-time) * (points[:, 0] ** 2 + points[:, 1] ** 2)


def zero_neumann(points: np.ndarray) -> np.ndarray:
    return np.zeros(points.shape[0])


def unit_dirichlet(points: np.ndarray) -> np.ndarray:
    return np.ones(points.shape[0])
