from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from typing import Callable

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
from scipy.spatial import cKDTree

from kernelpack.domain import DomainDescriptor
from kernelpack.geometry import EmbeddedSurface
from kernelpack.rbffd import OpProperties, RBFStencil, StencilProperties
from ._common import build_ilu_preconditioner, make_assembler


VelocityCallback = Callable[[float, np.ndarray], np.ndarray]


def rk3_step(time: float, points: np.ndarray, dt: float, velocity: VelocityCallback) -> np.ndarray:
    first = velocity(time, points)
    second = velocity(time + 0.5 * dt, points + 0.5 * dt * first)
    third = velocity(time + 0.75 * dt, points + 0.75 * dt * second)
    return points + dt * (2.0 * first + 3.0 * second + 4.0 * third) / 9.0


def trace_backward(points: np.ndarray, arrival_time: float, dt: float, velocity: VelocityCallback) -> np.ndarray:
    return rk3_step(arrival_time, points, -dt, velocity)


def _paper_stencil_properties(dim: int, ell: int, point_set: str, tree_mode: str) -> StencilProperties:
    polynomial_count = comb(dim + ell, dim)
    degree = ell if ell % 2 else ell - 1
    return StencilProperties(
        n=2 * polynomial_count + 1,
        dim=dim,
        ell=ell,
        spline_degree=max(5, min(11, degree)),
        npoly=polynomial_count,
        point_set=point_set,
        tree_mode=tree_mode,
    )


def _surface_seed_sites(surface: object) -> np.ndarray:
    sites = np.asarray(getattr(surface, "data_sites", np.zeros((0, 0))), dtype=float)
    if sites.size == 0:
        sites = np.asarray(surface.get_uniform_sample_sites(), dtype=float)
    if sites.ndim != 2 or sites.shape[0] == 0:
        raise ValueError("moving embedded surfaces require nonempty seed sites")
    return sites


def _rebuild_surface(seed_sites: np.ndarray, h: float) -> EmbeddedSurface:
    surface = EmbeddedSurface()
    surface.set_data_sites(seed_sites)
    surface.build_closed_geometric_model_ps(seed_sites.shape[1], h, seed_sites.shape[0])
    return surface


def _outside_inflated_boundary(
    points: np.ndarray,
    boundary: np.ndarray,
    outward_normals: np.ndarray,
    h: float,
) -> np.ndarray:
    inflated = boundary + 0.75 * h * outward_normals
    nearest = cKDTree(inflated).query(points, k=1)[1]
    displacement = points - inflated[nearest]
    return np.sum(displacement * outward_normals[nearest], axis=1) > 1.0e-12


def _moving_snapshot(background: DomainDescriptor, surfaces: list[EmbeddedSurface]) -> DomainDescriptor:
    h = background.get_sep_rad()
    interior = background.get_interior_nodes()
    keep = np.ones(interior.shape[0], dtype=bool)
    moving_interior = []
    moving_boundary = []
    moving_normals = []
    moving_ghosts = []
    for surface in surfaces:
        boundary = np.asarray(surface.get_uniform_sample_sites(), dtype=float)
        outward = np.asarray(surface.get_uniform_nrmls(), dtype=float)
        keep &= _outside_inflated_boundary(interior, boundary, outward, h)
        domain_normals = -outward
        moving_boundary.append(boundary)
        moving_normals.append(domain_normals)
        moving_interior.append(boundary - 0.5 * h * domain_normals)
        moving_ghosts.append(boundary + 0.5 * h * domain_normals)

    dim = background.get_dim()
    stack = lambda parts: np.vstack(parts) if parts else np.zeros((0, dim))
    snapshot = DomainDescriptor()
    snapshot.set_nodes(
        np.vstack((interior[keep], stack(moving_interior))),
        np.vstack((background.get_bdry_nodes(), stack(moving_boundary))),
        np.vstack((background.get_ghost_nodes(), stack(moving_ghosts))),
    )
    snapshot.set_normals(np.vstack((background.get_nrmls(), stack(moving_normals))))
    snapshot.set_sep_rad(h)
    snapshot.set_outer_level_set(background.get_outer_level_set())
    snapshot.set_boundary_level_sets(background.get_boundary_level_sets())
    snapshot.build_structs()
    return snapshot


class _LocalRBFInterpolator:
    def __init__(self, domain: DomainDescriptor, stencil: StencilProperties):
        self.points = np.asarray(domain.get_int_bdry_nodes(), dtype=float)
        self.stencil = stencil
        self.tree = cKDTree(self.points)
        self.neighbors = np.asarray(
            self.tree.query(self.points, k=min(stencil.n, self.points.shape[0]))[1], dtype=int
        )

    def evaluate(self, values: np.ndarray, query_points: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float).reshape(-1)
        query_points = np.atleast_2d(np.asarray(query_points, dtype=float))
        owners = np.asarray(self.tree.query(query_points, k=1)[1], dtype=int)
        result = np.empty(query_points.shape[0])
        op = OpProperties(decompose=False, store_weights=True, record_stencils=False)
        for owner in np.unique(owners):
            rows = np.flatnonzero(owners == owner)
            indices = self.neighbors[owner]
            local = RBFStencil()
            weights = local.compute_weights_at_points(
                self.points[indices], query_points[rows], self.stencil, op, "interp"
            )
            result[rows] = weights @ values[indices]
        return result


def _bdf_coefficients(order: int) -> tuple[np.ndarray, float]:
    if order == 1:
        return np.array([1.0]), 1.0
    if order == 2:
        return np.array([4.0 / 3.0, -1.0 / 3.0]), 2.0 / 3.0
    if order == 3:
        return np.array([18.0 / 11.0, -9.0 / 11.0, 2.0 / 11.0]), 6.0 / 11.0
    raise ValueError("BDF order must be 1, 2, or 3")


def _evaluate_field(specification: object, time: float, points: np.ndarray, *extra: float) -> np.ndarray:
    if callable(specification):
        try:
            values = specification(*extra, time, points)
        except TypeError:
            try:
                values = specification(time, points)
            except TypeError:
                values = specification(points)
    else:
        values = specification
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 1:
        values = np.full(points.shape[0], values[0])
    if values.size != points.shape[0]:
        raise ValueError("field callback returned the wrong number of values")
    return values


@dataclass
class MovingDomainADRSolver:
    gmres_tolerance: float | None = None
    gmres_restart: int = 20
    gmres_max_iterations: int = 400
    background_domain: DomainDescriptor | None = field(default=None, init=False)
    surface_seeds: list[np.ndarray] = field(default_factory=list, init=False)
    surfaces: list[EmbeddedSurface] = field(default_factory=list, init=False)
    domain: DomainDescriptor | None = field(default=None, init=False)
    xi: int = field(default=0, init=False)
    dt: float = field(default=np.nan, init=False)
    nu: float = field(default=np.nan, init=False)
    current_time: float = field(default=0.0, init=False)
    state_history: list[np.ndarray] = field(default_factory=list, init=False)
    interpolation_history: list[_LocalRBFInterpolator] = field(default_factory=list, init=False)
    last_linear_diagnostics: dict[str, object] = field(default_factory=dict, init=False)
    last_update_diagnostics: dict[str, int] = field(default_factory=dict, init=False)

    def init(
        self,
        background_domain: DomainDescriptor,
        embedded_surfaces: object | list[object] | tuple[object, ...],
        xi: int,
        dt: float,
        diffusivity: float,
        *,
        initial_time: float = 0.0,
    ) -> None:
        self.background_domain = background_domain
        self.xi = int(xi)
        self.dt = float(dt)
        self.nu = float(diffusivity)
        self.current_time = float(initial_time)
        models = list(embedded_surfaces) if isinstance(embedded_surfaces, (list, tuple)) else [embedded_surfaces]
        self.surface_seeds = [_surface_seed_sites(surface) for surface in models]
        h = background_domain.get_sep_rad()
        self.surfaces = [_rebuild_surface(sites, h) for sites in self.surface_seeds]
        self.domain = _moving_snapshot(background_domain, self.surfaces)
        self.state_history = []
        self.interpolation_history = []

    def set_initial_state(self, initial_state: object) -> None:
        points = self.domain.get_int_bdry_nodes()
        self.state_history = [_evaluate_field(initial_state, self.current_time, points)]
        stencil = _paper_stencil_properties(self.domain.get_dim(), self.xi + 1, "interior_boundary", "interior_boundary")
        self.interpolation_history = [_LocalRBFInterpolator(self.domain, stencil)]

    def step(
        self,
        next_time: float,
        velocity: VelocityCallback,
        forcing: object = 0.0,
        neu_coeff: object = 0.0,
        dir_coeff: object = 1.0,
        boundary_value: object = 0.0,
        reaction_coeff: object = 0.0,
    ) -> np.ndarray:
        if not self.state_history:
            raise RuntimeError("call set_initial_state before stepping")
        self.surface_seeds = [
            rk3_step(self.current_time, seeds, self.dt, velocity) for seeds in self.surface_seeds
        ]
        h = self.background_domain.get_sep_rad()
        self.surfaces = [_rebuild_surface(sites, h) for sites in self.surface_seeds]
        self.domain = _moving_snapshot(self.background_domain, self.surfaces)
        arrivals = self.domain.get_int_bdry_nodes()
        order = min(3, len(self.state_history))
        coefficients, scale = _bdf_coefficients(order)
        departure_values = []
        for history_index in range(order):
            departure = trace_backward(
                arrivals, next_time, (history_index + 1) * self.dt, velocity
            )
            departure_values.append(
                self.interpolation_history[history_index].evaluate(
                    self.state_history[history_index], departure
                )
            )
        rhs_physical = sum(
            coefficients[index] * departure_values[index] for index in range(order)
        )
        rhs_physical += scale * self.dt * _evaluate_field(
            forcing, next_time, arrivals, self.nu
        )

        lap_props = _paper_stencil_properties(self.domain.get_dim(), self.xi + 1, "interior_boundary", "all")
        bc_props = _paper_stencil_properties(self.domain.get_dim(), self.xi, "boundary", "all")
        lap = make_assembler("fdo", "rbf")
        lap.assemble_op(self.domain, "lap", lap_props, OpProperties(decompose=False, store_weights=True))
        boundary_points = self.domain.get_bdry_nodes()
        alpha = _evaluate_field(neu_coeff, next_time, boundary_points)
        beta = _evaluate_field(dir_coeff, next_time, boundary_points)
        bc = make_assembler("fd", "rbf")
        bc.assemble_op(
            self.domain,
            "bc",
            bc_props,
            OpProperties(decompose=False, store_weights=True),
            neu_coeff=alpha,
            dir_coeff=beta,
        )
        n = self.domain.get_num_int_bdry_nodes()
        nf = self.domain.get_num_total_nodes()
        identity = sparse.hstack((sparse.eye(n), sparse.csr_matrix((n, nf - n))), format="csr")
        reaction = _evaluate_field(reaction_coeff, next_time, arrivals)
        reaction_matrix = sparse.hstack(
            (sparse.diags(reaction), sparse.csr_matrix((n, nf - n))), format="csr"
        )
        top = identity - scale * self.dt * self.nu * lap.get_op() - scale * self.dt * reaction_matrix
        system = sparse.vstack((top, bc.get_op()), format="csr")
        if system.shape[0] != system.shape[1]:
            raise RuntimeError("moving-domain ghost closure did not produce a square system")
        normals = self.domain.get_nrmls()
        if callable(boundary_value):
            try:
                rhs_boundary = boundary_value(alpha, beta, normals, next_time, boundary_points)
            except TypeError:
                rhs_boundary = boundary_value(boundary_points)
        else:
            rhs_boundary = boundary_value
        rhs_boundary = np.asarray(rhs_boundary, dtype=float).reshape(-1)
        if rhs_boundary.size == 1:
            rhs_boundary = np.full(boundary_points.shape[0], rhs_boundary[0])
        rhs = np.concatenate((rhs_physical, rhs_boundary))
        tolerance = self.gmres_tolerance or min(0.1 * h**self.xi, 1.0e-7)
        preconditioner = build_ilu_preconditioner(system)
        guess_physical = departure_values[0]
        guess = np.concatenate((guess_physical, np.zeros(nf - n)))
        solution, info = spla.gmres(
            system,
            rhs,
            x0=guess,
            rtol=tolerance,
            atol=0.0,
            restart=min(self.gmres_restart, nf),
            maxiter=self.gmres_max_iterations,
            M=preconditioner,
        )
        residual = np.linalg.norm(system @ solution - rhs) / max(
            np.linalg.norm(rhs), np.finfo(float).eps
        )
        if info != 0 or not np.isfinite(residual):
            raise RuntimeError(f"moving-domain GMRES failed: info={info}, residual={residual:.3e}")
        state = solution[:n]
        stencil = _paper_stencil_properties(self.domain.get_dim(), self.xi + 1, "interior_boundary", "interior_boundary")
        self.interpolation_history = [_LocalRBFInterpolator(self.domain, stencil), *self.interpolation_history[:2]]
        self.state_history = [state, *self.state_history[:2]]
        self.current_time = float(next_time)
        self.last_linear_diagnostics = {"relative_residual": residual, "info": int(info)}
        self.last_update_diagnostics = {
            "laplacian_rows_recomputed": n,
            "boundary_rows_recomputed": boundary_points.shape[0],
        }
        return state

    def run(
        self,
        final_time: float,
        velocity: VelocityCallback,
        forcing: object = 0.0,
        neu_coeff: object = 0.0,
        dir_coeff: object = 1.0,
        boundary_value: object = 0.0,
        reaction_coeff: object = 0.0,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        step_count = round((final_time - self.current_time) / self.dt)
        times = self.current_time + np.arange(step_count + 1) * self.dt
        states = [self.current_state()]
        for time in times[1:]:
            states.append(
                self.step(
                    float(time),
                    velocity,
                    forcing,
                    neu_coeff,
                    dir_coeff,
                    boundary_value,
                    reaction_coeff,
                )
            )
        return times, states

    def current_state(self) -> np.ndarray:
        return np.zeros(0) if not self.state_history else self.state_history[0]

    def get_output_nodes(self) -> np.ndarray:
        return self.domain.get_int_bdry_nodes()

    def get_current_domain(self) -> DomainDescriptor:
        return self.domain
