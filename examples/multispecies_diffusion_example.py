"""Advance two species with standard, heterogeneous, and PU diffusion solvers."""

import numpy as np

from kernelpack import solvers

from _common import build_square_domain


def exact(time: float, points: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (
            np.exp(-time) * (points[:, 0] ** 2 + points[:, 1] ** 2),
            np.exp(-2.0 * time) * (1.0 + points[:, 0]),
        )
    )


def relative_error(numerical: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(numerical - truth) / np.linalg.norm(truth))


def main() -> None:
    domain = build_square_domain()
    points = domain.get_int_bdry_nodes()
    dt = 0.02
    diffusivity = 0.2
    forcing = lambda nu, time, x: np.column_stack(
        (
            -np.exp(-time) * (x[:, 0] ** 2 + x[:, 1] ** 2)
            - 4.0 * nu * np.exp(-time),
            -2.0 * np.exp(-2.0 * time) * (1.0 + x[:, 0]),
        )
    )
    boundary = lambda alpha, beta, normals, time, xb: exact(time, xb)
    zero = lambda xb: np.zeros(xb.shape[0])
    one = lambda xb: np.ones(xb.shape[0])

    standard = solvers.MultiSpeciesDiffusionSolver(
        lap_assembler="fd",
        bc_assembler="fd",
        lap_stencil="wls",
        bc_stencil="wls",
    )
    standard.init(domain, 3, dt, diffusivity)
    standard.set_initial_state(exact(0.0, points))
    standard_solution = standard.bdf1_step(dt, forcing, zero, one, boundary)

    pu = solvers.MultiSpeciesPUDiffusionSolver()
    pu.init(domain, 3, dt, diffusivity)
    pu.set_initial_state(exact(0.0, points))
    pu_solution = pu.bdf1_step(dt, forcing, zero, one, boundary)

    heterogeneous = solvers.HeterogeneousMultiSpeciesDiffusionSolver(
        lap_assembler="fd",
        bc_assembler="fd",
        lap_stencil="wls",
        bc_stencil="wls",
    )
    diffusivities = np.array([0.15, 0.25])
    heterogeneous.init(domain, 3, dt, diffusivities)
    heterogeneous.set_initial_state(exact(0.0, points))
    heterogeneous_solution = heterogeneous.bdf1_step(
        dt,
        lambda species, nu, time, x: forcing(nu, time, x)[:, species],
        lambda species, time, xb: zero(xb),
        lambda species, time, xb: one(xb),
        lambda species, alpha, beta, normals, time, xb: exact(time, xb)[:, species],
    )

    truth = exact(dt, points)
    print(f"standard relative Frobenius error: {relative_error(standard_solution, truth):.6e}")
    print(f"PU relative Frobenius error: {relative_error(pu_solution, truth):.6e}")
    print(f"heterogeneous relative Frobenius error: {relative_error(heterogeneous_solution, truth):.6e}")


if __name__ == "__main__":
    main()
