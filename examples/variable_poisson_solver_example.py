"""Solve linear and nonlinear variable-coefficient Poisson problems."""

import numpy as np

from kernelpack import solvers

from _common import build_square_domain, exact_solution, unit_dirichlet, zero_neumann


def main() -> None:
    domain = build_square_domain()
    points = domain.get_int_bdry_nodes()
    exact = lambda x: exact_solution(0.0, x)
    coefficient = lambda x: 1.0 + 0.2 * x[:, 0]

    linear = solvers.VariablePoissonSolver(
        lap_assembler="fd",
        bc_assembler="fd",
        lap_stencil="wls",
        bc_stencil="wls",
    )
    linear.init(domain, 3)
    linear_result = linear.solve(
        lambda x: -4.0 - 1.2 * x[:, 0],
        coefficient,
        zero_neumann,
        unit_dirichlet,
        lambda alpha, beta, normals, xb: exact(xb),
    )

    nonlinear_exact = lambda x: 2.0 + x[:, 0] + x[:, 1]
    nonlinear = solvers.NonlinearVariablePoissonSolver(
        lap_assembler="fd",
        bc_assembler="fd",
        lap_stencil="wls",
        bc_stencil="wls",
    )
    nonlinear.init(domain, 3)
    nonlinear_result = nonlinear.solve(
        lambda x, u: -0.2 * np.ones(x.shape[0]),
        lambda x, u: np.zeros(x.shape[0]),
        lambda x, u: 1.0 + 0.1 * u,
        lambda x, u: 0.1 * np.ones(x.shape[0]),
        zero_neumann,
        unit_dirichlet,
        lambda alpha, beta, normals, xb: nonlinear_exact(xb),
        initial_guess=np.zeros(points.shape[0]),
    )

    linear_error = np.linalg.norm(linear_result["u"] - exact(points)) / np.linalg.norm(exact(points))
    nonlinear_truth = nonlinear_exact(points)
    nonlinear_error = np.linalg.norm(nonlinear_result["u"] - nonlinear_truth) / np.linalg.norm(nonlinear_truth)
    print(f"linear relative l2 error: {linear_error:.6e}")
    print(f"nonlinear relative l2 error: {nonlinear_error:.6e}")
    print(f"nonlinear residual: {nonlinear_result['residual_norm']:.6e}")


if __name__ == "__main__":
    main()
