"""Solve a manufactured ADR problem on a breathing sphere."""

import numpy as np

from kernelpack import geometry, manifold, solvers


def main() -> None:
    node_count = 256
    xi = 2
    dt = 0.01
    step_count = 10
    diffusivity = 0.03
    material_sites = geometry.fibonacci_sphere(node_count)
    radius = lambda time: 1.0 + 0.1 * np.sin(time)
    radius_rate = lambda time: 0.1 * np.cos(time)
    points_at = lambda time: radius(time) * material_sites
    exact = lambda time: np.exp(-time) * (2.0 + material_sites[:, 0])
    ell = manifold.surface_polynomial_degree(xi)
    neighbors = manifold.build_surface_stencil_graph(
        material_sites, manifold.surface_stencil_size(ell)
    )
    history = solvers.initialize_moving_surface_history(points_at(0.0), exact(0.0))
    operators = None
    cache = None
    calibration = None

    for step in range(1, step_count + 1):
        time = step * dt
        points = points_at(time)
        if cache is None:
            operators, cache = manifold.assemble_tangent_plane_operators(
                points, material_sites, neighbors, xi=xi
            )
        else:
            operators, cache, _ = manifold.update_tangent_plane_operators(
                points,
                material_sites,
                neighbors,
                cache,
                xi=xi,
                tolerance=1.0e-8,
            )
        h = np.sqrt(1.0 / node_count)
        if calibration is None:
            calibration = manifold.calibrate_surface_hyperviscosity(
                operators, points, material_sites, h, target_order=xi
            )
        else:
            calibration, drift = manifold.predict_surface_hyperviscosity(
                operators, points, h, calibration
            )
            if drift > 0.15:
                calibration = manifold.calibrate_surface_hyperviscosity(
                    operators,
                    points,
                    material_sites,
                    h,
                    target_order=xi,
                    power=calibration.power,
                )
        truth = exact(time)
        forcing = (
            -truth
            + 2.0 * radius_rate(time) / radius(time) * truth
            + 2.0 * diffusivity * np.exp(-time) * material_sites[:, 0] / radius(time) ** 2
        )
        quadrature = np.full(node_count, 4.0 * np.pi * radius(time) ** 2 / node_count)
        history, info = solvers.moving_surface_adr_step(
            history,
            points,
            operators,
            forcing,
            quadrature,
            8.0 * np.pi * radius(time) ** 2 * np.exp(-time),
            dt,
            diffusivity,
            calibration.gamma,
            1.0e-9,
            order=min(step, 3),
            hyperviscosity_power=calibration.power,
        )

    truth = exact(step_count * dt)
    error = np.linalg.norm(history.concentration[0] - truth) / np.linalg.norm(truth)
    print(f"relative l2 error: {error:.6e}")
    print(f"linear relative residual: {info.relative_residual:.6e}")
    print(f"mass error after projection: {info.mass_after_projection - 8.0 * np.pi * radius(step_count * dt) ** 2 * np.exp(-step_count * dt):.6e}")


if __name__ == "__main__":
    main()
