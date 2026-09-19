"""Manufactured moving-hole ADR example using the NumPy/SciPy solver."""

import numpy as np

from kernelpack import geometry, nodes, solvers


REACTION = -0.3


def closed_curve(center, radius, seed_count, h, *, build_model):
    theta = np.arange(seed_count) * (2.0 * np.pi / seed_count)
    sites = np.asarray(center) + radius * np.column_stack((np.cos(theta), np.sin(theta)))
    surface = geometry.EmbeddedSurface()
    surface.set_data_sites(sites)
    if build_model:
        sample_count = max(8, int(np.ceil(2.0 * np.pi * radius / h)))
        surface.build_closed_geometric_model_ps(2, h, seed_count, sample_count)
        surface.build_level_set_from_geometric_model()
    return surface


def exact_solution(time, points):
    return np.exp(-time) * (2.0 + points[:, 0])


def rotation_velocity(time, points):
    del time
    return np.column_stack((-points[:, 1], points[:, 0]))


def manufactured_forcing(nu, time, points):
    del nu
    exact = exact_solution(time, points)
    return -exact - np.exp(-time) * points[:, 1] - REACTION * exact


def manufactured_boundary(alpha, beta, normals, time, points):
    del alpha, beta, normals
    return exact_solution(time, points)


def main(h=0.16, step_count=3):
    dt = 0.01
    outer = closed_curve([0.0, 0.0], 1.0, 120, h, build_model=True)
    hole = closed_curve([0.35, 0.0], 0.18, 48, h, build_model=False)
    background = nodes.DomainNodeGenerator().build_domain_descriptor_from_geometry(
        outer,
        h,
        seed=29,
        strip_count=1,
        do_outer_refinement=True,
        outer_fraction_of_h=0.5,
        outer_refinement_zone_size_as_multiple_of_h=2.0,
    )
    solver = solvers.MovingDomainADRSolver(gmres_tolerance=1.0e-9)
    solver.init(background, [hole], xi=2, dt=dt, diffusivity=0.03)
    solver.set_initial_state(exact_solution)
    times, states = solver.run(
        step_count * dt,
        rotation_velocity,
        manufactured_forcing,
        0.0,
        1.0,
        manufactured_boundary,
        REACTION,
    )
    truth = exact_solution(float(times[-1]), solver.get_output_nodes())
    relative_error = np.linalg.norm(states[-1] - truth) / np.linalg.norm(truth)
    print(f"nodes={truth.size}, relative_l2_error={relative_error:.6e}")
    print(solver.last_update_diagnostics)
    print(solver.last_linear_diagnostics)
    return {
        "node_count": int(truth.size),
        "relative_l2_error": float(relative_error),
        "step_count": int(step_count),
    }


if __name__ == "__main__":
    main()
