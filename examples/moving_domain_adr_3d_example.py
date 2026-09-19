"""Short three-dimensional moving-cavity ADR verification case."""

import numpy as np

from kernelpack import geometry, nodes, solvers


def sphere(center, radius, count, h, *, build_model):
    sites = np.asarray(center) + radius * geometry.fibonacci_sphere(count)
    surface = geometry.EmbeddedSurface()
    surface.set_data_sites(sites)
    if build_model:
        samples = max(12, int(np.ceil(4.0 * np.pi * radius**2 / h**2)))
        surface.build_closed_geometric_model_ps(3, h, count, samples)
        surface.build_level_set_from_geometric_model()
    return surface


def exact(time, points):
    x, y, z = points.T
    return 1.0 + np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z) * np.sin(np.pi * time)


def velocity(time, points):
    x, y, z = points.T
    scale = 1.5 * np.sin(np.pi * (x * x + y * y + z * z)) * np.sin(np.pi * time)
    return scale[:, None] * np.column_stack((y * z, -2.0 * x * z, x * y))


def forcing(nu, time, points):
    x, y, z = points.T
    sx, cx = np.sin(np.pi * x), np.cos(np.pi * x)
    sy, cy = np.sin(np.pi * y), np.cos(np.pi * y)
    sz, cz = np.sin(np.pi * z), np.cos(np.pi * z)
    st = np.sin(np.pi * time)
    time_derivative = np.pi * sx * cy * cz * np.cos(np.pi * time)
    gradient = np.pi * st * np.column_stack((cx * cy * cz, -sx * sy * cz, -sx * cy * sz))
    laplacian = -3.0 * np.pi**2 * sx * cy * cz * st
    return time_derivative + np.sum(velocity(time, points) * gradient, axis=1) - nu * laplacian


def main():
    h = 0.32
    dt = 0.01
    outer = sphere([0.0, 0.0, 0.0], 1.0, 180, h, build_model=True)
    hole = sphere([0.35, 0.20, 0.15], 0.18, 80, h, build_model=False)
    background = nodes.DomainNodeGenerator().build_domain_descriptor_from_geometry(
        outer, h, seed=2021, strip_count=3
    )
    solver = solvers.MovingDomainADRSolver(gmres_tolerance=1.0e-8)
    solver.init(background, [hole], xi=4, dt=dt, diffusivity=1.0e-3)
    solver.set_initial_state(exact)
    solver.step(
        dt,
        velocity,
        forcing,
        0.0,
        1.0,
        lambda alpha, beta, normals, time, points: exact(time, points),
        0.0,
    )
    truth = exact(dt, solver.get_output_nodes())
    relative_error = np.linalg.norm(solver.current_state() - truth) / np.linalg.norm(truth)
    print(f"nodes={truth.size}, relative_l2_error={relative_error:.6e}")
    print(solver.last_linear_diagnostics)


if __name__ == "__main__":
    main()
