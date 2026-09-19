import numpy as np

from kernelpack import _numba
from kernelpack import geometry, manifold, solvers


def sphere_operators(node_count=128, xi=2):
    points = geometry.fibonacci_sphere(node_count)
    ell = manifold.surface_polynomial_degree(xi)
    neighbors = manifold.build_surface_stencil_graph(
        points, manifold.surface_stencil_size(ell)
    )
    operators, cache = manifold.assemble_tangent_plane_operators(
        points, points, neighbors, xi=xi
    )
    return points, neighbors, operators, cache


def _write_synthetic_ibamr_trajectory(folder, points):
    count = points.shape[0]
    node_ids = np.arange(count)
    np.savetxt(
        folder / "material.csv",
        np.column_stack((node_ids, points)),
        delimiter=",",
        header="node_id,u_x,u_y,u_z",
        comments="",
    )
    np.savetxt(
        folder / "faces.csv",
        np.array([[0, 1, 2]]),
        delimiter=",",
        header="i,j,k",
        comments="",
        fmt="%d",
    )
    np.savetxt(
        folder / "diagnostics.csv",
        np.array([[0, 0.0], [1, 0.01]]),
        delimiter=",",
        header="step,time",
        comments="",
    )
    for step, radius in ((0, 1.0), (1, 1.001)):
        positions = radius * points
        velocities = 0.1 * points
        np.savetxt(
            folder / f"frame_{step:06d}.csv",
            np.column_stack((node_ids, positions, velocities)),
            delimiter=",",
            header="node_id,x,y,z,u,v,w",
            comments="",
        )


def test_surface_degree_stencil_and_laplacian_rules():
    assert [manifold.surface_polynomial_degree(xi) for xi in (2, 4, 6)] == [3, 5, 7]
    assert [manifold.surface_phs_degree(ell) for ell in (3, 5, 7)] == [5, 5, 7]
    assert [manifold.surface_stencil_size(ell) for ell in (3, 5, 7)] == [21, 43, 73]
    points, _, operators, _ = sphere_operators()
    numerical = manifold.apply_surface_operator(
        operators.laplacian_weights, operators.neighbors, points
    )
    error = np.linalg.norm(numerical + 2.0 * points) / np.linalg.norm(2.0 * points)
    assert error < 2.0e-2


def test_surface_operator_defect_update_and_adr_step():
    points, neighbors, operators, cache = sphere_operators()
    deformed = points.copy()
    deformed[:, 0] *= 1.0001
    normals = deformed / np.linalg.norm(deformed, axis=1, keepdims=True)
    _, _, update = manifold.update_tangent_plane_operators(
        deformed,
        normals,
        neighbors,
        cache,
        xi=2,
        tolerance=1.0e-8,
    )
    assert np.max(update.relative_residual) <= 1.0e-8

    dt = 0.01
    diffusivity = 0.03
    exact = lambda time: np.exp(-time) * (2.0 + points[:, 0])
    history = solvers.initialize_moving_surface_history(points, exact(0.0))
    forcing = -exact(dt) + 2.0 * diffusivity * np.exp(-dt) * points[:, 0]
    quadrature = np.full(points.shape[0], 4.0 * np.pi / points.shape[0])
    updated, info = solvers.moving_surface_adr_step(
        history,
        points,
        operators,
        forcing,
        quadrature,
        8.0 * np.pi * np.exp(-dt),
        dt,
        diffusivity,
        np.zeros(3),
        1.0e-9,
        order=1,
        hyperviscosity_power=0,
    )
    error = np.linalg.norm(updated.concentration[0] - exact(dt)) / np.linalg.norm(exact(dt))
    assert error < 1.0e-4
    assert info.relative_residual < 1.0e-8
    assert abs(info.mass_after_projection - 8.0 * np.pi * np.exp(-dt)) < 1.0e-12


def test_mean_curvature_flow_matches_sphere_radius_law():
    radius_zero = 1.4
    dt = 2.0e-4
    directions, neighbors, _, _ = sphere_operators()
    points = radius_zero * directions
    operators, _ = manifold.assemble_tangent_plane_operators(
        points, directions, neighbors, xi=2
    )
    updated, info = solvers.mean_curvature_flow_step(
        points, directions, operators, dt
    )
    exact_radius = np.sqrt(radius_zero**2 - 4.0 * dt)
    error = np.linalg.norm(np.linalg.norm(updated, axis=1) - exact_radius)
    error /= np.linalg.norm(np.full(points.shape[0], exact_radius))
    assert error < 5.0e-5
    assert info.linear_info == 0
    assert info.relative_residual < 1.0e-8


def test_sbf_geometry_transfer_and_hyperviscosity_are_finite():
    controls = geometry.fibonacci_sphere(64)
    evaluation = geometry.fibonacci_sphere(128)
    model = manifold.build_spherical_sbf_model(controls, evaluation)
    surface = manifold.evaluate_spherical_sbf_geometry(model, controls)
    assert np.max(np.abs(np.linalg.norm(surface.points, axis=1) - 1.0)) < 1.0e-4
    assert abs(np.sum(surface.quadrature_weights) - 4.0 * np.pi) < 3.0e-4

    points, _, operators, _ = sphere_operators(node_count=96)
    calibration = manifold.calibrate_surface_hyperviscosity(
        operators,
        points,
        points,
        np.sqrt(1.0 / points.shape[0]),
        target_order=2,
    )
    assert calibration.power >= 1
    assert np.all(np.isfinite(calibration.gamma))

    angle = 0.02
    query = points.copy()
    query[:, 0] = np.cos(angle) * points[:, 0] - np.sin(angle) * points[:, 1]
    query[:, 1] = np.sin(angle) * points[:, 0] + np.cos(angle) * points[:, 1]
    ell = manifold.surface_polynomial_degree(2)
    cross = manifold.build_cross_surface_stencil_graph(
        query, points, manifold.surface_stencil_size(ell)
    )
    weights = manifold.local_tangent_interpolation_weights(
        points, query, query, cross, xi=2
    )
    target = lambda sites: 2.0 + sites[:, 0] + 0.2 * sites[:, 1] * sites[:, 2]
    values = manifold.apply_local_tangent_interpolant(weights, cross, target(points))
    error = np.linalg.norm(values - target(query)) / np.linalg.norm(target(query))
    assert error < 5.0e-5


def test_ibamr_trajectory_reconstructs_geometry_and_velocity(tmp_path):
    points = np.asarray(geometry.fibonacci_sphere(64))
    _write_synthetic_ibamr_trajectory(tmp_path, points)
    trajectory = manifold.IBAMRSurfaceTrajectory(
        tmp_path, xi=2, control_point_count=points.shape[0]
    )
    surface, velocity = trajectory.geometry(1)
    assert surface.points.shape == points.shape
    assert velocity.shape == points.shape
    assert np.max(np.abs(np.linalg.norm(surface.normals, axis=1) - 1.0)) < 1.0e-10
    assert np.all(surface.quadrature_weights > 0.0)
    assert np.all(np.isfinite(velocity))


def test_surface_hot_paths_compile_numba_kernels():
    points, neighbors, operators, _ = sphere_operators(node_count=96)
    manifold.apply_surface_operator(
        operators.laplacian_weights, neighbors, points[:, 0]
    )
    model = manifold.build_spherical_sbf_model(points[:64], points)
    manifold.evaluate_spherical_sbf_geometry(model, points[:64])
    manifold.evaluate_spherical_sbf_field(model, points[:64], points)
    manifold.farthest_point_subset(points, count=32)

    assert _numba.NUMBA_AVAILABLE
    assert _numba._surface_local_systems_numba.signatures
    assert _numba._surface_apply_scalar_numba.signatures
    assert _numba._sbf_basis_derivatives_numba.signatures
    assert _numba._sbf_basis_numba.signatures
    assert _numba._farthest_point_subset_numba.signatures
