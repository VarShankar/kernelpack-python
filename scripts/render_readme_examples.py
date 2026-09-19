from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from kernelpack.geometry import EmbeddedSurface
from kernelpack.nodes import DomainNodeGenerator
from kernelpack.solvers import DiffusionSolver, PoissonSolver


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "docs" / "readme_assets"
OUTDIR.mkdir(parents=True, exist_ok=True)

BACKGROUND = "#fbfaf6"
INK = "#172033"
INTERIOR = "#3157d5"
BOUNDARY = "#0f8b8d"
GHOST = "#f2a541"


def style_figure(fig) -> None:
    fig.patch.set_facecolor(BACKGROUND)


def style_surface_axes(ax) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor(BACKGROUND)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def build_surface(n_sites: int, geom_radius: float, aspect: float = 0.7) -> EmbeddedSurface:
    t = np.linspace(0.0, 2.0 * np.pi, n_sites, endpoint=False)
    curve = np.column_stack([np.cos(t), aspect * np.sin(t)])
    surface = EmbeddedSurface()
    surface.set_data_sites(curve)
    surface.build_closed_geometric_model_ps(2, geom_radius, curve.shape[0])
    surface.build_level_set_from_geometric_model()
    return surface


def build_domain(*, do_outer_refinement: bool = True) -> tuple[EmbeddedSurface, object]:
    surface = build_surface(160, 0.06, aspect=1.0)
    generator = DomainNodeGenerator()
    domain = generator.build_domain_descriptor_from_geometry(
        surface,
        0.08,
        seed=17,
        strip_count=5,
        do_outer_refinement=do_outer_refinement,
        outer_fraction_of_h=0.5,
        outer_refinement_zone_size_as_multiple_of_h=2.0,
    )
    return surface, domain


def save_geometry(surface, domain) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    style_figure(fig)
    xb = domain.get_bdry_nodes()
    nr = domain.get_nrmls()
    xi = domain.get_interior_nodes()
    xg = domain.get_ghost_nodes()

    axes[0].scatter(
        surface.data_sites[:, 0], surface.data_sites[:, 1],
        s=14, color="#cbd3df", edgecolors="none",
    )
    axes[0].scatter(xb[:, 0], xb[:, 1], s=12, color=BOUNDARY, edgecolors="none")
    step = max(1, xb.shape[0] // 40)
    axes[0].quiver(
        xb[::step, 0],
        xb[::step, 1],
        nr[::step, 0],
        nr[::step, 1],
        angles="xy",
        scale_units="xy",
        scale=13,
        color="#df5b49",
        width=0.004,
        headwidth=4,
    )
    axes[0].set_title("Geometry and outward normals", color=INK, weight="semibold", pad=10)

    axes[1].scatter(xi[:, 0], xi[:, 1], s=10, color=INTERIOR, edgecolors="none", label="interior")
    axes[1].scatter(xb[:, 0], xb[:, 1], s=11, color=BOUNDARY, edgecolors="none", label="boundary")
    axes[1].scatter(xg[:, 0], xg[:, 1], s=11, color=GHOST, edgecolors="none", label="ghost")
    axes[1].legend(
        frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.08),
        ncol=3, fontsize=10, markerscale=1.4,
    )
    axes[1].set_title("Geometry-defined node cloud", color=INK, weight="semibold", pad=10)

    for ax in axes:
        style_surface_axes(ax)
    fig.suptitle("From sampled geometry to a meshfree domain", color=INK, weight="bold", fontsize=16)

    path = OUTDIR / "geometry_domain.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_poisson(domain) -> Path:
    solver = PoissonSolver(
        lap_assembler="fd",
        bc_assembler="fd",
        lap_stencil="rbf",
        bc_stencil="rbf",
    )
    solver.init(domain, 4)

    u_exact = lambda x: 1.0 - x[:, 0] ** 2 - x[:, 1] ** 2
    forcing = lambda x: 4.0 * np.ones(x.shape[0])
    neu_coeff = lambda xb: np.zeros(xb.shape[0])
    dir_coeff = lambda xb: np.ones(xb.shape[0])
    bc = lambda neu_coeffs, dir_coeffs, nr, xb: u_exact(xb)

    result = solver.solve(forcing, neu_coeff, dir_coeff, bc)
    x_phys = domain.get_int_bdry_nodes()
    u = result["u"]
    u_true = u_exact(x_phys)
    tri = mtri.Triangulation(x_phys[:, 0], x_phys[:, 1])

    fig, ax = plt.subplots(figsize=(9.4, 4.8), constrained_layout=True)
    style_figure(fig)
    field = ax.tripcolor(tri, u, shading="gouraud", cmap="viridis")
    xb = domain.get_bdry_nodes()
    ax.plot(xb[:, 0], xb[:, 1], color=INK, lw=1.1, alpha=0.8)
    ax.tricontour(tri, u, levels=9, colors="white", linewidths=0.45, alpha=0.38)
    style_surface_axes(ax)
    ax.set_title("Meshfree Poisson solution", color=INK, weight="bold", fontsize=16, pad=12)
    colorbar = fig.colorbar(field, ax=ax, shrink=0.82, pad=0.03)
    colorbar.set_label(r"$u_h$", color=INK, weight="semibold")
    colorbar.outline.set_visible(False)

    path = OUTDIR / "poisson_solution.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_diffusion(domain) -> Path:
    solver = DiffusionSolver(
        lap_assembler="fd",
        bc_assembler="fd",
        lap_stencil="rbf",
        bc_stencil="rbf",
    )
    nu = 0.25
    dt = 0.02
    t_final = 0.50
    nsteps = int(round(t_final / dt))
    solver.init(domain, 4, dt, nu)

    x_phys = domain.get_int_bdry_nodes()
    u_exact = lambda time, x: np.exp(-time) * (x[:, 0] ** 2 + x[:, 1] ** 2)
    forcing = lambda nu_value, time, x: -np.exp(-time) * (x[:, 0] ** 2 + x[:, 1] ** 2) - 4.0 * nu_value * np.exp(-time)
    neu_coeff = lambda xb: np.zeros(xb.shape[0])
    dir_coeff = lambda xb: np.ones(xb.shape[0])
    bc = lambda neu_coeffs, dir_coeffs, nr, time, xb: u_exact(time, xb)

    solver.set_initial_state(u_exact(0.0, x_phys))
    times = [0.0]
    errors = [0.0]

    for step in range(1, nsteps + 1):
        time = step * dt
        if step == 1:
            u_next = solver.bdf1_step(time, forcing, neu_coeff, dir_coeff, bc)
        elif step == 2:
            u_next = solver.bdf2_step(time, forcing, neu_coeff, dir_coeff, bc)
        else:
            u_next = solver.bdf3_step(time, forcing, neu_coeff, dir_coeff, bc)
        times.append(time)
        errors.append(float(np.max(np.abs(u_next - u_exact(time, x_phys)))))

    u_final = solver.current_physical_state()
    u_true_final = u_exact(t_final, x_phys)
    tri = mtri.Triangulation(x_phys[:, 0], x_phys[:, 1])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    style_figure(fig)
    field = axes[0].tripcolor(tri, u_final, shading="gouraud", cmap="viridis")
    axes[0].tricontour(tri, u_final, levels=8, colors="white", linewidths=0.4, alpha=0.35)
    style_surface_axes(axes[0])
    axes[0].set_title(rf"BDF3 diffusion at $t={t_final:.2f}$", color=INK, weight="semibold")
    colorbar = fig.colorbar(field, ax=axes[0], shrink=0.82, pad=0.03)
    colorbar.set_label(r"$u_h$", color=INK, weight="semibold")
    colorbar.outline.set_visible(False)

    positive_times = np.asarray(times[1:])
    positive_errors = np.maximum(np.asarray(errors[1:]), np.finfo(float).tiny)
    axes[1].semilogy(positive_times, positive_errors, color=INTERIOR, lw=2.6)
    axes[1].scatter(positive_times, positive_errors, color=BOUNDARY, s=24, zorder=3)
    axes[1].set_title("Error through the time march", color=INK, weight="semibold")
    axes[1].set_xlabel(r"time $t$", color=INK)
    axes[1].set_ylabel(r"$\|u_h-u\|_{\infty}$", color=INK)
    axes[1].grid(alpha=0.18, which="both")
    axes[1].set_facecolor(BACKGROUND)
    for spine in ("top", "right"):
        axes[1].spines[spine].set_visible(False)

    path = OUTDIR / "diffusion_solution.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    geometry_surface, solver_domain = build_domain(do_outer_refinement=False)
    paths = [
        save_geometry(geometry_surface, solver_domain),
        save_poisson(solver_domain),
        save_diffusion(solver_domain),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
