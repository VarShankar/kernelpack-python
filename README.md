# kernelpack-python

[![Python tests](https://github.com/VarShankar/kernelpack-python/actions/workflows/python.yml/badge.svg)](https://github.com/VarShankar/kernelpack-python/actions/workflows/python.yml)
[![Latest release](https://img.shields.io/github/v/release/VarShankar/kernelpack-python)](https://github.com/VarShankar/kernelpack-python/releases/latest)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776ab.svg)](#requirements)

**High-order meshfree numerics, from sampled geometry to PDE solution.**

`kernelpack-python` is the NumPy/SciPy implementation of the
[KernelPack](https://github.com/VarShankar/kernelpack) numerical toolkit. Its
purpose is to make modern meshfree methods usable as a coherent computational
stack: represent an irregular geometry, generate a well-spaced point cloud,
construct polynomially augmented local operators, and solve PDEs on fixed or
evolving domains and surfaces.

The library consolidates a broader research program in node generation,
geometric modeling, RBF-FD, partition-of-unity approximation, stabilization,
and PDEs on moving geometries. The Python implementation fits those methods
into familiar scientific-Python workflows while moving repeated local kernels
out of the interpreter with Numba.

![From sampled geometry to a meshfree domain](docs/readme_assets/geometry_domain.png)

[Approach](#the-kernelpack-approach) | [Capabilities](#numerical-stack) |
[Install](#installation) | [Quick start](#quick-start) |
[Workflows](#core-workflows) | [Examples](#examples) |
[Verification](#verification) | [Papers](#research-foundations)

## The KernelPack approach

KernelPack organizes a meshfree discretization into reusable numerical layers:

1. **Geometry.** Fit smooth or piecewise-smooth models to sampled boundaries
   and surfaces; evaluate positions, normals, projections, and level sets.
2. **Nodes.** Generate quasi-uniform or variable-density Poisson point clouds,
   then classify interior, boundary, ghost, and dual nodes from the geometry.
3. **Approximation.** Combine lower odd-degree polyharmonic splines with
   centered and scaled polynomial reproduction to build local RBF-FD,
   overlapped RBF-FD, weighted-least-squares, PU, and interpolation operators.
4. **PDEs.** Reuse the same domain descriptors and local approximation tools in
   elliptic, parabolic, advection-diffusion-reaction, and surface solvers.
5. **Evolution.** Update geometry, point clouds, operators, stabilization, and
   solution histories when a domain or manifold moves.

The result is a path from scattered geometric data to high-order PDE solvers
without constructing a conforming volume mesh.

## The KernelPack family

The repositories are sibling implementations of the same numerical ideas, not
language bindings and not exact API replicas.

| Implementation | Best suited for | Computational model |
| --- | --- | --- |
| [`kernelpack-matlab`](https://github.com/VarShankar/kernelpack-matlab) | Numerical-method development, transparent research prototypes, convergence studies, and publication workflows | MATLAB sparse linear algebra, vectorized kernels, and optional `parfor` assembly |
| **Python** (this repository) | Conventional scientific-Python applications and CPU workflows | NumPy/SciPy with Numba-compiled local kernels |
| [`kernelpack-jax`](https://github.com/VarShankar/kernelpack-jax) | Accelerator execution, batched studies, and fixed-topology differentiable computation | JAX `jit`/`vmap` kernels with optional Warp spatial primitives |

All three follow the same geometry -> nodes -> operators -> solvers
architecture. Features may arrive in one implementation before the others.

## Numerical stack

| Layer | Python implementation |
| --- | --- |
| Geometry | Smooth and piecewise-smooth embedded geometry, PHS fits, RBF level sets, projections, normals, parametric spherical and toroidal SBF models, and externally supplied material trajectories |
| Nodes and domains | Fixed- and variable-radius Poisson sampling, level-set clipping, boundary refinement, interior/boundary/ghost bookkeeping, dual node sets, and SciPy KD-tree search |
| Polynomial tools | Jacobi and Legendre recurrences, total-degree multi-indices, and centered/scaled polynomial evaluation |
| Local approximation | Standard and overlapped PHS+poly RBF-FD, cross-node operators, weighted least squares, localized PU approximation, and scalar or divergence-free interpolation |
| Fixed-domain PDEs | Poisson, variable and nonlinear variable-coefficient Poisson, BDF diffusion, standard and heterogeneous multispecies diffusion, and scalar or multispecies PU diffusion |
| Moving-domain PDEs | Semi-Lagrangian BDF1--BDF3 advection-diffusion-reaction with evolving embedded boundaries in two and three dimensions |
| Surface PDEs | Tangent-plane RBF-FD on stationary and moving manifolds, defect-corrected operator updates, surface hyperviscosity, geometry-based quadrature, conservation projection, marker rearrangement, and history backfill |
| Geometric evolution | Mean-curvature flow and interpolation of externally generated material-surface trajectories |
| Performance | Numba-parallel local kernels, SciPy batched factorizations and KD trees, sparse GMRES, and ILU preconditioning |

The primary namespaces are `kernelpack.geometry`, `kernelpack.nodes`,
`kernelpack.domain`, `kernelpack.poly`, `kernelpack.rbffd`,
`kernelpack.divfree`, `kernelpack.manifold`, and `kernelpack.solvers`.

Repeated local numerical kernels run in compiled Numba code. SciPy provides
KD-tree search, sparse matrices, batched linear algebra, GMRES, and ILU;
Python remains responsible for callbacks, orchestration, I/O, and
topology-changing node updates.

## Requirements

- Python 3.11 or newer
- NumPy 2.0 or newer
- SciPy 1.14 or newer
- Numba 0.61 or newer
- Matplotlib 3.9 or newer is optional and used by examples and figure scripts

## Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/VarShankar/kernelpack-python.git
cd kernelpack-python
python -m venv .venv
```

Activate it on macOS or Linux with `source .venv/bin/activate`, or on Windows
with `.venv\Scripts\Activate.ps1`. Then install the package:

```bash
python -m pip install -e ".[examples]"
```

For tests and package builds, install the development extras:

```bash
python -m pip install -e ".[dev]"
```

## Quick start

The following example samples a disk, builds a meshfree domain, and solves
$-\Delta u=4$ with homogeneous Dirichlet data.

![Poisson solution and nodal error on an embedded domain](docs/readme_assets/poisson_solution.png)

```python
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from kernelpack.geometry import EmbeddedSurface
from kernelpack.nodes import DomainNodeGenerator
from kernelpack.solvers import PoissonSolver

t = np.linspace(0.0, 2.0 * np.pi, 200, endpoint=False)
surface = EmbeddedSurface()
surface.set_data_sites(np.column_stack([np.cos(t), np.sin(t)]))
surface.build_closed_geometric_model_ps(2, 0.08, t.size)
surface.build_level_set_from_geometric_model()

domain = DomainNodeGenerator().build_domain_descriptor_from_geometry(
    surface, 0.08, seed=17, strip_count=5
)

solver = PoissonSolver(
    lap_assembler="fd",
    bc_assembler="fd",
    lap_stencil="rbf",
    bc_stencil="rbf",
)
solver.init(domain, 4)
result = solver.solve(
    lambda x: 4.0 * np.ones(x.shape[0]),
    lambda xb: np.zeros(xb.shape[0]),
    lambda xb: np.ones(xb.shape[0]),
    lambda neu, diri, normals, xb: np.zeros(xb.shape[0]),
)

x = domain.get_int_bdry_nodes()
tri = mtri.Triangulation(x[:, 0], x[:, 1])
plt.tripcolor(tri, result["u"], shading="gouraud")
plt.gca().set_aspect("equal")
plt.colorbar(label="u")
plt.show()
```

## Core workflows

### Fixed domains

A `DomainDescriptor` separates geometry and node generation from the PDE. The
same descriptor can feed standard or overlapped RBF-FD, weighted-least-
squares, and localized-PU operators. Elliptic solvers support Dirichlet,
Neumann, and mixed boundary rows; pure-Neumann Poisson uses an explicit
null-space constraint. BDF diffusion reuses time-independent operators and
preconditioners, while multispecies solvers support distinct diffusivities and
heterogeneous coupling.

```bash
python examples/poisson_solver_example.py
python examples/variable_poisson_solver_example.py
python examples/diffusion_solver_example.py
python examples/multispecies_diffusion_example.py
```

![BDF3 diffusion and its error history](docs/readme_assets/diffusion_solution.png)

### Moving domains

`MovingDomainADRSolver` advances advection-diffusion-reaction problems on
domains with moving embedded boundaries. RK3 advances boundary markers; the
geometric model reconstructs the new boundary; the background cloud is carved
and refilled; and local PHS+Legendre interpolation supplies semi-Lagrangian
BDF1--BDF3 history values. Diffusion and reaction are solved implicitly with
overlapped RBF-FD, GMRES, and ILU. The same implementation supports the public
two- and three-dimensional examples.

```bash
python examples/moving_domain_adr_example.py
python examples/moving_domain_adr_3d_example.py
```

### Surfaces and evolving manifolds

The manifold package constructs target-centered tangent-plane PHS+Legendre
RBF-FD operators. It supports stationary-surface ADR, conservative transport
on prescribed moving surfaces, cached-factor defect updates, adaptive
hyperviscosity, SBF or PCA normals, geometry-derived quadrature, conservation
projection, marker rearrangement, and semi-Lagrangian reconstruction of
multistep history. The same operator path drives mean-curvature flow.

```bash
python examples/stationary_surface_adr_example.py
python examples/moving_surface_adr_example.py
python examples/mean_curvature_flow_ellipsoid_example.py
python examples/surface_rearrangement_example.py
```

The IBAMR biconcave-membrane trajectory is one application of this general
surface machinery. It is distributed separately and installed with
`python scripts/download_rbc_capstone_data.py`; the complete replay is
`examples/moving_surface_rbc_capstone.py`.

### Interpolation and vector fields

`kernelpack.divfree` provides global and local divergence-free
PHS+polynomial interpolation. The shared polynomial, stencil, and search
layers can also be used directly for scattered-data approximation and custom
differential operators without adopting a packaged PDE solver.

## Examples

Complete workflows live in [`examples`](examples):

| Goal | Example |
| --- | --- |
| Solve Poisson on an embedded domain | [`poisson_solver_example.py`](examples/poisson_solver_example.py) |
| Verify pure-Neumann Poisson in two or three dimensions | [`poisson_convergence_2d_neumann.py`](examples/poisson_convergence_2d_neumann.py), [`poisson_convergence_3d_neumann.py`](examples/poisson_convergence_3d_neumann.py) |
| Solve variable and nonlinear variable-coefficient Poisson problems | [`variable_poisson_solver_example.py`](examples/variable_poisson_solver_example.py) |
| Compare RBF-FD and PU diffusion | [`diffusion_solver_example.py`](examples/diffusion_solver_example.py) |
| Compare multispecies formulations | [`multispecies_diffusion_example.py`](examples/multispecies_diffusion_example.py) |
| Solve moving-domain ADR in two or three dimensions | [`moving_domain_adr_example.py`](examples/moving_domain_adr_example.py), [`moving_domain_adr_3d_example.py`](examples/moving_domain_adr_3d_example.py) |
| Verify stationary or moving surface ADR | [`stationary_surface_adr_example.py`](examples/stationary_surface_adr_example.py), [`moving_surface_adr_example.py`](examples/moving_surface_adr_example.py) |
| Evolve surfaces by mean curvature | [`mean_curvature_flow_sphere_example.py`](examples/mean_curvature_flow_sphere_example.py), [`mean_curvature_flow_ellipsoid_example.py`](examples/mean_curvature_flow_ellipsoid_example.py) |
| Rearrange surface markers and backfill BDF history | [`surface_rearrangement_example.py`](examples/surface_rearrangement_example.py) |
| Replay transport on an IBAMR membrane trajectory | [`moving_surface_rbc_capstone.py`](examples/moving_surface_rbc_capstone.py) |

Generated tables and figures are written under `artifacts/`. Regenerate the
committed README figures with:

```bash
python scripts/render_readme_examples.py
```

## Verification

Install the development dependencies and run the complete public suite:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

The same suite runs on Python 3.11 and 3.12 in GitHub Actions for every push
and pull request.

## Research foundations

KernelPack consolidates methods developed across several publications. Cite
the software using [`CITATION.cff`](CITATION.cff), and cite the papers that
correspond to the components used in your work.

| Component | Publication |
| --- | --- |
| Surface RBF-FD | V. Shankar, G. B. Wright, R. M. Kirby, and A. L. Fogelson, [*A radial basis function (RBF)-finite difference (FD) method for diffusion and reaction-diffusion equations on surfaces*](https://doi.org/10.1007/s10915-014-9914-1), Journal of Scientific Computing 63 (2015), 745--768 |
| Overlapped RBF-FD | V. Shankar, [*The overlapped radial basis function-finite difference (RBF-FD) method: A generalization of RBF-FD*](https://doi.org/10.1016/j.jcp.2017.04.037), Journal of Computational Physics 342 (2017), 211--228 |
| Geometric models and Poisson node generation | V. Shankar, R. M. Kirby, and A. L. Fogelson, [*Robust node generation for mesh-free discretizations on irregular domains and surfaces*](https://doi.org/10.1137/17M114090X), SIAM Journal on Scientific Computing 40 (2018), A2584--A2608 |
| Bulk hyperviscosity and PHS-degree selection | V. Shankar and A. L. Fogelson, [*Hyperviscosity-based stabilization for radial basis function-finite difference (RBF-FD) discretizations of advection-diffusion equations*](https://doi.org/10.1016/j.jcp.2018.06.036), Journal of Computational Physics 372 (2018), 616--639 |
| Surface hyperviscosity | V. Shankar, G. B. Wright, and A. Narayan, [*A robust hyperviscosity formulation for stable RBF-FD discretizations of advection-diffusion-reaction equations on manifolds*](https://doi.org/10.1137/19M1288747), SIAM Journal on Scientific Computing 42 (2020), A2371--A2401 |
| Moving-domain ADR and matrix updates | V. Shankar, G. B. Wright, and A. L. Fogelson, [*An efficient high-order meshless method for advection-diffusion equations on time-varying irregular domains*](https://doi.org/10.1016/j.jcp.2021.110633), Journal of Computational Physics 445 (2021), 110633 |
| Lagrangian-Eulerian moving-surface ADR | M. Lowery, G. B. Wright, and V. Shankar, [*A high-order, meshless, Lagrangian--Eulerian RBF-FD method for advection--diffusion--reaction on moving manifolds*](https://doi.org/10.48550/arXiv.2608.19384), arXiv:2608.19384 (2026) |

## Project status

KernelPack is research software. The public API is usable and tested, but the
project is still evolving and may change as methods are consolidated across
the C++, MATLAB, Python, and JAX implementations. Bug reports, focused pull
requests, and reproducible numerical examples are welcome; see
[`CONTRIBUTING.md`](CONTRIBUTING.md) and [`SECURITY.md`](SECURITY.md).

## License

`kernelpack-python` is released under the [BSD 3-Clause License](LICENSE),
which permits academic and commercial use, modification, and redistribution
subject to its terms.
