# kernelpack-python

[![Python tests](https://github.com/VarShankar/kernelpack-python/actions/workflows/python.yml/badge.svg)](https://github.com/VarShankar/kernelpack-python/actions/workflows/python.yml)
[![Latest release](https://img.shields.io/github/v/release/VarShankar/kernelpack-python)](https://github.com/VarShankar/kernelpack-python/releases/latest)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776ab.svg)](#requirements)

**Meshfree geometry, RBF-FD, partition-of-unity methods, and PDE solvers for
Python.**

`kernelpack-python` provides Python implementations of meshfree geometry,
scattered-node discretizations, and PDE solvers on embedded fixed domains. It
combines reusable geometry and node-generation tools with standard and
overlapped PHS+poly RBF-FD, weighted least squares, localized
partition-of-unity approximations, and sparse elliptic and diffusion solvers.

The package is the Python member of the KernelPack family. The companion
[`kernelpack-matlab`](https://github.com/VarShankar/kernelpack-matlab)
repository includes the moving-surface implementation accompanying the
preprint [*A high-order, meshless, Lagrangian--Eulerian RBF-FD method for
advection--diffusion--reaction on moving manifolds*](https://arxiv.org/abs/2608.19384)
by Matthew Lowery, Grady B. Wright, and Varun Shankar. This Python release is
focused on the fixed-domain numerical core and does not claim that
moving-surface solver.

![Poisson solution and nodal error on an embedded domain](docs/readme_assets/poisson_solution.png)

The figure shows an end-to-end Poisson solve on a geometry-defined scattered
node set: the geometric model supplies the boundary and normals, the node
generator fills the domain, and PHS+poly RBF-FD supplies the differential and
boundary operators.

[Install](#installation) | [First solve](#first-solve) |
[Examples](#examples) | [Tests](#verification) |
[Papers](#research-foundations) | [Citation](#citation)

## Who this is for

This package is intended for numerical PDE researchers and Python users who
want to:

- prototype PHS+poly RBF-FD or weighted-least-squares discretizations;
- generate scattered nodes and differential operators on embedded domains;
- compare standard and overlapped local assembly;
- solve elliptic and diffusion problems without constructing a volume mesh;
- build localized PU or divergence-free RBF approximations; or
- extend a tested, inspectable numerical research codebase.

It is research software, not a general-purpose finite-element package.

## At a glance

| Component | What the public release provides |
| --- | --- |
| Geometry models | Smooth and piecewise-smooth embedded boundaries and surfaces, PHS geometric fits, RBF level sets, normals, projection, and geometry-aware bounding data |
| Node generation | Seeded fixed- and variable-radius Poisson sampling in boxes, clipping by embedded geometry, boundary and ghost nodes, boundary-zone outer refinement, and dual node sets |
| Local approximation | Centered and scaled Legendre polynomial bases, standard and overlapped PHS+poly RBF-FD, weighted-least-squares stencils, and local divergence-free PHS interpolation |
| Fixed-domain solvers | Poisson, variable-coefficient and nonlinear variable-coefficient Poisson, BDF1--BDF3 diffusion, localized PU diffusion, and homogeneous or heterogeneous multispecies diffusion |
| Numerical infrastructure | SciPy KD trees and sparse matrices, cached local polynomial templates, and targeted Numba kernels for repeated geometry, polynomial, and stencil calculations |

The main namespaces are `kernelpack.geometry`, `kernelpack.nodes`,
`kernelpack.domain`, `kernelpack.poly`, `kernelpack.rbffd`,
`kernelpack.divfree`, and `kernelpack.solvers`.

## Requirements

- Python 3.11 or newer
- NumPy 2.0 or newer
- SciPy 1.14 or newer
- Numba 0.61 or newer
- Matplotlib 3.9 or newer for examples and figures

## Installation

### Clone the repository

```bash
git clone https://github.com/VarShankar/kernelpack-python.git
cd kernelpack-python
python -m venv .venv
```

Activate the environment on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install the package and example dependencies:

```bash
python -m pip install -e ".[examples]"
```

For development, install the test and build tools as well:

```bash
python -m pip install -e ".[dev]"
```

## First solve

This example constructs a disk from boundary samples, generates interior and
ghost nodes, solves $-\Delta u = 4$ with $u=0$ on the boundary, and plots the
numerical solution.

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

generator = DomainNodeGenerator()
domain = generator.build_domain_descriptor_from_geometry(
    surface, 0.08, seed=17, strip_count=5
)

solver = PoissonSolver(
    lap_assembler="fd",
    bc_assembler="fd",
    lap_stencil="rbf",
    bc_stencil="rbf",
)
solver.init(domain, 4)

forcing = lambda x: 4.0 * np.ones(x.shape[0])
neumann = lambda xb: np.zeros(xb.shape[0])
dirichlet = lambda xb: np.ones(xb.shape[0])
boundary_data = lambda neu, diri, normals, xb: np.zeros(xb.shape[0])
result = solver.solve(forcing, neumann, dirichlet, boundary_data)

x = domain.get_int_bdry_nodes()
tri = mtri.Triangulation(x[:, 0], x[:, 1])
plt.tripcolor(tri, result["u"], shading="gouraud")
plt.gca().set_aspect("equal")
plt.colorbar(label="u")
plt.title("Poisson solution")
plt.show()
```

![Geometry-clipped interior, boundary, and ghost nodes](docs/readme_assets/geometry_domain.png)

## Solver workflows

All solvers use a `DomainDescriptor`, so geometry, node generation, and local
operator construction remain separate from the PDE definition. A target
spatial order `xi` determines the polynomial reproduction degree and the lower
odd-degree PHS used by the local stencil builders.

The elliptic solver family supports Dirichlet, Neumann, and mixed boundary
rows. Pure-Neumann systems use an explicit null-space augmentation. The
variable-coefficient solver assembles the divergence-form operator, while the
nonlinear variant applies Newton iterations with sparse linear solves.

`DiffusionSolver` advances fixed-domain problems with BDF1, BDF2, or BDF3 and
reuses time-independent operators and preconditioners where possible. The PU
variants localize approximation and support single- or multispecies diffusion,
including distinct diffusivities by species.

![Diffusion solution, error, and time history](docs/readme_assets/diffusion_solution.png)

## Examples

Complete convergence drivers live in [`examples`](examples):

| Goal | Example |
| --- | --- |
| Verify a two-dimensional pure-Neumann Poisson solve | [`poisson_convergence_2d_neumann.py`](examples/poisson_convergence_2d_neumann.py) |
| Verify a three-dimensional pure-Neumann Poisson solve | [`poisson_convergence_3d_neumann.py`](examples/poisson_convergence_3d_neumann.py) |

Run a study from the repository root, for example:

```bash
python examples/poisson_convergence_2d_neumann.py --orders 2 4 6
```

Generated tables, JSON data, and figures are written under `artifacts/`, which
is intentionally excluded from version control. The committed README figures
can be regenerated with:

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

`kernelpack-python` brings together methods developed across several papers.
Please cite the publications corresponding to the parts of the library used in
your work.

| Code or method | Publication |
| --- | --- |
| Overlapped RBF-FD assembly (`kernelpack.rbffd.FDODiffOp`) | V. Shankar, [*The overlapped radial basis function-finite difference (RBF-FD) method: A generalization of RBF-FD*](https://doi.org/10.1016/j.jcp.2017.04.037), Journal of Computational Physics 342 (2017), 211--228 |
| PHS geometric models and Poisson node generation (`kernelpack.geometry`, `kernelpack.nodes`) | V. Shankar, R. M. Kirby, and A. L. Fogelson, [*Robust node generation for mesh-free discretizations on irregular domains and surfaces*](https://doi.org/10.1137/17M114090X), SIAM Journal on Scientific Computing 40 (2018), A2584--A2608 |
| Lower odd-degree PHS selection used by the solver stencil defaults | V. Shankar and A. L. Fogelson, [*Hyperviscosity-based stabilization for radial basis function-finite difference (RBF-FD) discretizations of advection-diffusion equations*](https://doi.org/10.1016/j.jcp.2018.06.036), Journal of Computational Physics 372 (2018), 616--639 |

For moving-surface ADR, surface hyperviscosity, and the associated research
drivers, see the public
[`kernelpack-matlab`](https://github.com/VarShankar/kernelpack-matlab)
release and its research-foundations table.

## Citation

Software citation metadata are provided in [`CITATION.cff`](CITATION.cff).
Please also cite the method papers corresponding to the components used in
your work.

## Contributing

Bug reports, focused pull requests, and reproducible numerical examples are
welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the development workflow
and [`SECURITY.md`](SECURITY.md) for responsible vulnerability reporting.

## License

`kernelpack-python` is released under the [BSD 3-Clause License](LICENSE),
which permits academic and commercial use, modification, and redistribution
subject to its terms.
