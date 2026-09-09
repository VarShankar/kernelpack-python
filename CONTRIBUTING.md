# Contributing

Bug reports, focused pull requests, and reproducible numerical examples are
welcome.

## Development setup

```bash
git clone https://github.com/VarShankar/kernelpack-python.git
cd kernelpack-python
python -m venv .venv
python -m pip install -e ".[dev]"
python -m pytest -q
```

On Windows, activate the environment with `.venv\Scripts\activate`; on macOS
or Linux, use `source .venv/bin/activate`.

## Pull requests

- Keep each change focused and explain the numerical or software motivation.
- Add or update tests for changed behavior.
- Include a reproducible example when proposing a new numerical method.
- Run the complete test suite before opening a pull request.
- Do not commit generated build products, virtual environments, or large data.

Changes to a discretization should document the operator convention, stencil
construction, polynomial degree, boundary treatment, and validation problem.
Performance claims should include the problem size and a reproducible timing
procedure.
