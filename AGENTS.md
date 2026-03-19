# Repository Guidelines

## Project Structure & Module Organization
Core library code lives in `src/lerobot/`, with major areas such as `policies/`, `datasets/`, `envs/`, `robots/`, and CLI entrypoints under `src/lerobot/scripts/`. Repository-level automation and GR-2 deployment helpers live in `scripts/` and `scripts/train/`. Tests mirror the package layout in `tests/` (`tests/policies/`, `tests/datasets/`, `tests/robots/`, etc.). Documentation sources are in `docs/source/`, runnable examples in `examples/`, and generated artifacts should stay in `outputs/`, `Log/`, or local scratch directories, not in commits.

## Build, Test, and Development Commands
Use Python 3.10.

```bash
pip install -e ".[dev,test]"
pre-commit install
pre-commit run --all-files
pytest tests -vv --maxfail=10
make test-end-to-end DEVICE=cpu
```

Use `pip install -e ".[groot]"` or other extras only when working on those policy stacks. For fast iteration, run a focused test such as `pytest tests/datasets/test_dataset_tools.py -vv`.

## Coding Style & Naming Conventions
Formatting and linting are enforced through `pre-commit`, `ruff-format`, `ruff --fix`, `pyupgrade`, `typos`, `bandit`, and `mypy`. Follow Ruff settings in `pyproject.toml`: 4-space indentation, double quotes, and a soft line limit of 110. Use `snake_case` for modules, functions, variables, and CLI/config keys; use `PascalCase` for classes and dataclasses. Keep new code typed where practical; typed modules under `configs`, `envs`, `cameras`, `optim`, and `transport` already have stricter mypy coverage.

## Testing Guidelines
Tests use `pytest` with fixtures in `tests/fixtures/` and shared setup in `tests/conftest.py`. Name files `test_<feature>.py` and keep them near the subsystem they cover. Add targeted unit tests for every behavior change and run the smallest relevant slice locally before broader suites. Hardware- or extra-dependent tests should continue to use `pytest.importorskip(...)` or clear skip guards.

## Commit & Pull Request Guidelines
Recent history follows Conventional Commit style with optional scopes, for example `feat(scripts): add GR2 inference service` or `fix: correct camera handling`. Keep commit titles short, imperative, and scoped to one change. PRs should follow `.github/PULL_REQUEST_TEMPLATE.md`: include summary/motivation, linked issues, concrete test commands, documentation updates when needed, and any reviewer notes. Run `pre-commit run --all-files` and the relevant `pytest` commands before opening the PR.

## Security & Configuration Tips
Do not commit secrets, dataset credentials, model checkpoints, or large generated artifacts. `gitleaks` and `bandit` run in CI, so keep tokens in environment variables and verify local paths before hardcoding robot- or dataset-specific defaults.
