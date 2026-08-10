# Summary

<!-- What does this change and why? Link the issue it closes. -->

Closes #

## Type of change

- [ ] Bug fix (no API change)
- [ ] New feature (no API break)
- [ ] Breaking change
- [ ] Documentation only
- [ ] Build, CI or tooling

## Scientific correctness

Complete this section for any change to calibration, parameters of fit,
truth-table coding, or Boolean minimisation. Delete it otherwise.

- [ ] The change includes a mathematical description of what it computes.
- [ ] Unit tests cover the new behaviour.
- [ ] At least one edge case is tested.
- [ ] Where an equivalent R `QCA` feature exists, a parity fixture is included,
      or the divergence is documented with a stated reason.
- [ ] No heuristic minimisation has been introduced behind an API documented as exact.

## Checklist

- [ ] `poetry run ruff check .` passes
- [ ] `poetry run ruff format --check .` passes
- [ ] `poetry run mypy` passes
- [ ] `poetry run pytest --cov=setqca` passes and meets the coverage gate
- [ ] Public API changes are reflected in the docstrings and the docs site
- [ ] `CHANGELOG.md` has an entry under `Unreleased`
