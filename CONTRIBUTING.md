# Contributing to Photoelastimetry

Canonical contributor documentation lives in the developer docs:

- Setup: `docs/developer/setup.md`
- Testing: `docs/developer/testing.md`
- Contributing workflow: `docs/developer/contributing.md`
- Architecture context: `docs/developer/architecture.md`
- Docs/release maintenance: `docs/developer/release-and-docs.md`

## Quick Local Validation

```bash
pip install -e ".[dev,docs]"
black photoelastimetry tests
pytest
mkdocs build --strict
```

## Pull Requests

- Keep PR scope focused.
- Include tests for behavior changes.
- Update docs when interfaces, workflows, or config keys change.
