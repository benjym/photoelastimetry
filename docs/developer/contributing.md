# Contributing

## Workflow

1. Create a feature branch.
2. Make scoped changes with tests.
3. Run formatters/tests locally.
4. Open a PR with a clear summary and validation notes.

## Recommended Local Checks

```bash
black photoelastimetry tests
pytest
mkdocs build --strict
```

## PR Expectations

- Explain what changed and why.
- Include test evidence (command + result summary).
- Update docs for behavior/config/interface changes.
- Keep changes focused; avoid unrelated refactors in the same PR.

## Areas to Treat Carefully

- Parameter alias/precedence behavior in `main.py`
- Data shape conventions across IO, calibration, seeding, and solver modules
- Legacy compatibility keys that are still intentionally supported
