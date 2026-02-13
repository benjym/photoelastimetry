# Release and Docs Maintenance

## Versioning Checklist

1. Update version in `pyproject.toml`.
2. Run full tests locally.
3. Build docs with strict link checking.
4. Merge and tag release according to repository process.

## Docs QA Checklist

```bash
mkdocs build --strict
```

Confirm:

- no broken links
- no stale command/config references
- new or changed modules reflected in reference nav/pages

## Keep Docs in Sync With Code

- CLI changes in `photoelastimetry/main.py` require updates in:
  - `docs/user/workflows/*`
  - `docs/user/configuration.md`
- New modules require:
  - a reference page under `docs/reference/`
  - nav update in `mkdocs.yml`
- Behavioral changes should add/update tests under `tests/` and corresponding docs notes.
