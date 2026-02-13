# Developer Setup

## Clone and Install

```bash
git clone https://github.com/benjym/photoelastimetry.git
cd photoelastimetry
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,docs]"
pre-commit install
```

## Verify Core Tooling

```bash
pytest --version
mkdocs --version
black --version
```

## Useful Entry Points

- CLI orchestration: `photoelastimetry/main.py`
- Calibration pipeline: `photoelastimetry/calibrate.py`
- Solver: `photoelastimetry/optimise.py`
- Data IO: `photoelastimetry/io.py`

## Build Docs Locally

```bash
mkdocs serve
```

Local docs URL: `http://127.0.0.1:8000`
