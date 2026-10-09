# Reproducing the notebook and experiments

## Verified environment

The complete environment is pinned in `requirements-notebook-lock.txt` (also loaded
by `requirements.txt`). It was installed in a new Python 3.13.3 virtual environment
on macOS arm64. The project was extracted from the shareable ZIP into a separate
directory without the original workspace's raw data, prepared arrays, or checkpoints.
All 27 showcase code cells executed with zero errors, producing 10 figures. Fresh
comparison metrics matched the original run within floating-point tolerance.
Machine-readable evidence is in `reports/reproducibility.json`.

The lock reflects the tested macOS environment. For other supported Python 3.11+
platforms, `python -m pip install -e '.[notebook]'` resolves compatible versions,
but that environment has not been verified here. CPU training uses two PyTorch
threads. Exact scores can vary with library versions, hardware, and random seeds.

## Offline measured-data route

`reports/artifacts/O4_wave1.npz` and `O4_wave3.npz` contain actual processed laboratory
measurements, with dataset attribution and SHA-256 checksums. They are included in
the repository, so notebook execution does not need a data download or field visit.
The PINN reference is generated inside the notebook by a numerical solver.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -m unittest discover -s tests -v
floods run-notebook
floods sensor-ablation
```

`run-notebook` uses the current Python interpreter for its Jupyter kernel, eliminating
dependence on a separately registered global kernel. It writes an executed copy and
HTML preview to `outputs/notebooks/`; the submitted showcase notebook remains intact.
It also saves models/results to `outputs/kaggle_notebook/`.

## Original raw-data route

```bash
floods download --experiment O4
floods audit --experiment O4
floods prepare --experiment O4 --wave 1
floods prepare --experiment O4 --wave 3
floods run-notebook --data-source raw --output outputs/notebooks/raw_verified.ipynb
```

The downloader verifies published checksums and rejects unsafe archive paths.
The raw notebook path applies the mask, checks grids, preserves measured timestamps,
converts millimeters to meters, and spatially subsamples at stride 8. It was exercised
in the original workspace; the fresh ZIP verification used the processed-data route.

## Checks and automation

Tests check source checksums, unit conversion, masking, timestamps, disjoint holdouts,
nested sensor budgets, interpolation, numerical mass balance, and differentiable
physics residuals. The GitHub Actions workflow installs the pinned environment,
runs the tests, executes the showcase, and runs a short two-budget ablation on macOS.
The macOS runner label follows the [GitHub-hosted runners reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).
Current hosted results are available in
[GitHub Actions](https://github.com/dols30/Physics-Informed-Flood-Reconstruction/actions/workflows/checks.yml).

The sensor study includes three seeds for measured-data models. The synthetic PINN
result is still a single-seed benchmark; the measured sensor study does not establish
PINN robustness across seeds or across real disasters.

## Shareable ZIP

```bash
python scripts/package_project.py
```

The ZIP includes code, tests, both notebooks, processed observations, result summaries,
figures, attribution, and dependency files. It excludes virtual environments, raw
archives, private Git state, Jupyter scratch documents, and local model checkpoints.
The complete project is published at
[dols30/Physics-Informed-Flood-Reconstruction](https://github.com/dols30/Physics-Informed-Flood-Reconstruction).
Clone that repository or use the ZIP as a standalone copy. Repository description:

> Comparative machine learning and physics-informed neural networks for floodwater
> reconstruction using laboratory observations and simulated shallow-water dynamics.
