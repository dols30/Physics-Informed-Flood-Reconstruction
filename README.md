# Physics-Informed Learning for Flood Reconstruction

**[View on Kaggle](https://www.kaggle.com/code/bashcode223/physics-informed-flood-reconstruction-ml-vs-pinn)** · [Main notebook](notebooks/flood_reconstruction_model_comparison.ipynb) · [Sensor-budget study](notebooks/sensor_ablation.ipynb)

**Author:** Dol Raj Bashyal · [Kaggle: bashcode223](https://www.kaggle.com/bashcode223)

A notebook-first machine learning project comparing floodwater reconstruction from
sparse laboratory observations with a physics-informed shallow-water simulation.

I am from Nepal. Seeing the effects of Himalayan glacial lake outburst floods motivated
me to explore how machine learning could help us understand floodwater movement,
building on my [Nepal rainfall prediction project](https://www.kaggle.com/code/bashcode223/rainfall-prediction-nepal-rf-ann-xgb).
The experiments here use published laboratory data from Zaragoza, Spain, and a
controlled numerical simulation.

**Start with the [model-comparison notebook](notebooks/flood_reconstruction_model_comparison.ipynb).**
It includes data loading, preprocessing, exploratory plots, train/validation/test
splits, model training, evaluation, a personal introduction, and conclusions.
The submitted notebook is preserved unchanged, including its executed outputs.
Its 27 code cells match the published Kaggle notebook; the Kaggle version adds an
author line and a short opening description.

## What the project demonstrates

- Reconstructing water-surface elevation (WSE) with interpolation, polynomial ridge
  regression, Random Forest, and a coordinate neural network (MLP).
- Evaluating frozen models on a later flood wave without fitting to that wave.
- Comparing a data-only network and a PINN using shallow-water mass and momentum
  equations, with matched training observations and initial weights.
- Measuring the effect of 16/32/64/128 virtual sensors across three seeds using nested
  sensor sets and fixed evaluation locations within each seed.

## Results

Original showcase: 128 virtual sensor locations, seed 42, 1,500 MLP optimizer steps.
The validation set selects models and checkpoints; test data is reserved for evaluation.

| Measured model | Wave 1 test RMSE (mm) | Frozen wave 3 RMSE (mm) |
| --- | ---: | ---: |
| Interpolation | **3.44** | 6.80 |
| Random Forest | 3.81 | 6.86 |
| MLP | 4.73 | **6.72** |
| Polynomial Ridge | 5.99 | 7.01 |

The measured target is WSE relative to the initial bed, not a measured water-depth or
velocity field. Wave 3 shares the laboratory geometry but starts with an evolved bed.

| Simulated model | Depth RMSE | Discharge RMSE | Continuity RMS | Momentum RMS |
| --- | ---: | ---: | ---: | ---: |
| Data-only NN | **0.01001** | 0.10734 | 0.97741 | 0.68763 |
| PINN | 0.01136 | **0.01610** | **0.16597** | **0.19703** |

All simulated quantities are dimensionless. The PINN reduced discharge error by
approximately **85%** in this single-seed simulation, while depth error was slightly
higher. Both networks received initial/boundary depth and discharge values; interior
training observations contained depth only.

![Shallow-water neural and PINN comparison](reports/figures/synthetic_pinn.png)

### Sensor-budget study

Separate paired experiment: seeds 42, 43, 44; 1,500 MLP steps per run. Values below are
wave 1 test RMSE mean ± sample standard deviation in millimeters.

| Sensors | Interpolation | Random Forest | MLP | Polynomial Ridge |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 5.52 ± 0.12 | 5.60 ± 0.11 | **5.45 ± 0.12** | 7.46 ± 1.55 |
| 32 | **4.70 ± 0.22** | 5.02 ± 0.17 | 5.24 ± 0.18 | 6.16 ± 0.01 |
| 64 | **3.76 ± 0.35** | 4.51 ± 0.07 | 4.88 ± 0.04 | 5.95 ± 0.08 |
| 128 | **3.16 ± 0.13** | 3.81 ± 0.08 | 4.82 ± 0.12 | 5.91 ± 0.05 |

![Sensor-budget reconstruction and transfer study](reports/sensor_ablation/sensor_ablation.png)

The MLP's mean advantage at 16 sensors is small; three seeds do not establish
statistical superiority. Interpolation improves most as sensors are added, while
wave 3 transfer errors remain higher. This study uses a different paired sensor split
from the showcase, so its 128-sensor score is a separate result.
See the [experiment notebook](notebooks/sensor_ablation.ipynb),
[protocol](docs/sensor_ablation.md), and [full CSV results](reports/sensor_ablation/runs.csv).

## Run locally

The complete workflow was verified in a fresh **Python 3.13.3 / macOS arm64** environment
from an isolated ZIP copy, using only the included processed measurements. All 27
showcase code cells ran successfully, and the fresh metrics matched the original run.
No GPU or field data collection is required.

```bash
git clone https://github.com/dols30/Physics-Informed-Flood-Reconstruction.git
cd Physics-Informed-Flood-Reconstruction
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -m unittest discover -s tests -v

# Open the main notebook; Run All trains the models
python -m jupyterlab notebooks/flood_reconstruction_model_comparison.ipynb

# Or execute it and save an independent notebook/HTML copy
floods run-notebook

# Repeat all 12 measured-data sensor experiments
floods sensor-ablation --sensors 16 32 64 128 --seeds 42 43 44 --steps 1500
```

The full notebook environment is pinned in `requirements-notebook-lock.txt`.
`requirements-lock.txt` remains the smaller environment for the original CLI benchmarks.
For another Python 3.11+ platform, install `python -m pip install -e '.[notebook]'`;
that environment may resolve different package versions and is not verified here.
See [reproducibility instructions](docs/reproducibility.md) and
[verification evidence](reports/reproducibility.json).

### Data and Kaggle

The published [Kaggle notebook](https://www.kaggle.com/code/bashcode223/physics-informed-flood-reconstruction-ml-vs-pinn)
provides the main model-comparison presentation. GitHub contains the complete
reproducible project, including the separate three-seed sensor study.

The repository includes checksum-verified, processed O4 wave 1 and wave 3 observations
in `reports/artifacts/`, retaining their **CC BY 4.0** attribution. The original raw
archive is available through `floods download --experiment O4`; raw preparation and
execution commands are documented in the reproducibility guide.

For Kaggle, attach `O4_wave1.npz` and `O4_wave3.npz` as an input dataset and import the
main notebook. It discovers these files under `/kaggle/input/`. Install missing
notebook dependencies before running; the verified local environment is recorded
in the lock file. Models and result files are saved under `outputs/kaggle_notebook/`.

## Repository layout

| Path | Purpose |
| --- | --- |
| `notebooks/flood_reconstruction_model_comparison.ipynb` | Original executed showcase and model training |
| `notebooks/sensor_ablation.ipynb` | Paired experiment results and comparison plots |
| `src/floods/` | Data preparation, numerical physics, experiment adapters, and CLI |
| `reports/notebook_results/` | CSV metrics reproduced in the clean environment |
| `reports/sensor_ablation/` | Twelve-run results, mean/std summaries, plot, and fingerprints |
| `reports/artifacts/` | Compact attributed observations and earlier CLI artifacts |
| `tests/` | Scientific data contracts and experiment-design checks |
| `.github/workflows/checks.yml` | Automated tests, notebook execution, and ablation smoke run |
| `docs/` | Provenance, reproducibility, experiment protocol, and research directions |
| `scripts/package_project.py` | Shareable ZIP with local environments/checkpoints excluded |

## Scope and next research steps

This is a reconstruction benchmark, not an operational Nepal flood forecast or a new
scientific method. Its two measured waves come from one laboratory configuration.
Three measured-data seeds quantify sensitivity to sensor placement and initialization,
not uncertainty across independent flood events. The simulated PINN assumes a flat,
fixed, frictionless wet bed; the laboratory experiment has bed evolution and lacks
measured velocity, so the notebook evaluates the two tracks separately.

Next steps are multi-seed PINN runs, additional measured experiments, physics-mismatch
and shock-resolution studies, then terrain and event observations from the Himalayas.
See the [research plan](docs/research_plan.md).

## Attribution and license

Experimental observations: Segovia-Burillo, Martínez-Aranda, Morales-Hernández,
Fernández-Pato, and García-Navarro,
[Zaragoza dataset, Zenodo release 17777387](https://zenodo.org/records/17777387),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Changes include unit conversion, masking, spatial subsampling, sparse-observation
experiments, and derived reconstruction figures. Full details are in
[data provenance](docs/data_sources.md).

Project code and documentation use the [Apache License 2.0](LICENSE), matching the
published Kaggle notebook. Third-party observations
and their derivatives retain CC BY 4.0. Simulation references are generated by the
project's numerical solver.
