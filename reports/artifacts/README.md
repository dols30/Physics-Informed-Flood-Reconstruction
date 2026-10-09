# Compact notebook data

These arrays let the showcase notebook run offline from the repository without downloading
the 158 MB original archive or depending on ignored local checkpoints. They are actual
processed observations and executed predictions, not invented sample data.

- `O4_wave1.npz`, `O4_wave3.npz`: measured WSE after unit conversion, masking and spatial
  subsampling. Coordinates and elevations are meters, times are seconds. The original
  observations are from Segovia-Burillo, Martínez-Aranda, Morales-Hernández, Fernández-Pato
  and García-Navarro, [Zenodo release 17777387](https://zenodo.org/records/17777387),
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). These are modified data;
  see [provenance](../../docs/data_sources.md) for processing and attribution.
- `measured_predictions.npz`: measured WSE and model predictions at the 23 held-out wave 1
  timestamps. Values use meters, not millimeters. Source-derived observations retain the
  above attribution and license.
- `synthetic_reference.npz`, `synthetic_predictions.npz`: numerical reference states and
  executed neural predictions for the controlled shallow-water case. All quantities
  are dimensionless. These files contain no laboratory or Nepal field measurements.

`manifest.json` records each local source, file size and SHA-256. Regenerate the artifacts
after running the CLI benchmarks with `python scripts/export_notebook_data.py`. The notebook
verifies the measured-array checksums, then trains and evaluates models directly. Cached
prediction arrays support the earlier CLI report and are not used to train the notebook models.

The cached figures and scores correspond to the seed 42, 1,500-step initial runs. They are
provided for reproducibility and presentation, not as additional independent test cases.
