# Dashboard experiment artifacts

These are verified predictions from the main Kaggle showcase, exported from its saved
checkpoints by `scripts/export_dashboard.py`. They are separate from earlier CLI runs.

| File | Contents |
| --- | --- |
| `wave1.npz` | Actual measured WSE and four model predictions at 23 held-out timestamps and 1,151 held-out spatial locations; wave 1 sensor positions |
| `wave3.npz` | Actual wave 3 WSE and frozen wave 1 model predictions at common coordinates and overlapping times; no wave 3 fitting observations |
| `synthetic.npz` | Numerical reference, two neural predictions at eight held-out times, held-out spatial IDs, and matched seeded PDE diagnostic points/residuals |
| `manifest.json` | Checkpoints and notebook checksums, output checksums, run configuration, and numerical reference diagnostics |

Measured coordinates are in meters, time in seconds, and `reference_m`/model fields in
meters. The dashboard converts WSE and errors to millimeters. WSE is relative to the
initial bed, not measured depth. Synthetic quantities are dimensionless.

Measured observations: Segovia-Burillo, Martínez-Aranda, Morales-Hernández,
Fernández-Pato, and García-Navarro, [Zaragoza release 17777387](https://zenodo.org/records/17777387),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Derived measurements and
predictions retain that attribution and license. Processing includes unit conversion,
masking, subsampling, splitting, model reconstruction, and same-geometry wave transfer.
See [full provenance](../../docs/data_sources.md).

These files reproduce the notebook scores; they are not additional independent
experiments or observations from Nepal. The sensor page reads the separate paired
experiment's CSV results under `reports/sensor_ablation/`.
