# Sensor-budget experiment

The main notebook remains the original 128-sensor showcase. This separate experiment
reuses its measured-data loading, preprocessing, four models, training, validation
selection, and frozen wave 3 evaluation.

## Protocol

- Budgets: 16, 32, 64, and 128 virtual sensor locations.
- Seeds: 42, 43, and 44; 1,500 MLP optimizer steps per run.
- For each seed, a permutation selects a 128-location candidate pool. Lower budgets
  use prefixes of that same permutation, making sensor sets nested.
- The remaining 1,151 locations form an identical validation/test spatial holdout
  across all four budgets within that seed. Unused candidate sensors at smaller
  budgets are excluded from evaluation.
- The same 73 training, 24 validation, and 23 test frames are used in every run.
- Feature/target scaling uses training observations only. MLP checkpoints and the
  preferred model use validation RMSE, never test or transfer RMSE.
- Frozen transfer uses common coordinates and overlapping times in wave 3 without
  fitting to its observations. It is a later release in the same geometry with an
  evolved bed, not an independent geographic flood.

Sensor placement and neural initialization vary together across seeds. The plotted
error bars are sample standard deviations across three paired runs, not confidence
intervals or a measure of variability across independent flood events. This experiment
does not repeat the synthetic PINN track: the original depth/discharge comparison
remains a single-seed result.

## Reproduce

```bash
floods sensor-ablation --sensors 16 32 64 128 --seeds 42 43 44 --steps 1500
```

Each run saves its executed measured-track notebook, sensor/holdout indices, MLP
history, validation/test/transfer tables, and selected model under
`outputs/sensor_ablation/`. Public summary CSVs, a plot, and source fingerprints
are exported to `reports/sensor_ablation/`.

For a quick pipeline check or a different protocol, use separate output directories:

```bash
floods sensor-ablation --sensors 16 32 --seeds 42 --steps 10 \
  --output outputs/ablation_smoke --reports outputs/ablation_smoke_report
```

The permutation-based paired split differs from the showcase notebook's random
sampling. A 128-sensor ablation score therefore need not match the original notebook
score. Compare budgets within this experiment rather than combining the two designs.
