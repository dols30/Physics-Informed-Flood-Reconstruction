"""Publish small, attributable artifacts from completed local runs."""

import json
from pathlib import Path
import shutil


def summarize(measured: Path, synthetic: Path, transfer: Path, destination: Path) -> Path:
    measured_result = json.loads((measured / "metrics.json").read_text())
    synthetic_result = json.loads((synthetic / "metrics.json").read_text())
    transfer_result = json.loads((transfer / "metrics.json").read_text())
    destination.mkdir(parents=True, exist_ok=True)
    figures = destination / "figures"
    figures.mkdir(exist_ok=True)
    for name, source in [("measured_wave1", measured), ("synthetic_benchmark", synthetic), ("wave_transfer", transfer)]:
        shutil.copyfile(source / "metrics.json", destination / f"{name}.json")
    shutil.copyfile(measured / "reconstruction.png", figures / "measured_reconstruction.png")
    shutil.copyfile(synthetic / "pinn_comparison.png", figures / "synthetic_pinn.png")
    mlp = measured_result["supervised_mlp"]
    linear = measured_result["linear_interpolation"]
    data_only = synthetic_result["results"]["data_only"]
    pinn = synthetic_result["results"]["pinn"]
    transferred = transfer_result["frozen_source_mlp"]
    content = f"""# Initial benchmark results

These are executed results, not projected performance. The measured run used seed
{measured_result['seed']} and {measured_result['epochs']} optimizer steps; the synthetic ablation
used seed {synthetic_result['seed']} and {synthetic_result['epochs']} steps per model.
One seed does not establish research superiority.

## Measured O4 wave 1

{measured_result['sensors']} sampled grid locations serve as virtual sensors. Their WSE values are used
at {measured_result['training_frames']} training frames; {measured_result['validation_frames']} frames are
reserved for model selection. Final evaluation uses {measured_result['test_frames']} other frames
at {measured_result['heldout_spatial_points']} spatial locations that were not sensors.
The task is within-wave reconstruction, not forecasting a new disaster.

| Model | WSE RMSE (mm) | WSE MAE (mm) |
| --- | ---: | ---: |
| Linear spatial and temporal interpolation | {linear['rmse_mm']:.3f} | {linear['mae_mm']:.3f} |
| Supervised coordinate MLP | {mlp['rmse_mm']:.3f} | {mlp['mae_mm']:.3f} |

The interpolation baseline has lower error in this run. The measured-data model currently has no
PDE loss: evolving bed elevation and velocity have not been observed during the wave.

![Measured WSE reconstruction](figures/measured_reconstruction.png)

Source: Segovia-Burillo et al., [Zaragoza dataset](https://zenodo.org/records/17777387),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes: unit conversion,
masking, spatial subsampling, simulated observation removal, and reconstruction plots.

## Frozen wave 1 to wave 3 transfer

The models receive no wave 3 observations for fitting. Evaluation matches spatial coordinates
and restricts time to the wave 1 training interval: {transfer_result['evaluated_frames']} frames,
{transfer_result['evaluated_points']} locations. The frozen MLP RMSE is
{transferred['rmse_mm']:.3f} mm; frozen source interpolation RMSE is
{transfer_result['frozen_source_interpolation']['rmse_mm']:.3f} mm.
This is a test across waves in the same laboratory geometry with a changed bed, not across
independent Himalayan events. Higher error suggests that within-wave accuracy does not
establish transfer performance.

## Simulated shallow-water PINN ablation

Both networks receive the same simulated height observations at 12 virtual sensor locations,
plus the same known initial and boundary height/discharge conditions. Initial weights and
observation minibatches match. The PINN additionally receives continuity and momentum losses.
Checkpoints are selected using validation height error; test interior values are held out.

| Model | Height RMSE (dimensionless) | Discharge RMSE (dimensionless) | Continuity residual RMS | Momentum residual RMS |
| --- | ---: | ---: | ---: | ---: |
| Data-only NN | {data_only['heldout_height_rmse_dimensionless']:.5f} | {data_only['heldout_discharge_rmse_dimensionless']:.5f} | {data_only['continuity_residual_rms']:.5f} | {data_only['momentum_residual_rms']:.5f} |
| PINN | {pinn['heldout_height_rmse_dimensionless']:.5f} | {pinn['heldout_discharge_rmse_dimensionless']:.5f} | {pinn['continuity_residual_rms']:.5f} | {pinn['momentum_residual_rms']:.5f} |

In this run, physics improves discharge inference and conservation residuals, while height
reconstruction is slightly worse. Discharge truth is available only because this benchmark
is simulated. Residual reduction alone does not establish predictive accuracy.

![Synthetic PINN comparison](figures/synthetic_pinn.png)

The conservative reference solver's mass-balance error is
{synthetic_result['reference_solver']['mass_balance_error']:.2e} in dimensionless units.
Refining from 128 to 256 cells changes final height by an RMSE of
{synthetic_result['reference_solver']['final_height_rmse_against_256_cells']:.5f}; numerical
reference error is material relative to network error. Strong-form PINNs also smooth shocks.

## Scope of the evidence

The pipeline runs on real laboratory observations and the PINN passes a controlled numerical
benchmark. It has not been validated on Nepal floods, glacier failure, debris flows,
wetting/drying, sediment transport, or a real warning system. Next experiments must examine
multiple seeds, reduced sensor budgets, weak conservation losses, and physics mismatch.
"""
    target = destination / "initial_results.md"
    target.write_text(content)
    return target
