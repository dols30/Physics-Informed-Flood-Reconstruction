# Research scope and next experiments

## Question

Can shallow-water conservation improve inference of unobserved discharge from sparse
water-level observations, and how does performance change when the assumed physics differs
from the actual flow?

The longer-term application is Himalayan outburst-flood reconstruction. The first evidence
comes from an existing laboratory dataset and a controlled numerical benchmark. No field
visit or new experimental measurements are required for these stages.

## Evidence available now

The [Zaragoza dataset](https://zenodo.org/records/17777387) measures water-surface elevation
(WSE) and bed changes in obstacle configurations. The [paper](https://doi.org/10.1080/00221686.2025.2602056)
describes the coordinate system, time reference, measurement technique, and erodible bed.
The downloaded O4 archive has waves 1 and 3, each with 120 timestamped WSE frames, initial
reference geometry and bed maps after three waves. The second water sequence is absent.

Coordinates and elevations are in millimeters; frame attributes provide seconds relative
to gate opening. The loader converts spatial quantities to meters, uses the reference
mask, checks coordinate consistency, and keeps only locations finite throughout a wave.
Negative WSE values relative to the initial bed are retained: they are not automatically
negative physical water depths. No missing observations are filled by our pipeline.

The bed is only observed before/after releases, not continuously during them. Therefore,
subtracting the final bed from each transient WSE map would invent water-depth targets.
Velocity and discharge fields are not supplied as measured targets in this archive.

## Implemented benchmarks

### Measured WSE reconstruction

Sample 128 of the downsampled spatial locations as virtual sensors. Split complete
timestamps into training, validation and test sets, preserving the temporal endpoints
in training. Fit a coordinate MLP and a spatial/temporal interpolation baseline using
the same observation budget. Select the neural checkpoint by validation WSE RMSE.
Evaluate final errors only at nonsensor locations and withheld test timestamps.

This protocol examines interpolation within one event. Dense source frames are available
for checking the deliberately removed observations; the simulated sensor locations are
not actual instruments installed in the laboratory.

Freeze the wave 1 models and evaluate wave 3 without fitting to its observations. Match
coordinates rather than array positions, and omit timestamps outside the source interval.
This test isolates a change in initial bed state within one obstacle configuration. It is
not a geographically independent event or a Nepal transfer test.

### Controlled shallow-water PINN

Use a flat, fixed, frictionless bed with strictly positive downstream water depth. Generate
a smoothed release using a conservative Rusanov finite-volume solver. In dimensionless
coordinates, with gravity equal to one and discharge per unit width q = h u:

```text
continuity: h_t + q_x = 0
momentum:   q_t + (q^2/h + h^2/2)_x = 0
```

The model predicts positive h and unconstrained q. Both the data-only and PINN models get
the same sparse height observations and known synthetic initial/boundary h,q conditions.
They share initial weights, observation minibatches and checkpoint-selection rules. The
PINN additionally uses the two automatic-differentiation residuals with a fixed weight
of 0.01 and a 200-step warmup. Interior discharge observations are never training targets.

Use separate spatial/time holdouts, report height and discharge errors against the numerical
reference, and report both conservation residuals. Check solver positivity, mass balance,
and sensitivity to doubling the spatial resolution. The reference is an approximation;
the resolution check is not proof of convergence. See the
[Clawpack shallow-water chapter](https://www.clawpack.org/riemann_book/html/Shallow_water.html)
for equations and wave structure and the
[original PINN paper](https://doi.org/10.1016/j.jcp.2018.10.045) for the method's foundation.

These equations are currently used only for the simulated case. Applying them directly to
the erodible-bed O4 data would create a model mismatch that has not yet been quantified.

## Next research stages

1. Lock a protocol for at least five seeds and 16, 32, 64, 128 measured virtual sensors.
   Choose hyperparameters on development cases; reserve a new configuration for testing.
   Add contiguous spatial gaps and withheld time windows to avoid relying on local smoothness.
2. Extend the synthetic reference to bed topography and roughness. Compare strong-form
   residuals with weak or finite-volume conservation losses around advancing fronts.
   Check identifiability of discharge and friction when only elevation is observed.
3. Audit a second Zaragoza configuration and the large-scale dam-breach dataset before
   using new test outcomes to change the model. Assess geometry, velocity type, timestamps,
   discharge derivation, boundary observations and evolving-bed availability separately.
4. Introduce measured bed evolution through a justified approximation or coupled sediment
   model. Any assumption about between-snapshot bed changes must be declared and tested;
   interpolated bed histories are assumptions, not independent observations.
5. Add a Himalayan case only after verifying event geometry, forcing, boundaries and
   independent evaluation observations. Partial event data can support scenario checks,
   but do not establish continuous depth/velocity accuracy or future disaster prediction.

## Claim boundary

Current claim: a reproducible pipeline runs on measured laboratory WSE, and a shallow-water
PINN has been compared with a matched neural baseline on one simulated release.

Not yet supported: superiority across seeds/events, measured discharge reconstruction,
Nepal flood prediction, glacier-failure forecasting, warning-system readiness, or reliable
transfer from laboratory scale to a Himalayan valley. A portfolio can present the actual
methods and negative results; a research paper requires the wider evaluations above.
