# Data provenance

## Active measured dataset

**Title:** Experimental characterization of bed evolution and water surface elevation by
consecutive dambreaks using Kinect device.

**Creators:** Jose Segovia-Burillo, Sergio Martínez-Aranda, Mario Morales-Hernández,
Javier Fernández-Pato and Pilar García-Navarro.

**Pinned release:** [Zenodo 17777387](https://zenodo.org/records/17777387), version 2,
published 3 December 2025; DOI [10.5281/zenodo.17777387](https://doi.org/10.5281/zenodo.17777387).

**Associated paper:** [10.1080/00221686.2025.2602056](https://doi.org/10.1080/00221686.2025.2602056),
Journal of Hydraulic Research, 64(2), 133–152.

**License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) for data, confirmed
in the record API and source README. The paper is a separate artifact with its own license.
The paper is not redistributed in this repository.

**Downloaded file:** `O4.zip`, 158,070,327 bytes,
MD5 `3d8a769793402697c1b9fd081a4f1da8`, verified before extraction.
The source README and requirements file are also downloaded and checksum-verified.
The full source record is retained locally at
`data/raw/zaragoza-17777387/record.json`.

**Inspected contents:** 245 HDF5 files: initial offset, sand0/sand1/sand2/sand3,
120 water1 frames and 120 water3 frames. No water2 sequence is in this release's O4 archive.

**Grid:** 293 by 679 in native files. Coordinate and elevation datasets use millimeters;
timestamps use the HDF5 attribute `time_stamp[s]`. The initial reference contains a 0/1
mask. Water files contain `wse[mm]`, `gX`, and `gY`.

**What we change:** Convert millimeters to meters, apply the reference mask and finite-data
intersection, subsample the spatial grid, simulate sparse observation availability,
fit reconstruction baselines, and generate aggregate metrics/plots. We retain measured
timestamps and never treat missing pixels as zero elevation.

**What this does not contain:** Continuous during-wave bed measurements or measured velocity
fields. WSE is relative to the initial bed, not a complete evolving-bed depth record.

## Other published sources for later audits

| Source | Status in this repository |
| --- | --- |
| [Large-scale dam-breach experiments](https://zenodo.org/records/17828278) | Candidate for a second measured benchmark; not downloaded or modeled by this implementation. |
| [South Lhonak terrain products](https://zenodo.org/records/13124662) | Candidate Himalayan geometry; not flood time-series labels and not used in training. |

We have not downloaded or used a Nepal flood dataset in the implemented experiments.
Do not describe the numerical reference, virtual sensors or network outputs as field observations.
