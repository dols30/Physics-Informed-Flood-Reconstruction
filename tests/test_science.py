"""Numerical and data-contract checks that protect the research interpretation."""

from pathlib import Path
import tempfile
import unittest
import zipfile

import h5py
import numpy as np
import torch

from floods.data import prepare, timestamp
from floods.download import extract_archive
from floods.measured import split_indices, interpolate_space
from floods.physics import rusanov_flux, shallow_water_residual, solve_reference


class ScienceChecks(unittest.TestCase):
    def test_constant_state_flux(self):
        state = np.tile([2.0, 0.5], (8, 1))
        flux = rusanov_flux(state, state)
        np.testing.assert_allclose(flux, np.tile([0.5, 2.125], (8, 1)))

    def test_reference_positive_and_mass_conservative(self):
        result = solve_reference(cells=64, frames=11)
        self.assertTrue(np.all(result["state"][:, :, 0] > 0))
        self.assertLess(abs(result["diagnostics"]["mass_balance_error"]), 1e-12)

    def test_autograd_equations_on_exact_solution(self):
        class ExactUniformAcceleration(torch.nn.Module):
            def forward(self, xt):
                # h constant, q=t: continuity=0, momentum=1.
                return torch.stack([2 + 0 * xt[:, 0], xt[:, 1]], dim=1)
        points = torch.rand((12, 2), dtype=torch.float64)
        residual = shallow_water_residual(ExactUniformAcceleration(), points)
        np.testing.assert_allclose(residual.detach().numpy(), np.tile([0, 1], (12, 1)), atol=1e-12)

    def test_measurement_time_requires_attribute(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wse_t999.h5"
            with h5py.File(path, "w") as handle:
                handle.create_dataset("wse[mm]", data=np.ones((2, 2)))
            with self.assertRaisesRegex(ValueError, "Missing measured timestamp"):
                timestamp(path)
            with h5py.File(path, "a") as handle:
                handle.attrs["time_stamp[s]"] = 0.8
            self.assertEqual(timestamp(path), 0.8)

    def test_prepare_units_mask_and_missing_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = root / "raw/zaragoza-17777387/extracted/O4"
            experiment.mkdir(parents=True)
            x = np.array([[1000, 2000], [1000, 2000]], dtype=float)
            y = np.array([[0, 0], [100, 100]], dtype=float)
            with h5py.File(experiment / "offset.h5", "w") as handle:
                handle["gX"], handle["gY"] = x, y
                handle["mask"] = [[1, 1], [1, 0]]
            wave = experiment / "water1_h5"
            wave.mkdir()
            for i in range(2):
                with h5py.File(wave / f"frame{i}.h5", "w") as handle:
                    handle["gX"], handle["gY"] = x, y
                    handle["wse[mm]"] = [[10 + i, np.nan], [-5, 999]]
                    handle.attrs["time_stamp[s]"] = 0.8 + i * 0.1
            output = root / "processed.npz"
            result = prepare(root, "O4", 1, 1, output)
            self.assertEqual(result["spatial_points"], 2)
            with np.load(output, allow_pickle=False) as data:
                np.testing.assert_allclose(data["xy"], [[1, 0], [1, 0.1]])
                np.testing.assert_allclose(data["wse_m"], [[0.010, -0.005], [0.011, -0.005]])
                np.testing.assert_allclose(data["time_s"], [0.8, 0.9])

    def test_disjoint_holdouts(self):
        split = split_indices(120, 1279, 128, 42)
        for a, b in [("train_times", "validation_times"), ("train_times", "test_times"),
                     ("validation_times", "test_times"), ("sensor_points", "heldout_points")]:
            self.assertEqual(len(np.intersect1d(split[a], split[b])), 0)
        self.assertIn(0, split["train_times"])
        self.assertIn(119, split["train_times"])

    def test_spatial_interpolation_preserves_plane(self):
        xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.3, 0.4]], dtype=float)
        values = (2 * xy[:4, 0] + 3 * xy[:4, 1])[None, :]
        result, fallback = interpolate_space(xy, np.arange(4), values)
        np.testing.assert_allclose(result[0], 2 * xy[:, 0] + 3 * xy[:, 1])
        self.assertEqual(fallback, 0)

    def test_archive_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../outside.txt", "unsafe")
            with self.assertRaisesRegex(ValueError, "Unsafe archive"):
                extract_archive(archive, Path(directory) / "out")


if __name__ == "__main__":
    unittest.main()
