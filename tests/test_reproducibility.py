"""Check the published data and the paired experiment's evaluation design."""

import hashlib
import json
from pathlib import Path
import unittest
import tempfile

import nbformat
import numpy as np

from floods.ablation import build_run, paired_sensor_split, run_ablation


ROOT = Path(__file__).resolve().parents[1]


class ReproducibilityChecks(unittest.TestCase):
    def test_published_data_checksums(self):
        directory = ROOT / "reports/artifacts"
        manifest = json.loads((directory / "manifest.json").read_text())
        for name, entry in manifest.items():
            self.assertEqual(hashlib.sha256((directory / name).read_bytes()).hexdigest(), entry["sha256"])

    def test_nested_sensors_and_fixed_unseen_locations(self):
        reference = None
        previous = set()
        for budget in (16, 32, 64, 128):
            sensors, heldout = paired_sensor_split(1279, budget, 128, 42)
            self.assertEqual(len(sensors), budget)
            self.assertEqual(len(heldout), 1151)
            self.assertTrue(previous.issubset(set(sensors)))
            self.assertFalse(set(sensors).intersection(heldout))
            if reference is not None:
                np.testing.assert_array_equal(heldout, reference)
            reference, previous = heldout, set(sensors)

    def test_ablation_reuses_training_and_excludes_synthetic_track(self):
        source = ROOT / "notebooks/flood_reconstruction_model_comparison.ipynb"
        original = nbformat.read(source, as_version=4)
        run = build_run(source, 16, 43, 10, 128, ROOT / "outputs/test_adapter")
        cells = [c.source for c in run.cells]
        training = next(c.source for c in original.cells if c.cell_type == "code"
                        and c.source.startswith("optimizer = torch.optim.Adam(mlp.parameters()"))
        self.assertIn(training, cells)
        self.assertTrue(any("paired_sensor_split(len(xy)" in c for c in cells))
        self.assertFalse(any("class ShallowWaterNN" in c for c in cells))

    def test_different_protocol_cannot_mix_with_existing_results(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "runs"
            output.mkdir()
            (output / "protocol.json").write_text(json.dumps({"mlp_steps": 10}))
            with self.assertRaisesRegex(ValueError, "different protocol"):
                run_ablation(ROOT / "notebooks/flood_reconstruction_model_comparison.ipynb",
                             output, Path(directory) / "reports", [16, 32], [42], 20)


if __name__ == "__main__":
    unittest.main()
