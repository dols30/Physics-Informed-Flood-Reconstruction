"""Serving-time checks: metrics, split semantics, and every interactive research view."""

import unittest

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from app.data import (
    ROOT,
    MODELS,
    frame_table,
    load_archive,
    load_wave,
    measured_results,
    scores,
    sensor_results,
    summarize_sensors,
)


class DashboardScience(unittest.TestCase):
    def test_published_measured_scores_are_reproducible_from_displayed_arrays(self):
        for number in (1, 3):
            wave = load_wave(number)
            expected = measured_results(number).set_index("Model")
            for name, key in MODELS.items():
                actual = scores(wave["reference_m"], wave[key])
                for column, value in actual.items():
                    self.assertAlmostEqual(value, expected.loc[name, column], places=5)
            self.assertTrue((np.diff(wave["time_s"]) > 0).all())
            self.assertEqual(len(wave["sensor_xy_m"]), 128 if number == 1 else 0)

    def test_synthetic_metrics_and_residuals_match_the_same_saved_models(self):
        data = load_archive("synthetic.npz")
        expected = pd.read_csv(ROOT / "reports/notebook_results/synthetic_results.csv").set_index(
            "Model"
        )
        for name, key in (("Data-only NN", "data_only"), ("PINN", "pinn")):
            heldout = data["heldout_ids"]
            for i, column in enumerate(("Height RMSE", "Discharge RMSE")):
                actual = np.sqrt(
                    np.mean((data[key][:, heldout, i] - data["reference"][:, heldout, i]) ** 2)
                )
                self.assertAlmostEqual(float(actual), expected.loc[name, column], places=6)
            for i, column in enumerate(("Continuity RMS", "Momentum RMS")):
                actual = np.sqrt(np.mean(data[key + "_residual"][:, i] ** 2))
                self.assertAlmostEqual(float(actual), expected.loc[name, column], places=5)

    def test_sample_variation_recomputed_from_all_three_seeds(self):
        runs = sensor_results()
        for split in ("test", "transfer"):
            summary = summarize_sensors(runs[runs.Split == split])
            saved = pd.read_csv(ROOT / "reports/sensor_ablation/summary.csv")
            saved = saved[saved.Split == split].set_index(["Sensors", "Model"])
            for row in summary.itertuples():
                expected = saved.loc[(row.Sensors, row.Model)]
                self.assertEqual(row.count, 3)
                self.assertAlmostEqual(row.mean, expected.rmse_mean_mm, places=10)
                self.assertAlmostEqual(row.std, expected.rmse_std_mm, places=10)

    def test_frame_download_uses_the_displayed_model_time_and_millimeter_units(self):
        wave = load_wave(3)
        for frame in (0, len(wave["time_s"]) - 1):
            table = frame_table(wave, "MLP", frame)
            np.testing.assert_array_equal(table.time_s, wave["time_s"][frame])
            np.testing.assert_allclose(table.predicted_wse_mm, wave["mlp"][frame] * 1000)
            np.testing.assert_allclose(
                table.absolute_error_mm,
                np.abs(wave["mlp"][frame] - wave["reference_m"][frame]) * 1000,
            )


class DashboardInteractions(unittest.TestCase):
    def view(self, name):
        app = AppTest.from_string(
            f"from app.views import {name}\n{name}()", default_timeout=20
        ).run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertFalse(app.error, [e.value for e in app.error])
        return app

    def assert_clean(self, app):
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertFalse(app.error, [e.value for e in app.error])

    def test_entrypoint_and_all_views_render(self):
        app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=20).run()
        self.assert_clean(app)
        for name in ("comparison", "ablation", "pinn"):
            with self.subTest(view=name):
                self.view(name)

    def test_explorer_switches_models_waves_endpoints_3d_and_animation(self):
        app = self.view("explorer")
        for number in (1, 3):
            label = "Wave 1 · held-out test" if number == 1 else "Wave 3 · frozen transfer"
            app.selectbox[0].set_value(label).run()
            for model in MODELS:
                app.selectbox[1].set_value(model).run()
                self.assert_clean(app)
                expected = measured_results(number).set_index("Model").loc[model, "RMSE (mm)"]
                self.assertEqual(app.metric[2].value, f"{expected:.2f} mm")
            for frame in (1, len(load_wave(number)["time_s"])):
                app.slider[0].set_value(frame).run()
                self.assert_clean(app)
        app.segmented_control[0].set_value("3D surface").run()
        self.assert_clean(app)
        app.checkbox(key="animate_wave").check().run()
        self.assert_clean(app)

    def test_sensor_budgets_and_transfer_use_the_requested_results(self):
        app = self.view("ablation")
        for label in ("Wave 1 · held-out test", "Wave 3 · frozen transfer"):
            app.selectbox[0].set_value(label).run()
            split = "test" if label.startswith("Wave 1") else "transfer"
            for budget in (16, 32, 64, 128):
                app.segmented_control[0].set_value(budget).run()
                self.assert_clean(app)
                runs = sensor_results()
                expected = runs[
                    (runs.Split == split)
                    & (runs.Sensors == budget)
                    & (runs.Model == "Interpolation")
                ]["RMSE (mm)"].mean()
                self.assertEqual(app.metric[0].value, f"{expected:.2f} mm")

    def test_pinn_last_frame_and_momentum_residuals(self):
        app = self.view("pinn")
        app.slider[0].set_value(8).run()
        app.segmented_control[0].set_value("Momentum").run()
        self.assert_clean(app)
        self.assertEqual(app.metric[0].value, "85.0%")


if __name__ == "__main__":
    unittest.main()
