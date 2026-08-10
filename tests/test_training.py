from pathlib import Path
import tempfile
import unittest

import pandas as pd

from furnace_champion.training import compute_safety_summary, train_and_select


EXCEL = Path(__file__).parents[1] / "4_month_data_2026_02_01_2026_06_25.xlsx"


class TrainingTests(unittest.TestCase):
    def test_safety_gate_rejects_low_feasibility_boundary_and_uncertainty(self):
        safe = pd.DataFrame(
            {
                "model": ["Linear"] * 10,
                "scenario_has_feasible": [True] * 10,
                "boundary_fraction": [0.1] * 10,
                "prediction_std": [100.0] * 10,
            }
        )
        unsafe = pd.DataFrame(
            {
                "model": ["LightGBM"] * 10,
                "scenario_has_feasible": [False] * 2 + [True] * 8,
                "boundary_fraction": [0.5] * 10,
                "prediction_std": [1000.0] * 10,
            }
        )
        result = compute_safety_summary(pd.concat([safe, unsafe]), target_iqr=1000)
        flags = result.set_index("model")["passes_safety"].to_dict()
        self.assertTrue(flags["Linear"])
        self.assertFalse(flags["LightGBM"])

    def test_small_end_to_end_training_compares_three_and_saves_one(self):
        with tempfile.TemporaryDirectory() as directory:
            result = train_and_select(
                EXCEL,
                directory,
                evaluation_budget=100,
                seeds=[0],
                weight_quantiles=[0.5],
            )
            self.assertEqual(set(result["model_summary"]["model"]), {"LightGBM", "Linear", "Huber"})
            self.assertEqual(set(result["safety_summary"]["model"]), {"LightGBM", "Linear", "Huber"})
            self.assertTrue((Path(directory) / "artifacts" / "gas_champion.joblib").exists())
            self.assertIn(result["champion_model"], {"LightGBM", "Linear", "Huber"})
            self.assertIn(result["optimizer_name"], {"random_search", "genetic_algorithm"})


if __name__ == "__main__":
    unittest.main()
