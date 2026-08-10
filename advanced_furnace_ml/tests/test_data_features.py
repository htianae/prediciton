from pathlib import Path
import unittest

import numpy as np

from advanced_furnace_ml.data import (
    FEATURE_COLS,
    TARGET_COL,
    chronological_dev_lock_split,
    load_batch_data,
    mark_target_outliers,
)
from advanced_furnace_ml.features import (
    FurnaceFeatureEngineer,
    prediction_to_total_gas,
    target_for_route,
)


EXCEL = Path(__file__).parents[2] / "4_month_data_2026_02_01_2026_06_25.xlsx"


class DataFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = mark_target_outliers(load_batch_data(EXCEL))

    def test_contract_split_and_outliers(self):
        self.assertEqual(len(self.df), 292)
        self.assertFalse(self.df["batch_id"].duplicated().any())
        self.assertEqual(int(self.df["is_high_gas_outlier"].sum()), 5)
        dev, locked = chronological_dev_lock_split(self.df)
        self.assertEqual((len(dev), len(locked)), (248, 44))
        self.assertEqual(dev.iloc[-1]["batch_id"], self.df.iloc[247]["batch_id"])
        self.assertEqual(locked.iloc[0]["batch_id"], self.df.iloc[248]["batch_id"])

    def test_feature_engineering_and_unit_route(self):
        engineer = FurnaceFeatureEngineer()
        transformed = engineer.fit_transform(self.df[FEATURE_COLS])
        self.assertEqual(list(transformed.columns), engineer.get_feature_names_out().tolist())
        derived = [column for column in transformed if column not in FEATURE_COLS]
        self.assertEqual(len(derived), 6)
        self.assertTrue(np.isfinite(transformed[derived].dropna().to_numpy()).all())
        unit_target = target_for_route(self.df, "unit")
        restored = prediction_to_total_gas(unit_target.to_numpy(), self.df[FEATURE_COLS], "unit")
        np.testing.assert_allclose(restored, self.df[TARGET_COL].to_numpy())


if __name__ == "__main__":
    unittest.main()
