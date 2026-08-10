from pathlib import Path
import unittest

from furnace_champion.data import FEATURE_COLS, TARGET_COL, load_batch_data, mark_target_outliers


EXCEL = Path(__file__).parents[1] / "4_month_data_2026_02_01_2026_06_25.xlsx"


class DataContractTests(unittest.TestCase):
    def test_loads_expected_batch_contract(self):
        df = load_batch_data(EXCEL)

        self.assertEqual(len(df), 292)
        self.assertEqual(df["batch_id"].nunique(), 292)
        self.assertEqual(list(df.columns[:2]), ["batch_id", FEATURE_COLS[0]])
        self.assertTrue(all(column in df.columns for column in FEATURE_COLS + [TARGET_COL]))
        self.assertEqual(df["10#熔炼炉总投料重量(kg)"].isna().sum(), 14)

    def test_marks_but_keeps_target_outliers(self):
        df = mark_target_outliers(load_batch_data(EXCEL))

        self.assertEqual(len(df), 292)
        self.assertEqual(int(df["is_high_gas_outlier"].sum()), 5)
        self.assertEqual(
            set(df.loc[df["is_high_gas_outlier"], "batch_id"]),
            {"ER032", "ER140", "ER141", "ER196", "ER305"},
        )


if __name__ == "__main__":
    unittest.main()
