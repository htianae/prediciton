from pathlib import Path
import unittest

import pandas as pd


ROOT = Path(__file__).parents[1]


class ContractTests(unittest.TestCase):
    def test_irregular_event_contract(self):
        columns = set(pd.read_csv(ROOT / "contracts/irregular_realtime_event_schema.csv").columns)
        required = {"batch_id", "event_time", "measured_at", "entered_at", "elapsed_hours", "cumulative_gas", "is_missing_update", "correction_version"}
        self.assertTrue(required.issubset(columns))

    def test_intervention_contract(self):
        columns = set(pd.read_csv(ROOT / "contracts/intervention_trial_schema.csv").columns)
        required = {"batch_id", "recommended_value", "actual_value", "adjustment_reason", "operator_id", "quality_pass", "final_temperature", "production_weight", "safety_incident", "final_gas"}
        self.assertTrue(required.issubset(columns))


if __name__ == "__main__":
    unittest.main()
