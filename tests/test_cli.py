import unittest

from furnace_champion.data import FEATURE_COLS
from predict_champion import build_input_frame
from recommend_champion import create_parser as create_recommend_parser
from train_champion import create_parser as create_train_parser


class CliTests(unittest.TestCase):
    def test_prediction_input_maps_to_exact_model_schema(self):
        frame = build_input_frame(86000, 40, 9.2, 0.8, 13, 75)
        self.assertEqual(list(frame.columns), FEATURE_COLS)
        self.assertEqual(frame.iloc[0].tolist(), [86000, 40, 9.2, 0.8, 13, 75])

    def test_train_and_recommend_parsers_expose_required_arguments(self):
        train = create_train_parser().parse_args(["--excel", "data.xlsx"])
        self.assertEqual(train.excel, "data.xlsx")
        recommend = create_recommend_parser().parse_args(
            ["--model", "champion.joblib", "--total-weight", "86000"]
        )
        self.assertEqual(recommend.total_weight, 86000)


if __name__ == "__main__":
    unittest.main()
