from pathlib import Path
import argparse
import json
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from advanced_furnace_ml.artifacts import load_bundle, predict_bundle
from advanced_furnace_ml.data import FEATURE_COLS


def build_parser():
    parser = argparse.ArgumentParser(description="使用 Advanced Furnace ML 预测总气耗")
    parser.add_argument("--model", required=True)
    for option in ("total-weight", "solid-ratio", "melting-time", "waiting-time", "door-open-count", "door-open-duration"):
        parser.add_argument(f"--{option}", required=True, type=float)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    values = [args.total_weight, args.solid_ratio, args.melting_time, args.waiting_time, args.door_open_count, args.door_open_duration]
    result = predict_bundle(load_bundle(args.model), pd.DataFrame([values], columns=FEATURE_COLS))
    print(json.dumps(result.iloc[0].to_dict(), ensure_ascii=False, indent=2))
