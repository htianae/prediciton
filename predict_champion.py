"""Load the saved Champion and predict gas from six final batch values."""

import argparse
import json

import pandas as pd

from furnace_champion.artifact import load_artifact, predict_with_interval
from furnace_champion.data import FEATURE_COLS


def build_input_frame(total_weight, solid_ratio, melting_time, waiting_time, door_open_count, door_open_duration):
    return pd.DataFrame(
        [[total_weight, solid_ratio, melting_time, waiting_time, door_open_count, door_open_duration]],
        columns=FEATURE_COLS,
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="使用 Champion 预测炉次总气耗")
    parser.add_argument("--model", required=True)
    parser.add_argument("--total-weight", required=True, type=float)
    parser.add_argument("--solid-ratio", required=True, type=float)
    parser.add_argument("--melting-time", required=True, type=float)
    parser.add_argument("--waiting-time", required=True, type=float)
    parser.add_argument("--door-open-count", required=True, type=int)
    parser.add_argument("--door-open-duration", required=True, type=float)
    return parser


def main(argv=None) -> int:
    args = create_parser().parse_args(argv)
    bundle = load_artifact(args.model)
    frame = build_input_frame(
        args.total_weight,
        args.solid_ratio,
        args.melting_time,
        args.waiting_time,
        args.door_open_count,
        args.door_open_duration,
    )
    result = predict_with_interval(bundle, frame).iloc[0].to_dict()
    result["model_name"] = bundle["model_name"]
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
