from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from advanced_furnace_ml.artifacts import load_bundle, recommend_bundle


def build_parser():
    parser = argparse.ArgumentParser(description="生成通过安全门的离线工艺试验候选")
    parser.add_argument("--model", required=True)
    parser.add_argument("--total-weight", required=True, type=float)
    parser.add_argument("--budget", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    result = recommend_bundle(load_bundle(args.model), args.total_weight, args.budget, args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))
