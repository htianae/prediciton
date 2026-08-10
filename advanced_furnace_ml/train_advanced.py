from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from advanced_furnace_ml.pipeline import run_advanced_pipeline


def build_parser():
    parser = argparse.ArgumentParser(description="训练 Advanced Furnace ML 离线系统")
    parser.add_argument("--excel", required=True)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--optimizer-budget", type=int, default=600)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    result = run_advanced_pipeline(
        args.excel, args.output_dir, optimizer_budget=args.optimizer_budget, seeds=args.seeds
    )
    print(result["selected_before_lock"])
    print(result["production_recommendation"])
