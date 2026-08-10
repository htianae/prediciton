"""Train and export the single furnace gas Champion from the historical Excel file."""

import argparse
import json

from furnace_champion.training import train_and_select


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练并选择熔炼炉气耗 Champion 模型")
    parser.add_argument("--excel", required=True, help="历史 Excel 文件路径")
    parser.add_argument("--output-dir", default=".", help="artifact 和 report 输出根目录")
    parser.add_argument("--budget", type=int, default=5000, help="每次优化的候选评估预算")
    parser.add_argument("--seeds", type=int, default=10, help="优化算法重复随机种子数量")
    return parser


def main(argv=None) -> int:
    args = create_parser().parse_args(argv)
    result = train_and_select(
        args.excel,
        args.output_dir,
        evaluation_budget=args.budget,
        seeds=range(args.seeds),
    )
    print(
        json.dumps(
            {
                "champion_model": result["champion_model"],
                "optimizer_name": result["optimizer_name"],
                "fallback_to_history": result["fallback_to_history"],
                "artifact_path": result["artifact_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
