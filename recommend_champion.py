"""Recommend historically feasible furnace parameters with the saved Champion."""

import argparse
import json

from furnace_champion.artifact import load_artifact
from furnace_champion.optimization import (
    build_context,
    build_search_space,
    genetic_search,
    historical_fallback,
    random_search,
    summarize_recommendation,
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="使用 Champion 推荐低气耗可行参数")
    parser.add_argument("--model", required=True)
    parser.add_argument("--total-weight", required=True, type=float)
    parser.add_argument("--melting-time", type=float, default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None) -> int:
    args = create_parser().parse_args(argv)
    bundle = load_artifact(args.model)
    reference = bundle["training_reference"]
    context = build_context(reference, args.total_weight, args.melting_time)
    space = build_search_space(context["batches"])
    config = bundle.get("optimizer_config", {})

    if config.get("fallback_to_history", False):
        result = historical_fallback(context["batches"])
        result["warning"] = "Champion 未通过推荐安全门槛，已回退到历史相似低气耗 benchmark。"
    else:
        budget = args.budget or int(config.get("evaluation_budget", 5000))
        optimizer_name = bundle["optimizer_name"]
        function = random_search if optimizer_name == "random_search" else genetic_search
        candidates = function(
            bundle,
            context,
            space,
            bundle["feasibility_reference"],
            budget=budget,
            seed=args.seed,
        )
        result = summarize_recommendation(candidates, context)
        if result["source"] != "model_optimization":
            result = historical_fallback(context["batches"])
            result["warning"] = "没有找到历史可行候选，已回退到历史 benchmark。"
        result["optimizer_name"] = optimizer_name
        result["model_name"] = bundle["model_name"]
        result["estimated_improvement_only"] = True
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
