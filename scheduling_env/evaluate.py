"""
Evaluation script for CalendarSchedulingEnv.

Runs agents across multiple seeds and tasks, computes statistics, and
optionally generates visualizations. Designed for reproducible benchmarking.

Usage:
    python -m scheduling_env.evaluate
    python -m scheduling_env.evaluate --seeds 10 --save-charts
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from typing import Any, Dict, List

from scheduling_env.baseline import (
    BaseAgent,
    GreedyAgent,
    HeuristicAgent,
    RandomAgent,
    run_episode,
)


TASKS = [
    "simple_scheduling",
    "constrained_scheduling",
    "complex_scheduling",
]

AGENTS: List[BaseAgent] = [
    RandomAgent(),
    GreedyAgent(),
    HeuristicAgent(),
]


def evaluate(
    agents: List[BaseAgent] | None = None,
    tasks: List[str] | None = None,
    num_seeds: int = 5,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Run all agent × task × seed combinations and return results.

    Returns a list of dicts, one per (agent, task, seed).
    """
    agents = agents or AGENTS
    tasks = tasks or TASKS
    all_results: List[Dict[str, Any]] = []

    for task in tasks:
        for agent in agents:
            for seed in range(num_seeds):
                result = run_episode(
                    agent=agent,
                    task_name=task,
                    seed=seed,
                    verbose=verbose,
                )
                result["seed"] = seed
                all_results.append(result)

    return all_results


def summarize(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate results into per-(agent, task) summary statistics."""
    from itertools import groupby

    summaries: List[Dict[str, Any]] = []

    key_fn = lambda r: (r["agent"], r["task"])
    sorted_results = sorted(results, key=key_fn)

    for (agent, task), group in groupby(sorted_results, key=key_fn):
        runs = list(group)
        rewards = [r["total_reward"] for r in runs]
        conflicts = [r["conflicts"] for r in runs]
        qualities = [r["quality"] for r in runs]

        summaries.append({
            "agent": agent,
            "task": task,
            "runs": len(runs),
            "reward_mean": round(statistics.mean(rewards), 2),
            "reward_std": round(statistics.stdev(rewards), 2) if len(rewards) > 1 else 0.0,
            "reward_min": round(min(rewards), 2),
            "reward_max": round(max(rewards), 2),
            "conflicts_mean": round(statistics.mean(conflicts), 1),
            "quality_mean": round(statistics.mean(qualities), 3),
            "perfect_runs": sum(1 for r in runs if r["conflicts"] == 0),
        })

    return summaries


def print_summary_table(summaries: List[Dict[str, Any]]) -> None:
    """Print a formatted summary table to stdout."""
    header = (
        f" {'Agent':<12s} | {'Task':<25s} | {'Runs':>4s} | "
        f"{'Reward':>12s} | {'Conflicts':>9s} | {'Quality':>7s} | {'Perfect':>7s}"
    )
    sep = "=" * len(header)

    print(f"\n+{sep}+")
    print(f"|  EVALUATION RESULTS{' ' * (len(sep) - 21)}|")
    print(f"+{sep}+")
    print(f"|{header}|")
    print(f"+{sep}+")

    current_task = None
    for s in summaries:
        if s["task"] != current_task:
            if current_task is not None:
                print(f"+{'-' * len(header)}+")
            current_task = s["task"]

        reward_str = f"{s['reward_mean']:>6.1f} +- {s['reward_std']:<4.1f}"
        row = (
            f" {s['agent']:<12s} | {s['task']:<25s} | {s['runs']:>4d} | "
            f"{reward_str:>12s} | {s['conflicts_mean']:>9.1f} | "
            f"{s['quality_mean']:>7.3f} | "
            f"{s['perfect_runs']:>3d}/{s['runs']:<3d}"
        )
        print(f"|{row}|")

    print(f"+{sep}+\n")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate CalendarSchedulingEnv agents",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of random seeds per (agent, task) pair (default: 5)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print step-by-step for each episode",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Save raw results to a JSON file",
    )
    parser.add_argument(
        "--save-charts",
        action="store_true",
        help="Save comparison charts (requires matplotlib)",
    )
    args = parser.parse_args()

    print(f"Running evaluation with {args.seeds} seeds per combination...")
    print(f"Agents: {[a.name for a in AGENTS]}")
    print(f"Tasks: {TASKS}")
    print()

    results = evaluate(num_seeds=args.seeds, verbose=args.verbose)
    summaries = summarize(results)
    print_summary_table(summaries)

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Raw results saved to {args.save_json}")

    if args.save_charts:
        try:
            from scheduling_env.visualize import render_comparison

            # Aggregate to one result per (agent, task) for the chart
            chart_data = []
            for s in summaries:
                chart_data.append({
                    "agent": s["agent"],
                    "task": s["task"],
                    "total_reward": s["reward_mean"],
                    "conflicts": s["conflicts_mean"],
                    "quality": s["quality_mean"],
                })
            render_comparison(chart_data, save_path="comparison.png", show=False)
        except ImportError:
            print("matplotlib not installed — skipping charts.")


if __name__ == "__main__":
    main()
