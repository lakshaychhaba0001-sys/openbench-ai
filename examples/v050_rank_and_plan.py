#!/usr/bin/env python3
"""Rank three offline mock models and plan a sample size from two stored runs.

Run from a checkout with `python examples/v050_rank_and_plan.py demo`.
Omit the output directory to keep the runs and report in a new temporary directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from tempfile import mkdtemp


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir", nargs="?", type=Path, help="Directory for runs and ranking.html"
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    # Use the checkout's package so the demo also works before installation.
    sys.path.insert(0, str(repo / "src"))
    from openbench import (
        DatasetLoader,
        EvaluationConfig,
        ExecutionConfig,
        ExperimentManager,
        MockModelProvider,
    )
    from openbench.cli import main as openbench

    output = (args.output_dir or Path(mkdtemp(prefix="openbench-rank-and-plan-"))).absolute()
    experiments = str(output / "experiments")
    dataset = DatasetLoader.load(repo / "datasets/reasoning/sample.jsonl")
    manager = ExperimentManager(experiments)
    metric = "exact_match"
    evaluation = EvaluationConfig(metric_names=[metric])
    execution = ExecutionConfig(use_cache=False)
    run_ids = []
    for model in ("mock-cedar", "mock-maple", "mock-birch"):
        run, _, _ = manager.execute_and_record(
            dataset,
            MockModelProvider(model_name=model),
            evaluation_config=evaluation,
            execution_config=execution,
        )
        run_ids.append(run.run_id)
        print(f"{model} stored as {run.run_id}")

    print("The mock provider returns the expected answer for every item, so all three models tie.")
    status = openbench([
        "rank", "--runs", *run_ids, "--metric", metric, "--experiments-dir", experiments,
        "--format", "html", "--output", str(output / "ranking.html"),
    ])
    if status:
        return status

    print(
        "The pilots agree on every item, so the planner uses the requested difference "
        "as the minimum discordance."
    )
    print("Sample-size plan from two stored pilot runs")
    return openbench([
        "plan", "--pilot-runs", *run_ids[:2], "--metric", metric, "--experiments-dir", experiments,
        "--difference", "0.05", "--format", "json",
    ])


if __name__ == "__main__":
    # Keep imports from writing bytecode caches into the checkout.
    sys.dont_write_bytecode = True
    raise SystemExit(main())
