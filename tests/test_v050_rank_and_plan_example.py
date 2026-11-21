"""Smoke test for the offline v0.5.0 ranking and planning example."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from shutil import copy2, copytree, ignore_patterns

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("output_args", [("demo",), ("--", "-demo")])
def test_v050_rank_and_plan_example_runs(tmp_path, output_args):
    checkout = tmp_path / "checkout"
    copytree(
        ROOT / "src/openbench", checkout / "src/openbench", ignore=ignore_patterns("__pycache__")
    )
    copytree(ROOT / "datasets", checkout / "datasets")
    script = Path("examples/v050_rank_and_plan.py")
    (checkout / script.parent).mkdir()
    copy2(ROOT / script, checkout / script)
    before = set(tmp_path.rglob("*"))

    env = {key: os.environ[key] for key in ("SystemRoot",) if key in os.environ}
    result = subprocess.run(
        [sys.executable, str(script), *output_args],
        capture_output=True,
        encoding="utf-8",
        cwd=checkout,
        env=env,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    output = checkout / output_args[-1]
    assert {path for path in tmp_path.rglob("*") if not path.is_relative_to(output)} == before
    configs = sorted((output / "experiments").glob("run_*/config.json"))
    assert len(configs) == 3
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in configs]
    assert len({run["model_name"] for run in runs}) == 3
    run_ids = {run["run_id"] for run in runs}
    assert run_ids == {path.parent.name for path in configs}
    for run in runs:
        assert run["model_provider"] == "mock"
        assert run["status"] == "completed"
        assert Path(run["dataset_path"]) == checkout / "datasets/reasoning/sample.jsonl"
        assert run["aggregate_scores"]["exact_match"] == 1.0

    html = (output / "ranking.html").read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert "<h1>Model ranking</h1>" in html
    assert all(f"<td>{run_id}</td>" in html for run_id in run_ids)

    ranking_output, heading, plan_output = result.stdout.partition(
        "Sample-size plan from two stored pilot runs\n"
    )
    assert "all three models tie" in ranking_output
    assert heading
    plan = json.loads(plan_output)
    pilots = set(plan["pilot"]["pilot_runs"])
    assert len(pilots) == 2 and pilots <= run_ids
    assert plan["pilot"]["pilot_items"] == runs[0]["total_items"] > 1
    assert plan["pilot"]["p01"] == plan["pilot"]["p10"] == 0.0
    assert plan["discordance"] == plan["difference"]
    assert plan["items_needed"] > 0
    assert plan["provider_calls"] == 0
