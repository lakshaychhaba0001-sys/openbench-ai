"""Unit tests for local evaluation result history store and historical comparisons."""

from __future__ import annotations

from pathlib import Path

import pytest

from openbench.comparison import ModelComparator
from openbench.dataset import Dataset, DatasetRecord
from openbench.evaluator import Evaluator, ModelResponseSet, ResponseItem
from openbench.history import ResultHistoryStore


def test_result_history_store_record_list_and_show(tmp_path: Path) -> None:
    """ResultHistoryStore records runs to JSONL, filters by model/dataset, and retrieves by ID."""
    ds = Dataset(
        name="hist_ds",
        source_path="memory://hist_ds",
        records=[
            DatasetRecord(
                id="h1",
                category="reasoning",
                prompt="2 + 2?",
                expected_answer="4",
            ),
            DatasetRecord(
                id="h2",
                category="reasoning",
                prompt="3 + 3?",
                expected_answer="6",
            ),
        ],
    )
    evaluator = Evaluator()
    run1 = evaluator.evaluate(
        ds,
        ModelResponseSet(
            model_name="TrackedModel",
            model_version="1.0",
            responses=[
                ResponseItem(id="h1", response="4"),
                ResponseItem(id="h2", response="0"),
            ],
        ),
    )
    run2 = evaluator.evaluate(
        ds,
        ModelResponseSet(
            model_name="TrackedModel",
            model_version="1.1",
            responses=[
                ResponseItem(id="h1", response="4"),
                ResponseItem(id="h2", response="6"),
            ],
        ),
    )

    store_path = tmp_path / "history.jsonl"
    store = ResultHistoryStore(store_path)
    store.record(run1, report_path=tmp_path / "run1.json")
    store.record(run2, report_path=tmp_path / "run2.json")

    entries = store.list_entries(model_name="TrackedModel")
    assert len(entries) == 2
    fetched = store.get_entry(run1.evaluation_id)
    assert fetched.evaluation_id == run1.evaluation_id
    assert fetched.result_summary["exact_match"] == 0.5

    with pytest.raises(KeyError):
        store.get_entry("eval_nonexistent")

    hist_cmp = ModelComparator.compare_history([run1, run2])
    assert hist_cmp["total_runs_compared"] == 2
    assert hist_cmp["metric_deltas"]["exact_match"]["delta"] == 0.5


def _write_history(path: Path, rows: list[tuple[str, str, dict[str, float]]]) -> None:
    import json

    lines = []
    for i, (model, dataset, scores) in enumerate(rows):
        lines.append(
            json.dumps(
                {
                    "evaluation_id": f"eval_{i}",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "model_name": model,
                    "model_version": "1.0",
                    "dataset_name": dataset,
                    "dataset_version": "1",
                    "dataset_checksum": "x",
                    "metric_configuration": {},
                    "openbench_version": "0.4.0",
                    "result_summary": scores,
                    "total_items": 1,
                    "evaluated_items": 1,
                    "missing_responses_count": 0,
                    "report_path": None,
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_history_list_csv_format(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from openbench.cli import main

    store = tmp_path / "h.jsonl"
    _write_history(store, [("a,b|c", "ds", {"exact_match": 0.5}), ("m2", "ds", {"f1": 1.0})])
    assert main(["history", "list", "--store", str(store), "--format", "csv"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "evaluation_id,timestamp,model,model_version,dataset,exact_match,f1"
    assert '"a,b|c"' in lines[1]
    assert lines[1].endswith(",0.5000,")
    assert lines[2].endswith(",,1.0000")


def test_history_list_markdown_format(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from openbench.cli import main

    store = tmp_path / "h.jsonl"
    _write_history(store, [("a,b|c", "ds", {"exact_match": 0.5})])
    assert main(["history", "list", "--store", str(store), "--format", "markdown"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("| evaluation_id | timestamp | model |")
    assert lines[1].startswith("| ---")
    assert r"a,b\|c" in lines[2]


def test_history_list_export_empty_prints_header_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from openbench.cli import main

    store = tmp_path / "empty.jsonl"
    store.write_text("", encoding="utf-8")
    assert main(["history", "list", "--store", str(store), "--format", "csv"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "evaluation_id,timestamp,model,model_version,dataset"
    ]
