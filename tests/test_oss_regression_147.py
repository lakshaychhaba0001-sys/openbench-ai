"""Markdown format coverage for openbench dataset-stats."""


import json

from openbench.cli import main


def test_dataset_stats_markdown(tmp_path, capsys):
    dataset = tmp_path / "small.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps({"id": f"item_{i}", "category": "a|b" if i == 0 else "plain",
                        "prompt": f"q{i}", "expected_answer": f"a{i}"})
            for i in range(3)
        ) + "\n",
        encoding="utf-8",
        )

    assert main(["dataset-stats", str(dataset), "--format", "markdown"]) == 0
    out = capsys.readouterr().out
    assert "# Dataset Statistics:" in out
    assert "| Category | Records |" in out
    assert "| a\\|b | 1 |" in out
    assert "| plain | 2 |" in out
    assert "| a|b |" not in out

    assert main(["dataset-stats", str(dataset)]) == 0
    text_out = capsys.readouterr().out
    assert text_out.startswith("Dataset Statistics:")
    assert "# Dataset Statistics" not in text_out

    assert main(["dataset-stats", str(dataset), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["categories"] == {"a|b": 1, "plain": 2}
    assert payload["total_records"] == 3
