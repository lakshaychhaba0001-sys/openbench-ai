"""End-to-end unit tests for the OpenBench AI command-line interface."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from openbench.cli import _version_string, main


def test_cli_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    """`openbench --version` prints version and exits with code 0."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "openbench 0.5.0" in captured.out


def test_cli_version_flag_does_not_wrap_on_narrow_terminal(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """#109: `--version` is one unbroken line even when the terminal is very narrow."""
    monkeypatch.setenv("COLUMNS", "20")
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert captured.out == _version_string() + "\n"


def test_cli_no_arguments_returns_exit_code_2(capsys: pytest.CaptureFixture[str]) -> None:
    """Running `openbench` without subcommands prints help and returns exit code 2."""
    code = main([])
    assert code == 2
    captured = capsys.readouterr()
    assert "usage: openbench" in captured.out


def test_cli_validate_dataset_success_and_failure(tmp_path: Path) -> None:
    """`openbench validate-dataset` returns 0 on valid dataset and 1 on invalid dataset."""
    repo_root = Path(__file__).resolve().parent.parent
    valid_ds = repo_root / "datasets" / "reasoning" / "sample.jsonl"

    assert main(["validate-dataset", str(valid_ds)]) == 0

    invalid_ds = tmp_path / "bad.jsonl"
    invalid_ds.write_text("{corrupt json\n", encoding="utf-8")
    assert main(["validate-dataset", str(invalid_ds)]) == 1


def test_cli_evaluate_and_compare_workflow(tmp_path: Path) -> None:
    """`openbench evaluate` and `openbench compare` generate valid JSON and Markdown reports."""
    repo_root = Path(__file__).resolve().parent.parent
    dataset_path = repo_root / "datasets" / "reasoning" / "sample.jsonl"
    responses_a = repo_root / "examples" / "sample_responses.json"

    responses_b = tmp_path / "responses_b.json"
    responses_b.write_text(
        json.dumps(
            {
                "model_name": "ModelB",
                "model_version": "0.5",
                "responses": [
                    {"id": "reasoning_001", "response": "180"},
                    {"id": "reasoning_002", "response": "wrong"},
                ],
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "results"
    code_eval_a = main(
        [
            "evaluate",
            "--dataset",
            str(dataset_path),
            "--responses",
            str(responses_a),
            "--metrics",
            "exact_match",
            "accuracy",
            "keyword_coverage",
            "--output",
            str(out_dir),
        ]
    )
    assert code_eval_a == 0
    model_a_json = out_dir / "alphabaseline-7b.json"
    assert model_a_json.exists()
    assert (out_dir / "alphabaseline-7b.md").exists()

    code_eval_b = main(
        [
            "evaluate",
            "--dataset",
            str(dataset_path),
            "--responses",
            str(responses_b),
            "--output",
            str(out_dir),
        ]
    )
    assert code_eval_b == 0
    model_b_json = out_dir / "modelb.json"
    assert model_b_json.exists()

    cmp_md = out_dir / "comparison.md"
    code_cmp = main(
        [
            "compare",
            "--results",
            str(model_a_json),
            "--results",
            str(model_b_json),
            "--output",
            str(cmp_md),
        ]
    )
    assert code_cmp == 0
    assert cmp_md.exists()
    assert (out_dir / "comparison.json").exists()


def test_cli_compare_fewer_than_two_files_fails(tmp_path: Path) -> None:
    """`openbench compare` with only 1 result file returns exit code 2."""
    dummy = tmp_path / "one.json"
    dummy.write_text("{}", encoding="utf-8")
    code = main(["compare", "--results", str(dummy), "--output", str(tmp_path / "out.md")])
    assert code == 2


@pytest.mark.parametrize(
    ("host", "browser_host"),
    [("127.0.0.1", "127.0.0.1"), ("localhost", "localhost"),
     ("0.0.0.0", "127.0.0.1"), ("::", "[::1]"), ("::1", "[::1]"),
     ("192.0.2.1", "192.0.2.1"), ("2001:db8::1", "[2001:db8::1]")],
)
def test_cli_serve_open_opens_dashboard(monkeypatch, capsys, host, browser_host):
    opened = threading.Event()
    opened_urls = []

    def fake_open(url):
        opened_urls.append(url)
        opened.set()
        return True

    class FakeServer:
        server_address = (host, 8765)

        def serve_forever(self):
            raise KeyboardInterrupt  # Simulate Ctrl+C to stop the server

        def server_close(self):
            pass

    def fake_server(*args):
        assert args[2:] == (host, 8765, True)
        return FakeServer()

    monkeypatch.setattr("openbench.serve.create_server", fake_server)
    monkeypatch.setattr("openbench.cli.webbrowser.open", fake_open)
    assert main(["serve", "--host", host, "--allow-remote", "--open"]) == 0
    assert opened.wait(timeout=1)
    assert opened_urls == [f"http://{browser_host}:8765/"]
    printed_host = f"[{host}]" if ":" in host else host
    output = capsys.readouterr().out
    assert f"http://{printed_host}:8765/" in output
    if host in ("0.0.0.0", "::"):
        assert f"http://{browser_host}:8765/" in output


def test_cli_runs_compare_format_csv(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`openbench runs compare --format csv` outputs valid CSV matching --output with .csv."""
    repo_root = Path(__file__).resolve().parent.parent
    dataset_path = repo_root / "datasets" / "reasoning" / "sample.jsonl"
    exp_dir = tmp_path / "exp"

    assert main([
        "run",
        "--dataset", str(dataset_path),
        "--provider", "mock",
        "--model", "mock-model-a",
        "--experiments-dir", str(exp_dir),
        "--no-cache",
    ]) == 0

    assert main([
        "run",
        "--dataset", str(dataset_path),
        "--provider", "mock",
        "--model", "mock-model-b",
        "--experiments-dir", str(exp_dir),
        "--no-cache",
    ]) == 0

    from openbench.experiments import ExperimentManager

    manager = ExperimentManager(experiments_root=str(exp_dir))
    runs = manager.list_runs()
    assert len(runs) >= 2
    run_a = runs[0].run_id
    run_b = runs[1].run_id

    csv_out = tmp_path / "comparison.csv"
    assert main([
        "runs", "compare", run_a, run_b,
        "--experiments-dir", str(exp_dir),
        "--output", str(csv_out),
    ]) == 0

    with open(csv_out, "r", encoding="utf-8", newline="") as f:
        file_content = f.read()

    capsys.readouterr()
    assert main([
        "runs", "compare", run_a, run_b,
        "--experiments-dir", str(exp_dir),
        "--format", "csv",
    ]) == 0

    captured = capsys.readouterr().out
    assert captured == file_content

