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
    assert "openbench 0.5.1" in captured.out


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


@pytest.fixture
def compare_result_files(tmp_path, capsys):
    root = Path(__file__).resolve().parents[1]
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    sample = json.loads((root / "examples/sample_responses.json").read_text(encoding="utf-8"))
    results = []
    for model in ("ModelA", "ModelB", "ModelC"):
        responses = inputs / f"{model}-responses.json"
        responses.write_text(json.dumps({**sample, "model_name": model}), encoding="utf-8")
        result = inputs / f"{model}.json"
        assert main([
            "evaluate", "--dataset", str(root / "datasets/reasoning/sample.jsonl"),
            "--responses", str(responses), "--output", str(result),
        ]) == 0
        results.append(str(result))
    assert capsys.readouterr().err == ""
    return results


@pytest.mark.parametrize(
    ("suffix", "count", "companion"),
    [
        (".json", 2, None), (".csv", 2, None), (".md", 2, ".json"),
        (".markdown", 2, ".json"), (".MARKDOWN", 2, ".json"),
        (".md", 3, ".json"), (".markdown", 3, ".json"),
    ],
)
def test_cli_compare_single_file_output(tmp_path, capsys, compare_result_files, suffix, count, companion):
    output = tmp_path / f"comparison{suffix}"
    assert main([
        "compare", "--results", *compare_result_files[:count], "--output", str(output),
    ]) == 0
    assert output.is_file()
    expected = {output.name, "inputs"}
    if companion:
        expected.add(output.with_suffix(companion).name)
    assert {path.name for path in tmp_path.iterdir()} == expected
    report = output.read_text(encoding="utf-8")
    for model in ("ModelA", "ModelB", "ModelC")[:count]:
        assert model in report

    captured = capsys.readouterr()
    assert captured.err == ""
    if companion:
        assert captured.out == f"Comparison reports written to: {output}, {output.with_suffix(companion)}\n"
    else:
        assert captured.out == f"Comparison {suffix[1:].upper()} report written to: {output}\n"


@pytest.mark.parametrize("suffix", ["", ".txt", ".html", ".LOG"])
def test_cli_compare_directory_output(tmp_path, capsys, compare_result_files, suffix):
    output = tmp_path / f"comparison{suffix}"
    assert main([
        "compare", "--results", *compare_result_files[:2], "--output", str(output),
    ]) == 0
    assert output.is_dir()
    assert {path.name for path in output.iterdir()} == {"comparison.md", "comparison.json"}
    captured = capsys.readouterr()
    assert captured.out == (
        f"Comparison reports written to: {output / 'comparison.md'}, {output / 'comparison.json'}\n"
    )
    if suffix:
        assert captured.err == (
            f"Note: Using --output {output} as a directory because {suffix!r} is an unsupported extension. "
            "Single-file output requires a .json, .md, .markdown, or .csv extension.\n"
        )
    else:
        assert captured.err == ""


@pytest.mark.parametrize(
    ("suffix", "companion"),
    [(".json", ".md"), (".md", ".json"), (".csv", ".json"),
     (".markdown", ".json"), (".MARKDOWN", ".json")],
)
def test_cli_evaluate_single_file_output(tmp_path, capsys, suffix, companion):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / f"report{suffix}"
    assert main([
        "evaluate", "--dataset", str(root / "datasets/reasoning/sample.jsonl"),
        "--responses", str(root / "examples/sample_responses.json"), "--output", str(output),
    ]) == 0
    assert output.is_file()
    assert output.with_suffix(companion).is_file()
    if suffix.lower() in {".md", ".markdown"}:
        assert "AlphaBaseline-7B" in output.read_text(encoding="utf-8")
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    ("suffix", "multiple", "notice"),
    [
        ("", False, None),
        ("", True, None),
        (".txt", False, "unsupported extension"),
        (".html", False, "unsupported extension"),
        (".json", True, "more than one model"),
        (".md", True, "more than one model"),
        (".markdown", True, "more than one model"),
        (".csv", True, "more than one model"),
    ],
)
def test_cli_evaluate_directory_output(tmp_path, capsys, suffix, multiple, notice):
    root = Path(__file__).resolve().parents[1]
    responses = root / "examples/sample_responses.json"
    models = {"alphabaseline-7b"}
    if multiple:
        first = json.loads(responses.read_text(encoding="utf-8"))
        second = {**first, "model_name": "SecondModel"}
        responses = tmp_path / "models.json"
        responses.write_text(json.dumps({"models": [first, second]}), encoding="utf-8")
        models.add("secondmodel")
    output = tmp_path / f"report{suffix}"
    assert main([
        "evaluate", "--dataset", str(root / "datasets/reasoning/sample.jsonl"),
        "--responses", str(responses), "--output", str(output),
    ]) == 0
    assert output.is_dir()
    assert {p.name for p in output.iterdir()} == {
        f"{model}.{extension}" for model in models for extension in ("json", "md")
    }
    captured = capsys.readouterr()
    assert "Reports written to:" in captured.out
    if notice is None:
        assert captured.err == ""
    else:
        assert str(output) in captured.err
        assert "as a directory" in captured.err
        assert notice in captured.err
        assert all(ext in captured.err for ext in (".json", ".md", ".markdown", ".csv"))


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
