"""v0.5.0 Milestone 3 (benchmarks and reports): import, semantic near-duplicates, contamination,
dataset cards, HTML reports and the read-only dashboard."""

from __future__ import annotations

import csv
import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from _v040_helpers import correct_answers, evaluate, snapshot, write_dataset
from openbench.cli import main
from openbench.contamination import audit_contamination, normalise_words
from openbench.datacard import build_card, render_card
from openbench.dataset import DatasetLoader
from openbench.importers import import_benchmark
from openbench.metrics.semantic_similarity import CallableEmbeddingProvider
from openbench.quality import DatasetAuditor
from openbench.serve import Dashboard, create_server, serve_in_thread

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------------------------
# Import (#35)
# ---------------------------------------------------------------------------------------------
def test_import_mcq_csv_and_scoring(tmp_path, capsys):
    src = tmp_path / "mcq.csv"
    src.write_text(
        "id,question,A,B,C,D,answer\n"
        "q1,What is 2+2?,3,4,5,6,B\n"
        "q2,Capital of France?,Paris,Rome,Madrid,Berlin,a\n"
        "q3,Broken row,1,2,3,4,Z\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.jsonl"
    assert main(["import-dataset", str(src), "--format", "mcq-csv", "--category", "quiz", "--license", "CC-BY-4.0",
                 "--source", "unit test", "--output", str(out)]) == 0
    err = capsys.readouterr().err
    assert "skipped 1" in err and "line 4" in err
    ds = DatasetLoader.load(out)
    assert [r.id for r in ds.records] == ["quiz_q1", "quiz_q2"]
    first = ds.records[0]
    assert first.expected_answer == "B" and "A. 3\nB. 4" in first.prompt and "letter of the correct option only" in first.prompt
    assert first.metadata["license"] == "CC-BY-4.0" and first.metadata["source_id"] == "q1" and first.metadata["answer_text"] == "4"
    assert ds.records[1].expected_answer == "A"
    assert main(["import-dataset", str(src), "--format", "mcq-csv", "--category", "quiz", "--output", str(out)]) == 2


def test_import_mcq_jsonl_answer_forms_and_qa(tmp_path):
    mcq = tmp_path / "mcq.jsonl"
    rows = [
        {"question": "Pick two", "choices": ["one", "two", "three"], "answer": 1},
        {"question": "Pick red", "choices": {"A": "blue", "B": "red"}, "answer": "red"},
        {"question": "No choices", "answer": "A"},
        {"question": "Bad index", "choices": ["x", "y"], "answer": 7},
    ]
    mcq.write_text("\n".join(json.dumps(r) for r in rows) + "\n{not json\n", encoding="utf-8")
    res = import_benchmark(mcq, "mcq-jsonl", category="mc", difficulty="easy")
    assert [r["expected_answer"] for r in res.records] == ["B", "B"]
    assert [r["id"] for r in res.records] == ["mc_00001", "mc_00002"]
    assert {s["line"] for s in res.skipped} == {3, 4, 5}
    qa = tmp_path / "qa.csv"
    qa.write_text("q,a\nWho wrote Hamlet?,Shakespeare\n,empty\n", encoding="utf-8")
    res = import_benchmark(qa, "qa-csv", category="lit", question_field="q", answer_field="a", id_prefix="lit-set")
    assert res.records[0]["prompt"] == "Who wrote Hamlet?" and res.records[0]["expected_answer"] == "Shakespeare"
    assert res.records[0]["id"].startswith("lit-set_") and len(res.skipped) == 1
    out = tmp_path / "qa.jsonl"
    out.write_text(res.to_jsonl(), encoding="utf-8")
    assert len(DatasetLoader.load(out)) == 1


# ---------------------------------------------------------------------------------------------
# Semantic near-duplicates (#21)
# ---------------------------------------------------------------------------------------------
def test_semantic_near_duplicates_find_paraphrases(tmp_path):
    ds = tmp_path / "ds.jsonl"
    rows = [
        {"id": "a", "category": "c", "prompt": "What is the capital city of France?", "expected_answer": "Paris"},
        {"id": "b", "category": "c", "prompt": "Name France's seat of government.", "expected_answer": "Paris"},
        {"id": "c", "category": "c", "prompt": "How many legs does a spider have?", "expected_answer": "8"},
    ]
    ds.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    def fake_embed(text: str) -> list[float]:
        return [1.0, 0.0] if "France" in text else [0.0, 1.0]

    lexical = DatasetAuditor().audit(ds).to_dict()
    assert not [f for f in lexical["findings"] if f["check_id"] == "DQ003"]
    semantic = DatasetAuditor(embedding_provider=CallableEmbeddingProvider(fake_embed, model_id="fake")).audit(ds).to_dict()
    dq3 = [f for f in semantic["findings"] if f["check_id"] == "DQ003"]
    assert len(dq3) == 1 and dq3[0]["record_ids"] == ["a", "b"]
    assert dq3[0]["details"]["method"] == "semantic" and dq3[0]["details"]["embedding_model"] == "fake"
    assert semantic["summary"]["semantic"]["pairs_compared"] == 3
    assert semantic["parameters"]["semantic_near_duplicates"]["threshold"] == 0.9


def test_cli_semantic_method_without_extra_fails_cleanly(tmp_path, capsys, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_st(name, *args, **kwargs):
        if name.startswith("sentence_transformers"):
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_st)
    code = main(["validate-dataset", str(ROOT / "datasets/reasoning/sample.jsonl"), "--quality-checks", "--near-duplicate-method", "semantic"])
    assert code == 2 and "semantic" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------------
# Contamination (#47)
# ---------------------------------------------------------------------------------------------
def test_contamination_flags_copied_records(tmp_path):
    ds = tmp_path / "ds.jsonl"
    long_prompt = "Explain in detail why the sky appears blue during a clear day and red at sunset to an observer"
    rows = [
        {"id": "copied", "category": "sci", "prompt": long_prompt, "expected_answer": "Rayleigh scattering of sunlight by air molecules"},
        {"id": "fresh", "category": "sci", "prompt": "Describe how a refrigerator moves heat out of its interior compartment using a compressor", "expected_answer": "It compresses refrigerant"},
        {"id": "tiny", "category": "sci", "prompt": "Two plus two?", "expected_answer": "4"},
    ]
    ds.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "forum.txt").write_text("Someone asked: " + long_prompt.upper() + "!!! Great question.", encoding="utf-8")
    (corpus / "other.jsonl").write_text(json.dumps({"text": "unrelated words about cooking pasta in salted water for ten minutes"}) + "\n", encoding="utf-8")
    payload = audit_contamination(DatasetLoader.load(ds), corpus, ngram=8, min_words=5)
    by = {r["id"]: r for r in payload["records"]}
    assert by["copied"]["flagged"] and by["copied"]["fields"]["prompt"]["overlap"] == 1.0
    assert by["copied"]["fields"]["prompt"]["longest_shared_span_words"] == len(normalise_words(long_prompt))
    assert not by["fresh"]["flagged"] and by["fresh"]["fields"]["prompt"]["overlap"] == 0.0
    assert by["tiny"]["fields"]["prompt"]["skipped"] == "fewer_than_min_words"
    assert payload["flagged_ids"] == ["copied"] and payload["corpus"]["files"] == 2
    assert "does not prove" in payload["disclaimer"]


def test_cli_audit_contamination_formats(tmp_path, capsys):
    ds = ROOT / "datasets/reasoning/sample.jsonl"
    for fmt in ("json", "csv", "markdown"):
        assert main(["audit-contamination", str(ds), "--corpus", str(ds), "--ngram", "6", "--min-words", "4", "--format", fmt]) == 0
        out = capsys.readouterr().out
        assert "reasoning_001" in out
    with pytest.raises(FileNotFoundError):
        audit_contamination(DatasetLoader.load(ds), tmp_path / "missing")


@pytest.mark.parametrize(
    ("filename", "format_args", "expected"),
    [
        ("audit.json", [], "json"),
        ("audit.csv", [], "csv"),
        ("audit.json", ["--format", "csv"], "csv"),
        ("audit.txt", [], "markdown"),
        (None, [], "markdown"),
    ],
)
def test_cli_contamination_output_format(tmp_path, capsys, filename, format_args, expected):
    dataset = write_dataset(tmp_path / "dataset.jsonl", 1)
    args = ["audit-contamination", str(dataset), "--corpus", str(dataset), *format_args]
    if filename is None:
        assert main(args) == 0
        text = capsys.readouterr().out
    else:
        output = tmp_path / filename
        assert main([*args, "--output", str(output)]) == 0
        assert capsys.readouterr().out == f"Contamination audit ({expected}) written to: {output}\n"
        text = output.read_text(encoding="utf-8")
    if expected == "json":
        assert json.loads(text)["records"][0]["id"] == "item_001"
    elif expected == "csv":
        assert next(csv.DictReader(text.splitlines()))["id"] == "item_001"
    else:
        assert text.startswith("# OpenBench AI contamination risk audit\n")


# ---------------------------------------------------------------------------------------------
# Dataset cards (#48)
# ---------------------------------------------------------------------------------------------
def test_dataset_card_generated_and_placeholders(tmp_path, capsys):
    ds = ROOT / "datasets/reasoning/sample.jsonl"
    card = build_card(ds)
    assert card["generated"]["records"] == 20 and len(card["generated"]["checksum_sha256"]) == 64
    assert all(section["text"].startswith("TODO (author): ") for section in card["manual"].values())
    md = render_card(card)
    assert md == render_card(build_card(ds))  # deterministic: no timestamps
    assert "## Purpose" in md and "## Source and licence" in md and "**TODO (author):" in md
    out = tmp_path / "card.md"
    assert main(["dataset-card", str(ds), "--output", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == md
    capsys.readouterr()
    assert main(["dataset-card", str(ds), "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["kind"] == "dataset_card"


def test_dataset_card_html_is_deterministic_and_escapes_provenance(tmp_path):
    ds = tmp_path / "formatted.jsonl"
    ds.write_text(json.dumps({
        "id": "sample", "category": "example", "prompt": "<b>Which fruit is yellow?</b>",
        "expected_answer": "lemon", "metadata": {"license": "CC0-1.0", "source": "<b>Example & source</b>"},
    }) + "\n", encoding="utf-8")
    output = tmp_path / "card.html"
    assert main(["dataset-card", str(ds), "--format", "html", "--output", str(output)]) == 0
    document = output.read_text(encoding="utf-8")
    assert document == render_card(build_card(ds), "html")
    assert "<h2>At a glance</h2>" in document and "<h2>Quality checks</h2>" in document
    assert "<td>Licence</td><td>CC0-1.0 (all records)</td>" in document
    assert "&lt;b&gt;Example &amp; source&lt;/b&gt;" in document
    assert document.count('<p class="banner">TODO (author):') == 4
    assert "<b>" not in document and "<script" not in document and "<link" not in document


@pytest.mark.parametrize(
    ("name", "expected"),
    [("card.html", "html"), ("card.HTM", "html"), ("card.json", "json"), ("card.md", "markdown"),
     ("card.markdown", "markdown"), ("card.txt", "markdown"), ("card", "markdown")],
)
def test_dataset_card_format_follows_output_extension(tmp_path, capsys, name, expected):
    ds = ROOT / "datasets/reasoning/sample.jsonl"
    out = tmp_path / name
    assert main(["dataset-card", str(ds), "--output", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == render_card(build_card(ds), expected)
    assert f"Dataset card ({expected}) written to: " in capsys.readouterr().out


def test_dataset_card_stdout_without_output_stays_markdown(capsys):
    ds = ROOT / "datasets/reasoning/sample.jsonl"
    assert main(["dataset-card", str(ds)]) == 0
    assert capsys.readouterr().out == render_card(build_card(ds))


def test_dataset_card_explicit_format_wins_over_extension(tmp_path, capsys):
    ds = ROOT / "datasets/reasoning/sample.jsonl"
    out = tmp_path / "card.html"
    assert main(["dataset-card", str(ds), "--format", "markdown", "--output", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == render_card(build_card(ds), "markdown")
    assert "Dataset card (markdown) written to: " in capsys.readouterr().out
    assert main(["dataset-card", str(ds), "--format", "html", "--output", str(tmp_path / "card.json")]) == 0
    assert (tmp_path / "card.json").read_text(encoding="utf-8") == render_card(build_card(ds), "html")


def test_dataset_card_licence_and_source_shared_mixed_missing(tmp_path, capsys):
    src = tmp_path / "qa.csv"
    src.write_text("question,answer\nWho wrote Hamlet?,Shakespeare\nCapital of Italy?,Rome\n", encoding="utf-8")
    shared = tmp_path / "shared.jsonl"
    assert main(["import-dataset", str(src), "--format", "qa-csv", "--category", "lit", "--license", "CC-BY-4.0",
                 "--source", "unit test", "--output", str(shared)]) == 0
    capsys.readouterr()
    card = build_card(shared)
    assert card["generated"]["license"] == "CC-BY-4.0" and card["generated"]["source"] == "unit test"
    md = render_card(card)
    assert "| Licence | CC-BY-4.0 (all records) |" in md and "| Source | unit test (all records) |" in md
    assert card["manual"]["source_and_licence"]["text"].startswith("TODO (author): ")

    rows = [
        {"id": "a", "category": "c", "prompt": "Q one?", "expected_answer": "1", "metadata": {"license": "MIT", "source": "s1"}},
        {"id": "b", "category": "c", "prompt": "Q two?", "expected_answer": "2", "metadata": {"license": "CC0-1.0"}},
    ]
    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    card = build_card(mixed)
    assert card["generated"]["license"] is None and card["generated"]["source"] is None
    assert card["generated"]["provenance"]["license"]["distinct"] == ["CC0-1.0", "MIT"]
    assert card["generated"]["provenance"]["source"] == {"shared": None, "distinct": ["s1"], "records_missing": 1}
    md = render_card(card)
    assert "| Licence |" not in md and "| Source |" not in md
    text = card["manual"]["source_and_licence"]["text"]
    assert text.startswith("TODO (author): ") and "CC0-1.0, MIT" in text and "s1" in text

    card = build_card(ROOT / "datasets/reasoning/sample.jsonl")
    assert card["generated"]["license"] is None
    assert "| Licence |" not in render_card(card)
    assert card["manual"]["source_and_licence"]["text"].startswith("TODO (author): ")



# ---------------------------------------------------------------------------------------------
# HTML reports (#18)
# ---------------------------------------------------------------------------------------------
def _files(tmp_path, name_b="<script>alert('x')</script>"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    ds = write_dataset(tmp_path / "ds.jsonl", 12)
    a = evaluate(ds, correct_answers(12, wrong=(1, 2)), "model-a", tmp_path / "a.json")
    b = evaluate(ds, correct_answers(12, wrong=(3,)), name_b, tmp_path / "b.json")
    return a, b


def test_analyze_and_rank_html_are_self_contained_and_escaped(tmp_path, capsys):
    a, b = _files(tmp_path)
    out = tmp_path / "report.html"
    assert main(["analyze", "--results", str(a), str(b), "--format", "html", "--output", str(out),
                 "--resamples", "300", "--permutations", "999"]) == 0
    page = out.read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>") and "Content-Security-Policy" in page
    assert "<script>" not in page and "&lt;script&gt;" in page
    body = page.split("<main>")[1].replace('xmlns="http://www.w3.org/2000/svg"', "")
    links = (
        "https://lakshaychhaba0001-sys.github.io/openbench-ai/reading_a_report/",
        "https://lakshaychhaba0001-sys.github.io/openbench-ai/glossary/",
    )
    for url in links:
        assert f'href="{url}"' in body
        body = body.replace(url, "")
    assert "<svg" in page and "http://" not in body and "https://" not in body and "src=" not in page
    assert main(["rank", "--results", str(a), str(b), "--metric", "exact_match", "--format", "html",
                 "--resamples", "300", "--permutations", "999"]) == 0
    html_rank = capsys.readouterr().out
    assert "Model ranking" in html_rank and "<script>" not in html_rank
    assert all(f'href="{url}"' in html_rank for url in links)


def test_markdown_reports_link_to_the_documentation(tmp_path, capsys):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    a, b = _files(tmp_path, name_b="model-b")
    inputs = ["--results", str(a), str(b), "--resamples", "300", "--permutations", "999"]
    for command in (
        ["analyze", *inputs],
        ["rank", *inputs, "--metric", "exact_match"],
        ["plan", "--items", "200", "--baseline-accuracy", "0.7"],
        ["dataset-card", str(ROOT / "datasets/reasoning/sample.jsonl")],
        ["dataset-stats", str(ROOT / "datasets/reasoning/sample.jsonl")],
    ):
        assert main([*command, "--format", "markdown"]) == 0
        lines = capsys.readouterr().out.rstrip().splitlines()
        footer = lines[-1]
        assert "[Reading guide](https://lakshaychhaba0001-sys.github.io/openbench-ai/reading_a_report/)" in footer
        assert "[Glossary](https://lakshaychhaba0001-sys.github.io/openbench-ai/glossary/)" in footer
        if command[0] == "dataset-card":
            assert lines[-3:-1] == ["---", ""]
            assert "[Benchmark tools guide](https://lakshaychhaba0001-sys.github.io/openbench-ai/benchmark_tools/)" in footer

    responses = ROOT / "examples/sample_responses.json"
    for command, report, page_slug in (
        (["evaluate", "--dataset", str(ROOT / "datasets/reasoning/sample.jsonl"), "--responses", str(responses),
          "--output", str(tmp_path / "eval.md")], tmp_path / "eval.md", "metrics"),
        (["compare", "--results", str(a), str(b), "--output", str(tmp_path / "cmp.md")], tmp_path / "cmp.md",
         "model_comparison"),
    ):
        assert main(command) == 0
        lines = report.read_text(encoding="utf-8").rstrip().splitlines()
        assert lines[-2] == ""
        footer = lines[-1]
        assert f"https://lakshaychhaba0001-sys.github.io/openbench-ai/{page_slug}/" in footer
        assert "[Reading guide](https://lakshaychhaba0001-sys.github.io/openbench-ai/reading_a_report/)" in footer
        assert "[Glossary](https://lakshaychhaba0001-sys.github.io/openbench-ai/glossary/)" in footer

    prefs_dir = tmp_path / "prefs"
    assert main(["preferences", "create", "--results", str(a), str(b), "--output", str(prefs_dir)]) == 0
    key = json.loads((prefs_dir / "tasks.key.json").read_text(encoding="utf-8"))
    ann = tmp_path / "ann.jsonl"
    ann.write_text(
        "\n".join(
            json.dumps({"annotation_id": f"h{n}", "task_id": t["task_id"], "annotator_id": "h1", "outcome": "response_1"})
            for n, t in enumerate(key["tasks"])
        ) + "\n",
        encoding="utf-8",
    )
    capsys.readouterr()

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            body = json.dumps({"choices": [{"message": {"content": "1"}}]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    ds = ROOT / "datasets/reasoning/sample.jsonl"
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        commands = [
            (["preferences", "analyze", "--key", str(prefs_dir / "tasks.key.json"), "--annotations", str(ann), "--resamples", "200"], "human_preferences"),
            (["judge", "--results", str(a), str(b), "--judge-provider", "openai_compatible",
              "--judge-endpoint", f"http://127.0.0.1:{server.server_port}/v1", "--judge-model", "judge-z",
              "--cache-dir", str(tmp_path / "cache"), "--resamples", "200"], "llm_judge"),
            (["audit-contamination", str(ds), "--corpus", str(ds), "--ngram", "6", "--min-words", "4"], "benchmark_tools"),
        ]
        for command, page_slug in commands:
            assert main([*command, "--format", "markdown"]) == 0
            lines = capsys.readouterr().out.rstrip().splitlines()
            assert lines[-2] == ""
            footer = lines[-1]
            assert f"https://lakshaychhaba0001-sys.github.io/openbench-ai/{page_slug}/" in footer
            assert "[Reading guide](https://lakshaychhaba0001-sys.github.io/openbench-ai/reading_a_report/)" in footer
            assert "[Glossary](https://lakshaychhaba0001-sys.github.io/openbench-ai/glossary/)" in footer
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------------------------
# Dashboard (#36)
# ---------------------------------------------------------------------------------------------
@pytest.fixture
def dashboard_env(tmp_path, capsys):
    ds = tmp_path / "ds.jsonl"
    rows = [{"id": f"r{i}", "category": "c", "prompt": f"<script>alert({i})</script> question {i}?", "expected_answer": f"a{i}"} for i in range(5)]
    ds.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    exp = tmp_path / "exp"
    assert main(["run", "--dataset", str(ds), "--provider", "mock", "--model", "m1", "--repetitions", "2", "--seed", "1",
                 "--experiments-dir", str(exp), "--no-cache"]) == 0
    reports = tmp_path / "reports"
    reports.mkdir()
    a, b = _files(tmp_path / "files", name_b="model-b")
    assert main(["analyze", "--results", str(a), str(b), "--output", str(reports / "cmp.json"), "--resamples", "200", "--permutations", "199"]) == 0
    (reports / "notes.txt").write_text("ignored", encoding="utf-8")
    capsys.readouterr()
    return exp, reports


def _get(url: str) -> tuple[int, str, dict]:
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return resp.status, resp.read().decode("utf-8"), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8"), dict(exc.headers)


def test_dashboard_pages_escape_and_stay_read_only(dashboard_env):
    exp, reports = dashboard_env
    before = snapshot(exp) | {("reports", k): v for k, v in snapshot(reports).items()}
    server = create_server(exp, reports, port=0)
    serve_in_thread(server)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        status, body, headers = _get(base + "/")
        assert status == 200 and "Runs (2)" in body and "cmp.json" in body and "notes.txt" not in body
        assert "script-src" not in headers["Content-Security-Policy"] and "default-src 'none'" in headers["Content-Security-Policy"]
        run_id = body.split('href="/runs/')[1].split('"')[0]
        status, body, _ = _get(f"{base}/runs/{run_id}")
        assert status == 200 and "&lt;script&gt;alert(" in body and "<script>" not in body
        group_id = body.split("Repeat group</td><td>")[1].split("<")[0]
        status, body, _ = _get(f"{base}/groups/{group_id}")
        assert status == 200 and "Share from re-running" in body
        status, body, _ = _get(f"{base}/reports/cmp.json")
        assert status == 200 and "Paired model comparison" in body
        for bad in ("/runs/../../etc/passwd", "/reports/..%2Fsecret.json", "/reports/notes.txt", "/groups/grp_missing", "/nope"):
            assert _get(base + bad)[0] == 404
        req = urllib.request.Request(base + "/", data=b"x", method="POST")
        with pytest.raises(urllib.error.HTTPError) as info:
            urllib.request.urlopen(req, timeout=30)
        assert info.value.code == 405
    finally:
        server.shutdown()
        server.server_close()
    after = snapshot(exp) | {("reports", k): v for k, v in snapshot(reports).items()}
    assert before == after


def test_dashboard_refuses_remote_without_flag(tmp_path):
    with pytest.raises(ValueError, match="local-only"):
        create_server(tmp_path, host="0.0.0.0", port=0)
    assert "No runs found" in Dashboard(tmp_path / "missing").index()
