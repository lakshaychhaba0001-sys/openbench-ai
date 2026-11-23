"""HTTP coverage for filtering the dashboard runs list by status, provider and group."""

import json
import urllib.request
from types import SimpleNamespace

from openbench.cli import main
from openbench.serve import Dashboard, create_server, serve_in_thread


def test_dashboard_runs_filter(tmp_path, capsys):
    dataset = tmp_path / "small.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps({"id": f"item_{i}", "category": "c", "prompt": f"q{i}",
                        "expected_answer": f"a{i}"})
            for i in range(3)
        ) + "\n",
        encoding="utf-8",
        )
    experiments = tmp_path / "experiments"
    assert main(["run", "--dataset", str(dataset), "--provider", "mock", "--model", "alpha",
                 "--experiments-dir", str(experiments), "--no-cache"]) == 0
    assert main(["run", "--dataset", str(dataset), "--provider", "mock", "--model", "beta",
                 "--experiments-dir", str(experiments), "--no-cache",
                 "--repetitions", "2", "--seed", "1"]) == 0
    capsys.readouterr()

    records = Dashboard(experiments).manager.list_runs()
    assert len(records) == 3
    alpha = [r for r in records if r.model_name == "alpha"]
    beta = [r for r in records if r.model_name == "beta"]
    assert len(alpha) == 1 and len(beta) == 2
    group_id = beta[0].repeat_group_id
    assert group_id and all(r.repeat_group_id == group_id for r in beta)
    all_ids = [r.run_id for r in records]
    beta_ids = [r.run_id for r in beta]

    server = create_server(experiments, port=0)
    serve_in_thread(server)
    base = f"http://127.0.0.1:{server.server_address[1]}/"

    def get_page(query=""):
        with urllib.request.urlopen(base + query, timeout=10) as response:
            assert response.status == 200
            assert "default-src 'none'" in response.headers["Content-Security-Policy"]
            return response.read().decode("utf-8")

    try:
        body = get_page()
        assert "Runs (3)" in body and "Active filters" not in body
        for run_id in all_ids:
            assert run_id in body

        body = get_page("?status=completed")
        assert "status=completed" in body
        assert 'Active filters: status=completed. <a href="/">Show all runs</a>' in body
        for run_id in all_ids:
            assert run_id in body

        body = get_page("?provider=mock")
        assert "provider=mock" in body
        for run_id in all_ids:
            assert run_id in body

        body = get_page(f"?group={group_id}")
        assert f"group={group_id}" in body
        for run_id in beta_ids:
            assert run_id in body
        assert alpha[0].run_id not in body

        body = get_page(f"?status=completed&provider=mock&group={group_id}")
        for run_id in beta_ids:
            assert run_id in body
        assert alpha[0].run_id not in body

        body = get_page("?status=failed")
        assert "Runs (0)" in body and "No runs found." in body
        for run_id in all_ids:
            assert run_id not in body

        body = get_page("?provider=doesnotexist")
        assert "Runs (0)" in body

        body = get_page("?status=")
        assert "Runs (3)" in body and "Active filters" not in body

        body = get_page("?bogus=1")
        assert "Runs (3)" in body and "Active filters" not in body

        body = get_page("?status=%3Cscript%3Ealert(1)%3C/script%3E")
        assert "<script>" not in body and "&lt;script&gt;" in body
    finally:
        server.shutdown()
        server.server_close()

def test_dashboard_filter_links(tmp_path, capsys):
    dataset = tmp_path / "small.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps({"id": f"item_{i}", "category": "c", "prompt": f"q{i}",
                        "expected_answer": f"a{i}"})
            for i in range(3)
        ) + "\n",
        encoding="utf-8",
        )
    experiments = tmp_path / "experiments"
    assert main(["run", "--dataset", str(dataset), "--provider", "mock", "--model", "alpha",
                 "--experiments-dir", str(experiments), "--no-cache"]) == 0
    assert main(["run", "--dataset", str(dataset), "--provider", "mock", "--model", "beta",
                 "--experiments-dir", str(experiments), "--no-cache",
                 "--repetitions", "2", "--seed", "1"]) == 0
    capsys.readouterr()

    records = Dashboard(experiments).manager.list_runs()
    alpha = [r for r in records if r.model_name == "alpha"]
    beta = [r for r in records if r.model_name == "beta"]
    group_id = beta[0].repeat_group_id
    beta_ids = [r.run_id for r in beta]

    server = create_server(experiments, port=0)
    serve_in_thread(server)
    base = f"http://127.0.0.1:{server.server_address[1]}/"

    def get_page(query=""):
        with urllib.request.urlopen(base + query, timeout=10) as response:
            assert response.status == 200
            return response.read().decode("utf-8")

    try:
        body = get_page()
        assert 'Status: <a href="/?status=completed">completed</a>' in body
        assert 'Provider: <a href="/?provider=mock">mock</a>' in body
        assert f'Group: <a href="/?group={group_id}">{group_id}</a>' in body
        assert f"(<a href=\"/?group={group_id}\">show only this group's runs</a>)" in body

        marker = 'href="/?group='
        start = body.index(marker)
        end = body.index('"', start + len(marker))
        href = body[start + len('href="'):end]
        followed = get_page(href[1:])
        for run_id in beta_ids:
            assert run_id in followed
        assert alpha[0].run_id not in followed

        body = get_page("?provider=mock")
        assert 'href="/?status=completed&amp;provider=mock"' in body
        assert 'href="/?status=completed&provider=mock"' not in body
    finally:
        server.shutdown()
        server.server_close()


def test_dashboard_filter_links_escape_values(tmp_path, monkeypatch):
    fake = SimpleNamespace(
        run_id="run_fake1", status="completed", model_provider="evil&<script>",
        model_name="m", is_mock_provider=False, completed_items=1, total_items=1,
        aggregate_scores={}, repeat_group_id=None, created_at="2026-01-01",
    )
    dashboard = Dashboard(tmp_path)
    monkeypatch.setattr(dashboard.manager, "list_runs", lambda **kwargs: [fake])
    body = dashboard.index()
    expected = ('Provider: <a href="/?provider=evil%26%3Cscript%3E">'
                'evil&amp;&lt;script&gt;</a>')
    assert expected in body
    assert "<script>" not in body

def test_dashboard_model_filter(tmp_path, capsys):
    dataset = tmp_path / "small.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps({"id": f"item_{i}", "category": "c", "prompt": f"q{i}",
                        "expected_answer": f"a{i}"})
            for i in range(3)
        ) + "\n",
        encoding="utf-8",
        )
    experiments = tmp_path / "experiments"
    assert main(["run", "--dataset", str(dataset), "--provider", "mock", "--model", "alpha",
                 "--experiments-dir", str(experiments), "--no-cache"]) == 0
    assert main(["run", "--dataset", str(dataset), "--provider", "mock", "--model", "beta",
                 "--experiments-dir", str(experiments), "--no-cache",
                 "--repetitions", "2", "--seed", "1"]) == 0
    capsys.readouterr()

    records = Dashboard(experiments).manager.list_runs()
    alpha = [r for r in records if r.model_name == "alpha"]
    beta = [r for r in records if r.model_name == "beta"]
    beta_ids = [r.run_id for r in beta]

    server = create_server(experiments, port=0)
    serve_in_thread(server)
    base = f"http://127.0.0.1:{server.server_address[1]}/"

    def get_page(query=""):
        with urllib.request.urlopen(base + query, timeout=10) as response:
            assert response.status == 200
            return response.read().decode("utf-8")

    try:
        body = get_page()
        assert ('Model: <a href="/?model=alpha">alpha</a> · '
                '<a href="/?model=beta">beta</a>') in body

        body = get_page("?model=alpha")
        assert "Runs (1)" in body
        assert 'Active filters: model=alpha. <a href="/">Show all runs</a>' in body
        assert alpha[0].run_id in body
        for run_id in beta_ids:
            assert run_id not in body

        body = get_page("?model=BETA")
        for run_id in beta_ids:
            assert run_id in body
        assert alpha[0].run_id not in body

        body = get_page("?model=beta&status=completed")
        for run_id in beta_ids:
            assert run_id in body
        assert alpha[0].run_id not in body

        body = get_page("?model=doesnotexist")
        assert "Runs (0)" in body and "No runs found." in body

        body = get_page("?provider=mock")
        assert 'href="/?provider=mock&amp;model=alpha"' in body
    finally:
        server.shutdown()
        server.server_close()


def test_runs_list_model_filter(tmp_path, capsys):
    dataset = tmp_path / "small.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps({"id": f"item_{i}", "category": "c", "prompt": f"q{i}",
                        "expected_answer": f"a{i}"})
            for i in range(3)
        ) + "\n",
        encoding="utf-8",
        )
    experiments = tmp_path / "experiments"
    assert main(["run", "--dataset", str(dataset), "--provider", "mock", "--model", "alpha",
                 "--experiments-dir", str(experiments), "--no-cache"]) == 0
    assert main(["run", "--dataset", str(dataset), "--provider", "mock", "--model", "beta",
                 "--experiments-dir", str(experiments), "--no-cache",
                 "--repetitions", "2", "--seed", "1"]) == 0
    capsys.readouterr()

    records = Dashboard(experiments).manager.list_runs()
    alpha = [r for r in records if r.model_name == "alpha"]
    beta = [r for r in records if r.model_name == "beta"]

    assert main(["runs", "list", "--experiments-dir", str(experiments),
                 "--model", "ALPHA", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [r["run_id"] for r in payload] == [alpha[0].run_id]

    assert main(["runs", "list", "--experiments-dir", str(experiments),
                 "--model", "beta", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert sorted(r["run_id"] for r in payload) == sorted(r.run_id for r in beta)

    assert main(["runs", "list", "--experiments-dir", str(experiments),
                 "--model", "doesnotexist"]) == 0
    assert "No experiment runs found." in capsys.readouterr().out


def test_dashboard_model_link_escapes_values(tmp_path, monkeypatch):
    fake = SimpleNamespace(
        run_id="run_fake1", status="completed", model_provider="mock",
        model_name="evil&<script>", is_mock_provider=False, completed_items=1, total_items=1,
        aggregate_scores={}, repeat_group_id=None, created_at="2026-01-01",
    )
    dashboard = Dashboard(tmp_path)
    monkeypatch.setattr(dashboard.manager, "list_runs", lambda **kwargs: [fake])
    body = dashboard.index()
    expected = ('Model: <a href="/?model=evil%26%3Cscript%3E">'
                'evil&amp;&lt;script&gt;</a>')
    assert expected in body
    assert "<script>" not in body

def test_dashboard_dataset_filter(tmp_path, capsys):
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    alpha_ds = datasets_dir / "alpha.jsonl"
    beta_ds = datasets_dir / "beta.jsonl"
    rows = "\n".join(
        json.dumps({"id": f"item_{i}", "category": "c", "prompt": f"q{i}",
                    "expected_answer": f"a{i}"})
        for i in range(3)
    ) + "\n"
    alpha_ds.write_text(rows, encoding="utf-8")
    beta_ds.write_text(rows, encoding="utf-8")

    experiments = tmp_path / "experiments"
    assert main(["run", "--dataset", str(alpha_ds), "--provider", "mock", "--model", "m1",
                 "--experiments-dir", str(experiments), "--no-cache"]) == 0
    assert main(["run", "--dataset", str(beta_ds), "--provider", "mock", "--model", "m2",
                 "--experiments-dir", str(experiments), "--no-cache",
                 "--repetitions", "2", "--seed", "1"]) == 0
    capsys.readouterr()

    records = Dashboard(experiments).manager.list_runs()
    alpha = [r for r in records if r.dataset_identifier == "alpha"]
    beta = [r for r in records if r.dataset_identifier == "beta"]
    beta_ids = [r.run_id for r in beta]

    server = create_server(experiments, port=0)
    serve_in_thread(server)
    base = f"http://127.0.0.1:{server.server_address[1]}/"

    def get_page(query=""):
        with urllib.request.urlopen(base + query, timeout=10) as response:
            assert response.status == 200
            return response.read().decode("utf-8")

    try:
        body = get_page()
        assert ('Dataset: <a href="/?dataset=alpha">alpha</a> · '
                '<a href="/?dataset=beta">beta</a>') in body

        body = get_page("?dataset=alpha")
        assert "Runs (1)" in body
        assert 'Active filters: dataset=alpha. <a href="/">Show all runs</a>' in body
        assert 'href="/?status=completed&amp;dataset=alpha"' in body
        assert alpha[0].run_id in body
        for run_id in beta_ids:
            assert run_id not in body

        body = get_page("?dataset=BETA")
        for run_id in beta_ids:
            assert run_id in body
        assert alpha[0].run_id not in body

        body = get_page("?dataset=beta&model=m2&status=completed")
        for run_id in beta_ids:
            assert run_id in body
        assert alpha[0].run_id not in body

        body = get_page("?dataset=doesnotexist")
        assert "Runs (0)" in body and "No runs found." in body

        body = get_page("?provider=mock")
        assert 'href="/?provider=mock&amp;dataset=alpha"' in body
    finally:
        server.shutdown()
        server.server_close()


def test_runs_list_dataset_filter(tmp_path, capsys):
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    alpha_ds = datasets_dir / "alpha.jsonl"
    beta_ds = datasets_dir / "beta.jsonl"
    rows = "\n".join(
        json.dumps({"id": f"item_{i}", "category": "c", "prompt": f"q{i}",
                    "expected_answer": f"a{i}"})
        for i in range(3)
    ) + "\n"
    alpha_ds.write_text(rows, encoding="utf-8")
    beta_ds.write_text(rows, encoding="utf-8")

    experiments = tmp_path / "experiments"
    assert main(["run", "--dataset", str(alpha_ds), "--provider", "mock", "--model", "m1",
                 "--experiments-dir", str(experiments), "--no-cache"]) == 0
    assert main(["run", "--dataset", str(beta_ds), "--provider", "mock", "--model", "m2",
                 "--experiments-dir", str(experiments), "--no-cache",
                 "--repetitions", "2", "--seed", "1"]) == 0
    capsys.readouterr()

    manager = Dashboard(experiments).manager
    records = manager.list_runs()
    alpha = [r for r in records if r.dataset_identifier == "alpha"]
    beta = [r for r in records if r.dataset_identifier == "beta"]

    # Existing positional call signature still works
    assert [r.run_id for r in manager.list_runs("completed", "mock", None, "m1")] == [alpha[0].run_id]

    assert main(["runs", "list", "--experiments-dir", str(experiments),
                 "--dataset", "ALPHA", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [r["run_id"] for r in payload] == [alpha[0].run_id]

    assert main(["runs", "list", "--experiments-dir", str(experiments),
                 "--dataset", "beta", "--wide"]) == 0
    wide_out = capsys.readouterr().out
    assert "dataset=beta" in wide_out
    assert alpha[0].run_id not in wide_out
    for run in beta:
        assert run.run_id in wide_out

    assert main(["runs", "list", "--experiments-dir", str(experiments),
                 "--dataset", "doesnotexist"]) == 0
    assert "No experiment runs found." in capsys.readouterr().out


def test_dashboard_dataset_link_escapes_values(tmp_path, monkeypatch):
    fake = SimpleNamespace(
        run_id="run_fake1", status="completed", model_provider="mock",
        model_name="m", dataset_identifier="evil&<script>",
        is_mock_provider=False, completed_items=1, total_items=1,
        aggregate_scores={}, repeat_group_id=None, created_at="2026-01-01",
    )
    dashboard = Dashboard(tmp_path)
    monkeypatch.setattr(dashboard.manager, "list_runs", lambda **kwargs: [fake])
    body = dashboard.index()
    expected = ('Dataset: <a href="/?dataset=evil%26%3Cscript%3E">'
                'evil&amp;&lt;script&gt;</a>')
    assert expected in body
    assert "<script>" not in body

    filtered = dashboard.index(dataset="evil&<script>")
    assert "dataset=evil&amp;&lt;script&gt;" in filtered
    assert "<script>" not in filtered
