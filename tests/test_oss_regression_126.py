"""HTTP coverage for filtering the dashboard runs list by status, provider and group."""

import json
import urllib.request

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
