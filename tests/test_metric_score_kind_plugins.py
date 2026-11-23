"""analyze must use the score kind a plugin metric declares, in a fresh process (issue #107)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRIC = "probe_binary_match"

PLUGIN_MODULE = '''
from openbench.metrics.base import BaseMetric, MetricResult


class ProbeBinaryMatch(BaseMetric):
    name = "probe_binary_match"
    description = "1 when the response equals the expected answer."
    score_kind = "binary"

    def compute_item(self, record, response_text):
        ok = response_text.strip() == (record.expected_answer or "").strip()
        return MetricResult(metric_name=self.name, score=1.0 if ok else 0.0, passed=ok)
'''

RECORDS = [
    {"id": "a", "category": "probe", "prompt": "Say alpha.", "expected_answer": "alpha"},
    {"id": "b", "category": "probe", "prompt": "Say beta.", "expected_answer": "beta"},
    {"id": "c", "category": "probe", "prompt": "Say gamma.", "expected_answer": "gamma"},
]
ANSWERS = {
    "baseline": ["alpha", "wrong", "wrong"],
    "challenger": ["alpha", "beta", "gamma"],
}


def make_plugin_site(tmp_path: Path) -> Path:
    """Write a throwaway distribution with one metric entry point; nothing is pip-installed."""
    site = tmp_path / "site"
    (site / "ob_probe_plugin").mkdir(parents=True)
    (site / "ob_probe_plugin" / "__init__.py").write_text(textwrap.dedent(PLUGIN_MODULE), encoding="utf-8")
    dist = site / "ob_probe_plugin-0.1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: ob-probe-plugin\nVersion: 0.1.0\n", encoding="utf-8")
    (dist / "entry_points.txt").write_text(
        f"[openbench.metrics]\n{METRIC} = ob_probe_plugin:ProbeBinaryMatch\n", encoding="utf-8"
    )
    return site


def make_inputs(work: Path) -> None:
    """Write one dataset and two sets of saved responses (no provider calls are needed)."""
    (work / "dataset.jsonl").write_text("".join(json.dumps(r) + "\n" for r in RECORDS), encoding="utf-8")
    for name, answers in ANSWERS.items():
        responses = [{"id": r["id"], "response": a} for r, a in zip(RECORDS, answers, strict=True)]
        (work / f"{name}-responses.json").write_text(
            json.dumps({"model_name": name, "model_version": "1.0", "responses": responses}),
            encoding="utf-8",
        )


def run_cli(site: Path, work: Path, *args: str) -> None:
    """Run `openbench <args>` in a brand-new interpreter, so no plugin has been loaded yet."""
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join([str(site), str(ROOT / "src")]))
    environment.pop("OPENBENCH_DISABLE_PLUGINS", None)
    result = subprocess.run(
        [sys.executable, "-m", "openbench.cli", *args],
        cwd=work, env=environment, text=True, capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_analyze_uses_the_score_kind_declared_by_a_plugin_metric(tmp_path):
    site = make_plugin_site(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    make_inputs(work)

    for name in ANSWERS:
        run_cli(
            site, work, "evaluate", "--dataset", "dataset.jsonl",
            "--responses", f"{name}-responses.json", "--metrics", METRIC, "--output", f"{name}.json",
        )
    # A separate process from `evaluate`: analyze only reads the saved scores.
    run_cli(
        site, work, "analyze", "--results", "baseline.json", "challenger.json",
        "--metrics", METRIC, "--format", "json", "--output", "comparison.json",
    )

    comparison = json.loads((work / "comparison.json").read_text(encoding="utf-8"))
    assert '"score_kind": "binary"' in json.dumps(comparison)


def fresh_score_kinds(site: Path, names: list[str], *, plugins_disabled: bool = False) -> list[str]:
    """Return metric_score_kind(name) for each name, asked of one brand-new interpreter."""
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join([str(site), str(ROOT / "src")]))
    environment.pop("OPENBENCH_DISABLE_PLUGINS", None)
    if plugins_disabled:
        environment["OPENBENCH_DISABLE_PLUGINS"] = "1"
    code = (
        "from openbench.itemscores import metric_score_kind; "
        f"print(*[metric_score_kind(n) for n in {names!r}])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], env=environment, text=True, capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.split()


def test_disabled_plugins_leave_the_kind_unknown(tmp_path):
    site = make_plugin_site(tmp_path)
    assert fresh_score_kinds(site, [METRIC], plugins_disabled=True) == ["unknown"]
    assert fresh_score_kinds(site, [METRIC]) == ["binary"]


def test_broken_plugin_gives_unknown_instead_of_crashing(tmp_path):
    site = make_plugin_site(tmp_path)
    entry_points = site / "ob_probe_plugin-0.1.0.dist-info" / "entry_points.txt"
    entry_points.write_text(
        f"[openbench.metrics]\n{METRIC} = ob_probe_plugin:DoesNotExist\n", encoding="utf-8"
    )
    # The broken plugin and an unknown name stay "unknown"; a built-in still reports its own kind.
    kinds = fresh_score_kinds(site, [METRIC, "no_such_metric", "exact_match"])
    assert kinds == ["unknown", "unknown", "binary"]


def test_broken_plugin_is_tried_once_across_lookups(monkeypatch):
    from openbench import itemscores, plugins
    from openbench.exceptions import PluginError

    attempts = []

    def failing_load(kind=None):
        attempts.append(kind)
        raise PluginError("broken plugin")

    monkeypatch.setattr(plugins, "load_plugins", failing_load)
    monkeypatch.setattr(itemscores, "_plugin_load_failed", False)
    for name in ("missing_one", "missing_two", "missing_one"):
        assert itemscores.metric_score_kind(name) == "unknown"
    assert attempts == ["metrics"]


def test_registered_metrics_do_not_load_plugins(monkeypatch):
    from openbench import itemscores, plugins

    def fail_if_called(kind=None):
        raise AssertionError("load_plugins must not run for a registered metric")

    monkeypatch.setattr(plugins, "load_plugins", fail_if_called)
    monkeypatch.setattr(itemscores, "_plugin_load_failed", False)
    assert itemscores.metric_score_kind("exact_match") == "binary"
