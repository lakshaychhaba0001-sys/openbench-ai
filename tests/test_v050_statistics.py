"""v0.5.0 Milestone 2 (statistics and judges): rank, plan, variance components, agreement, judge."""

from __future__ import annotations

import csv
import json
import math
import random
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from _v040_helpers import correct_answers, evaluate, write_dataset
from openbench.adapters import LocalCallableAdapter
from openbench.agreement import agreement_report, human_majority, metric_preference
from openbench.cli import main
from openbench.judge import PairwiseJudge, build_prompt, parse_verdict
from openbench.paired import ComparisonArm, ComparisonPolicy, PairedComparisonEngine
from openbench.preferences import PreferenceSource
from openbench.ranking import rank_arms
from openbench.stats import decompose_variability
from openbench.stats.components import two_level_bootstrap_ci, variance_components
from openbench.stats.power import (
    discordance_under_independence,
    mcnemar_power,
    minimum_detectable_difference_binary,
    minimum_detectable_difference_continuous,
    plan_binary,
    plan_continuous,
)
from openbench.stats.ranking import _average_ranks, rank_models
from openbench.stats.tests import mcnemar_exact

FAST = {"resamples": 300, "permutations": 999}
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}")


def _without_timestamp(text):
    """Blank the report's 'Generated at' time so two runs a second apart still compare equal."""
    return _TIMESTAMP.sub("TIMESTAMP", text)


# ---------------------------------------------------------------------------------------------
# Ranking (#33)
# ---------------------------------------------------------------------------------------------
def test_average_ranks_share_ties():
    assert _average_ranks([0.5, 0.9, 0.5, 0.1]) == [2.5, 1.0, 2.5, 4.0]


def test_rank_models_clear_separation():
    rng = random.Random(3)
    rows = [[rng.gauss(0.2, 0.05), rng.gauss(0.5, 0.05), rng.gauss(0.8, 0.05)] for _ in range(40)]
    out = rank_models(["low", "mid", "high"], rows, **FAST).to_dict()
    by = {m["model"]: m for m in out["models"]}
    assert [m["model"] for m in out["models"]] == ["high", "mid", "low"]
    assert by["high"]["rank_interval"] == [1, 1] and by["mid"]["rank_interval"] == [2, 2] and by["low"]["rank_interval"] == [3, 3]
    assert by["high"]["tied_with_top"] and not by["mid"]["tied_with_top"]
    assert by["high"]["p_best_bootstrap"] == pytest.approx(1.0)
    assert by["low"]["significantly_worse_than"] == ["high", "mid"]
    assert out["top_model"] == "high" and len(out["pairwise_tests"]) == 3
    assert all(t["significant"] for t in out["pairwise_tests"])


def test_rank_models_ties_give_wide_intervals_and_warning():
    rows = [[1.0, 1.0, 0.0] if i % 2 else [0.0, 0.0, 0.0] for i in range(20)]
    out = rank_models(["a", "b", "c"], rows, score_kind="binary", **FAST).to_dict()
    by = {m["model"]: m for m in out["models"]}
    assert by["a"]["observed_rank"] == by["b"]["observed_rank"] == 1.5
    assert by["a"]["rank_interval"][0] == 1
    assert any(t["test"] == "mcnemar_exact" for t in out["pairwise_tests"])
    assert {w["code"] for w in out["warnings"]} >= {"low_power_wide_intervals", "no_clear_winner"}


def test_rank_models_validation():
    with pytest.raises(ValueError):
        rank_models(["a"], [[1.0]])
    with pytest.raises(ValueError):
        rank_models(["a", "a"], [[1.0, 0.0]])
    with pytest.raises(ValueError):
        rank_models(["a", "b"], [[1.0]])
    with pytest.raises(ValueError):
        rank_models(["a", "b"], [[1.0, float("nan")]])


def test_rank_intervals_simultaneous_coverage_small_simulation():
    """All rank intervals contain the true ranks in at least ~95% of null-and-separated datasets."""
    rng = random.Random(11)
    true_means = [0.0, 0.0, 0.3, 0.6]
    true_ranks = [3.5, 3.5, 2, 1]  # tied truth: either 3 or 4 is a correct rank
    covered = 0
    sims = 60
    for _ in range(sims):
        rows = [[rng.gauss(m, 1.0) for m in true_means] for _ in range(30)]
        out = rank_models(list("abcd"), rows, resamples=200, permutations=499, analysis_seed=rng.randrange(10**6)).to_dict()
        ok = True
        for m in out["models"]:
            lo, hi = m["rank_interval"]
            t = true_ranks["abcd".index(m["model"])]
            allowed = {3, 4} if t == 3.5 else {t}
            ok &= any(lo <= r <= hi for r in allowed)
        covered += ok
    assert covered / sims >= 0.9


def _three_files(tmp_path):
    ds = write_dataset(tmp_path / "ds.jsonl", 30)
    a = evaluate(ds, correct_answers(30, wrong=tuple(range(1, 20))), "model-a", tmp_path / "a.json")
    b = evaluate(ds, correct_answers(30, wrong=tuple(range(1, 10))), "model-b", tmp_path / "b.json")
    c = evaluate(ds, correct_answers(30, wrong=(1,)), "model-c", tmp_path / "c.json")
    return a, b, c


def test_rank_arms_and_cli(tmp_path, capsys):
    a, b, c = _three_files(tmp_path)
    arms = [ComparisonArm.from_evaluation_file(p) for p in (a, b, c)]
    out = rank_arms(arms, "exact_match", **FAST)
    assert out["analysis"] == "model_ranking" and out["analysis_mode"] == "inferential"
    assert out["models"][0]["model"] == "c" and out["items_shared"] == 30
    assert out["provenance"]["provider_calls"] == 0
    for fmt in ("json", "markdown", "csv"):
        assert main(["rank", "--results", str(a), str(b), str(c), "--metric", "exact_match", "--format", fmt,
                     "--resamples", "300", "--permutations", "999"]) == 0
        text = capsys.readouterr().out
        assert "c" in text
    assert main(["rank", "--results", str(a), str(b), "--metric", "nope"]) == 1
    assert "not computed" in capsys.readouterr().err


def _rank_args(files, *extra):
    return ["rank", "--results", *map(str, files), "--metric", "exact_match",
            "--resamples", "300", "--permutations", "999", *extra]


@pytest.mark.parametrize(
    ("name", "expected"),
    [("ranking.html", "html"), ("ranking.HTM", "html"), ("ranking.json", "json"),
     ("ranking.csv", "csv"), ("ranking.md", "markdown"), ("ranking.markdown", "markdown"),
     ("ranking.txt", "markdown"), ("ranking", "markdown")],
)
def test_rank_format_follows_output_extension(tmp_path, capsys, name, expected):
    files = _three_files(tmp_path)
    out = tmp_path / name
    assert main(_rank_args(files, "--output", str(out))) == 0
    assert f"Ranking ({expected}) written to: " in capsys.readouterr().out
    text = out.read_text(encoding="utf-8")
    if expected == "json":  # the JSON payload carries a timestamp, so check its shape instead
        assert json.loads(text)["analysis"] == "model_ranking"
        return
    reference = tmp_path / "reference.out"
    assert main(_rank_args(files, "--format", expected, "--output", str(reference))) == 0
    assert _without_timestamp(text) == _without_timestamp(reference.read_text(encoding="utf-8"))
    assert text.startswith({"html": "<!doctype html>", "csv": "Rank,"}.get(expected, "# OpenBench"))


def test_rank_explicit_format_wins_over_extension(tmp_path, capsys):
    files = _three_files(tmp_path)
    out = tmp_path / "ranking.html"
    assert main(_rank_args(files, "--format", "markdown", "--output", str(out))) == 0
    assert "Ranking (markdown) written to: " in capsys.readouterr().out
    assert out.read_text(encoding="utf-8").startswith("# OpenBench AI model ranking")
    assert main(_rank_args(files, "--format", "csv", "--output", str(tmp_path / "ranking.json"))) == 0
    assert "Ranking (csv) written to: " in capsys.readouterr().out
    assert (tmp_path / "ranking.json").read_text(encoding="utf-8").startswith("Rank,")


def test_rank_without_output_prints_markdown(tmp_path, capsys):
    files = _three_files(tmp_path)
    assert main(_rank_args(files)) == 0
    assert capsys.readouterr().out.startswith("# OpenBench AI model ranking")


@pytest.fixture
def analyze_args(tmp_path):
    a, b, _ = _three_files(tmp_path)
    return ["analyze", "--results", str(a), str(b), "--metrics", "exact_match",
            "--resamples", "300", "--permutations", "999"]


@pytest.mark.parametrize(
    ("name", "expected", "prefix"),
    [
        ("report.html", "html", "<!doctype html>"),
        ("report.HTM", "html", "<!doctype html>"),
        ("report.md", "markdown", "# OpenBench AI paired comparison"),
        ("report.markdown", "markdown", "# OpenBench AI paired comparison"),
        ("report.csv", "csv", "comparison_id,metric,"),
        ("report.json", "json", "{"),
        ("report.txt", "json", "{"),
        ("report", "json", "{"),
    ],
)
def test_analyze_format_follows_output_extension(
    tmp_path, capsys, analyze_args, name, expected, prefix
):
    output = tmp_path / name
    assert main([*analyze_args, "--output", str(output)]) == 0
    assert capsys.readouterr().out == (
        f"Analysis (paired_comparison, {expected}) written to: {output}\n"
    )
    text = output.read_text(encoding="utf-8")
    assert text.startswith(prefix)

    reference = tmp_path / "explicit.out"
    assert main([*analyze_args, "--format", expected, "--output", str(reference)]) == 0
    assert _without_timestamp(text) == _without_timestamp(reference.read_text(encoding="utf-8"))


def test_analyze_explicit_format_wins_over_extension(tmp_path, capsys, analyze_args):
    output = tmp_path / "report.html"
    assert main([*analyze_args, "--format", "markdown", "--output", str(output)]) == 0
    assert capsys.readouterr().out == (
        f"Analysis (paired_comparison, markdown) written to: {output}\n"
    )
    assert output.read_text(encoding="utf-8").startswith("# OpenBench AI paired comparison")


def test_analyze_without_output_prints_json(capsys, analyze_args):
    assert main(analyze_args) == 0
    default = capsys.readouterr().out
    assert json.loads(default)["analysis"] == "paired_comparison"
    assert main([*analyze_args, "--format", "json"]) == 0
    assert _without_timestamp(default) == _without_timestamp(capsys.readouterr().out)


def test_rank_refuses_incompatible_inputs_without_exploratory(tmp_path, capsys):
    a, b, _ = _three_files(tmp_path)
    other = write_dataset(tmp_path / "other.jsonl", 30, tweak={"item_002": "changed"})
    d = evaluate(other, correct_answers(30), "model-d", tmp_path / "d.json")
    assert main(["rank", "--results", str(a), str(d), "--metric", "exact_match"]) == 1
    assert "incompatible" in capsys.readouterr().err.lower()
    assert main(["rank", "--results", str(a), str(d), "--metric", "exact_match", "--exploratory", "--format", "json"] + ["--resamples", "300", "--permutations", "999"]) == 0
    assert json.loads(capsys.readouterr().out.split("\nWarning")[0])["analysis_mode"] == "exploratory"


# ---------------------------------------------------------------------------------------------
# Sample-size planning (#34)
# ---------------------------------------------------------------------------------------------
def _brute_force_power(n, p01, p10, alpha=0.05):
    total = 0.0
    p11 = 1 - p01 - p10
    for b in range(n + 1):
        for c in range(n + 1 - b):
            rest = n - b - c
            prob = math.comb(n, b) * math.comb(n - b, c) * p01**b * p10**c * p11**rest
            if mcnemar_exact(b, c).p_value <= alpha:
                total += prob
    return total


@pytest.mark.parametrize("n,p01,p10", [(12, 0.3, 0.05), (25, 0.2, 0.1), (30, 0.15, 0.15)])
def test_exact_power_matches_brute_force_enumeration(n, p01, p10):
    assert mcnemar_power(n, p01, p10) == pytest.approx(_brute_force_power(n, p01, p10), abs=1e-9)


def test_power_matches_monte_carlo():
    rng = random.Random(7)
    n, p01, p10, sims = 80, 0.2, 0.06, 3000
    hits = 0
    for _ in range(sims):
        b = c = 0
        for _ in range(n):
            u = rng.random()
            b += u < p01
            c += p01 <= u < p01 + p10
        hits += mcnemar_exact(b, c).p_value <= 0.05
    mc = hits / sims
    assert abs(mc - mcnemar_power(n, p01, p10)) < 4 * math.sqrt(mc * (1 - mc) / sims)


def test_plan_binary_inputs_and_monotonicity():
    q01, q10 = discordance_under_independence(0.7, 0.05)
    assert (q01, q10) == pytest.approx((0.3 * 0.75, 0.7 * 0.25))
    small = plan_binary(delta=0.1, discordance=0.2)
    large = plan_binary(delta=0.05, discordance=0.2)
    assert small["items_needed"] < large["items_needed"]
    assert small["power_at_items_needed"] >= 0.8
    assert mcnemar_power(small["items_needed"] - 1, small["p01"], small["p10"]) < 0.8
    more_disagreement = plan_binary(delta=0.1, discordance=0.4)
    assert more_disagreement["items_needed"] > small["items_needed"]
    with pytest.raises(ValueError):
        plan_binary(delta=0.3, discordance=0.2)
    with pytest.raises(ValueError):
        plan_binary(delta=0.0, discordance=0.2)
    assert plan_binary(delta=0.01, discordance=0.5, max_n=200)["items_needed"] is None


def test_continuous_plan_and_reverse_mode_agree():
    plan = plan_continuous(delta=0.05, sd_difference=0.3)
    z = 1.959963984540054 + 0.8416212335729143
    assert plan["items_needed"] == math.ceil((z * 0.3 / 0.05) ** 2)
    mdd = minimum_detectable_difference_continuous(plan["items_needed"], sd_difference=0.3)
    assert mdd <= 0.05
    mdd_bin = minimum_detectable_difference_binary(300, baseline_accuracy=0.7)
    q01, q10 = discordance_under_independence(0.7, mdd_bin)
    assert mcnemar_power(300, q01, q10) == pytest.approx(0.8, abs=0.01)


def test_cli_plan_modes(tmp_path, capsys):
    assert main(["plan", "--discordance", "0.2", "--difference", "0.1", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["items_needed"] == plan_binary(delta=0.1, discordance=0.2)["items_needed"]
    assert main(["plan", "--items", "200", "--baseline-accuracy", "0.7"]) == 0
    assert "smallest difference" in capsys.readouterr().out
    assert main(["plan", "--continuous", "--sd-difference", "0.3", "--difference", "0.05", "--format", "csv"]) == 0
    assert "items_needed" in capsys.readouterr().out
    assert main(["plan", "--difference", "0.1"]) == 2
    a, b, _ = _three_files(tmp_path)
    assert main(["plan"]) == 2


def test_cli_plan_from_pilot_runs(tmp_path, capsys):
    exp = tmp_path / "exp"
    ds = write_dataset(tmp_path / "ds.jsonl", 20)
    for model in ("p-a", "p-b"):
        assert main(["run", "--dataset", str(ds), "--provider", "mock", "--model", model, "--experiments-dir", str(exp), "--no-cache"]) == 0
    capsys.readouterr()
    assert main(["runs", "list", "--experiments-dir", str(exp), "--format", "json"]) == 0
    ids = [r["run_id"] for r in json.loads(capsys.readouterr().out)]
    assert main(["plan", "--pilot-runs", *ids, "--difference", "0.05", "--experiments-dir", str(exp), "--format", "json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["pilot"]["pilot_items"] == 20 and out["provider_calls"] == 0


# ---------------------------------------------------------------------------------------------
# Variance components (#45)
# ---------------------------------------------------------------------------------------------
def test_variance_components_hand_example():
    matrix = [[1, 1, 1], [1, 0, 1], [0, 0, 0], [0, 1, 1]]
    comps = variance_components(matrix)
    means = [1, 2 / 3, 0, 2 / 3]
    grand = sum(means) / 4
    msw = sum((v - means[i]) ** 2 for i, r in enumerate(matrix) for v in r) / (4 * 2)
    msb = 3 * sum((m - grand) ** 2 for m in means) / 3
    assert comps["sigma2_repetition"] == pytest.approx(msw)
    assert comps["sigma2_item"] == pytest.approx(max(0, (msb - msw) / 3))
    assert comps["share_repetition"] + comps["share_item"] == pytest.approx(1.0)
    assert variance_components([[1.0], [0.0]])["unavailable_reason"]


def test_variance_components_recover_known_values():
    """Averaged over seeds, the ANOVA estimates are close to the true components (4 and 1)."""
    items, reps = [], []
    for seed in range(5):
        rng = random.Random(seed)
        rows = []
        for _ in range(400):
            item = rng.gauss(0, 2.0)  # sigma2_item = 4
            rows.append([item + rng.gauss(0, 1.0) for _ in range(4)])  # sigma2_rep = 1
        comps = variance_components(rows)
        items.append(comps["sigma2_item"])
        reps.append(comps["sigma2_repetition"])
    assert sum(items) / 5 == pytest.approx(4.0, rel=0.1)
    assert sum(reps) / 5 == pytest.approx(1.0, rel=0.06)


def test_two_level_ci_is_seeded_and_covers_mean():
    rows = [[0.1 * i + 0.01 * r for r in range(3)] for i in range(10)]
    a = two_level_bootstrap_ci(rows, resamples=400, analysis_seed=1)
    b = two_level_bootstrap_ci(rows, resamples=400, analysis_seed=1)
    assert a == b and a["lower"] <= a["estimate"] <= a["upper"]


def test_two_level_ci_coverage_simulation():
    rng = random.Random(21)
    sims, hits = 80, 0
    for _ in range(sims):
        rows = []
        for _ in range(25):
            item = rng.gauss(0.0, 1.0)
            rows.append([item + rng.gauss(0.0, 1.0) for _ in range(3)])
        ci = two_level_bootstrap_ci(rows, resamples=400, analysis_seed=rng.randrange(10**6))
        hits += ci["lower"] <= 0.0 <= ci["upper"]
    assert hits / sims >= 0.88


def test_decompose_variability_reports_components(tmp_path):
    from test_variability import _evals  # reuse the v0.4.0 helper

    rep = decompose_variability(_evals(tmp_path, [[1, 1, 1], [1, 0, 1], [0, 0, 0], [0, 1, 1]]), resamples=300)
    vc = rep["metrics"]["exact_match"]["variance_components"]
    assert vc["items_used_complete_cases"] == 4
    assert "Re-running explains" in vc["summary"]
    assert vc["two_level_confidence_interval"]["resamples"] == 300


# ---------------------------------------------------------------------------------------------
# Metric–human agreement (#46)
# ---------------------------------------------------------------------------------------------
def test_human_majority_and_metric_preference():
    assert human_majority(["A", "A", "B"]) == "A"
    assert human_majority(["A", "B"]) is None
    assert human_majority(["neither", "tie"]) == "tie"
    assert human_majority(["neither"]) is None
    assert metric_preference(1.0, 0.0) == "A" and metric_preference(0.0, 1.0) == "B"
    assert metric_preference(0.5, 0.5) == "tie" and metric_preference(None, 1.0) is None


def test_agreement_report_counts_and_kappa():
    human = {"t1": ["A"], "t2": ["B", "B"], "t3": ["A", "B"], "t4": ["tie"], "t5": ["A"]}
    auto = {"t1": "A", "t2": "B", "t3": "A", "t4": "A", "t5": None, "t6": "B"}
    rep = agreement_report(human, auto, categories={"t1": "x", "t2": "y", "t4": "x"}, resamples=200)
    assert rep["tasks_compared"] == 3
    assert rep["tasks_excluded"] == {"no_automatic_label": 1, "no_human_annotations": 1, "no_human_majority": 1}
    assert rep["percent_agreement"] == pytest.approx(2 / 3)
    assert rep["decisive_tasks"] == 2 and rep["decisive_agreement"] == 1.0
    assert rep["confusion_human_by_automatic"]["tie"]["A"] == 1
    assert rep["by_category"]["x"]["tasks"] == 2


def _prefs_setup(tmp_path, capsys):
    ds = write_dataset(tmp_path / "ds.jsonl", 12)
    a = evaluate(ds, correct_answers(12, wrong=(1, 2, 3)), "model-a", tmp_path / "a.json")
    b = evaluate(ds, correct_answers(12, wrong=(4,)), "model-b", tmp_path / "b.json")
    out = tmp_path / "prefs"
    assert main(["preferences", "create", "--results", str(a), str(b), "--output", str(out)]) == 0
    capsys.readouterr()
    key = json.loads((out / "tasks.key.json").read_text())
    return a, b, out, key


def _display(key, task, model_outcome):
    if model_outcome in ("tie", "neither"):
        return model_outcome
    order = task["display_order"]
    return {"AB": {"A": "response_1", "B": "response_2"}, "BA": {"A": "response_2", "B": "response_1"}}[order][model_outcome]


def test_cli_preferences_compare_metric(tmp_path, capsys):
    a, b, out, key = _prefs_setup(tmp_path, capsys)
    rows = []
    for t in key["tasks"]:
        i = int(t["record_id"].split("_")[1])
        truth = "B" if i in (1, 2, 3) else ("A" if i == 4 else "tie")
        rows.append({"annotation_id": f"h-{t['task_id']}", "task_id": t["task_id"], "annotator_id": "h1",
                     "outcome": _display(key, t, truth)})
    ann = tmp_path / "ann.jsonl"
    ann.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert main(["preferences", "analyze", "--key", str(out / "tasks.key.json"), "--annotations", str(ann),
                 "--compare-metric", "exact_match", "--metric-results", str(a), str(b), "--resamples", "200"]) == 0
    result = json.loads(capsys.readouterr().out)
    agreement = result["metric_agreement"]
    assert agreement["tasks_compared"] == 12 and agreement["percent_agreement"] == 1.0
    assert agreement["automatic"] == "metric:exact_match"
    assert main(["preferences", "analyze", "--key", str(out / "tasks.key.json"), "--annotations", str(ann),
                 "--compare-metric", "exact_match", "--metric-results", str(a), str(b), "--resamples", "200", "--format", "markdown"]) == 0
    assert "Agreement of `metric:exact_match`" in capsys.readouterr().out


@pytest.fixture
def preference_analysis_args(tmp_path, capsys):
    _, _, out, key = _prefs_setup(tmp_path, capsys)
    annotations = tmp_path / "annotations.jsonl"
    rows = [
        {"annotation_id": f"a-{i}", "task_id": task["task_id"],
         "annotator_id": "rater", "outcome": "response_1"}
        for i, task in enumerate(key["tasks"])
    ]
    annotations.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return [
        "preferences", "analyze", "--key", str(out / "tasks.key.json"),
        "--annotations", str(annotations), "--resamples", "200",
    ]


@pytest.mark.parametrize(
    ("filename", "expected", "prefix"),
    [
        ("prefs.md", "markdown", "# OpenBench AI human preference analysis"),
        ("prefs.markdown", "markdown", "# OpenBench AI human preference analysis"),
        ("prefs.csv", "csv", "model_a,model_b,breakdown,"),
        ("prefs.CSV", "csv", "model_a,model_b,breakdown,"),
        ("prefs.json", "json", "{"),
        ("prefs.txt", "json", "{"),
        ("report", "json", "{"),
        ("prefs.html", "json", "{"),
    ],
)
def test_preferences_output_infers_format(
    tmp_path, capsys, preference_analysis_args, filename, expected, prefix
):
    output = tmp_path / filename
    assert main([*preference_analysis_args, "--output", str(output)]) == 0
    captured = capsys.readouterr()
    assert captured.out == f"Preference analysis ({expected}) written to: {output}\n"
    assert captured.err == ""
    text = output.read_text(encoding="utf-8")
    assert text.startswith(prefix)

    explicit = tmp_path / "explicit.out"
    assert main([
        *preference_analysis_args, "--format", expected, "--output", str(explicit),
    ]) == 0
    assert _without_timestamp(text) == _without_timestamp(explicit.read_text(encoding="utf-8"))


def test_preferences_explicit_format_overrides_extension(
    tmp_path, capsys, preference_analysis_args
):
    output = tmp_path / "prefs.md"
    assert main([*preference_analysis_args, "--format", "csv", "--output", str(output)]) == 0
    assert capsys.readouterr().out == f"Preference analysis (csv) written to: {output}\n"
    assert output.read_text(encoding="utf-8").startswith("model_a,model_b,breakdown,")


def test_preferences_stdout_defaults_to_json(capsys, preference_analysis_args):
    assert main(preference_analysis_args) == 0
    default = capsys.readouterr().out
    assert json.loads(default)["analysis"] == "preference_analysis"
    assert main([*preference_analysis_args, "--format", "json"]) == 0
    assert _without_timestamp(default) == _without_timestamp(capsys.readouterr().out)


# ---------------------------------------------------------------------------------------------
# LLM judge (#19)
# ---------------------------------------------------------------------------------------------
def test_parse_verdict_and_prompt():
    assert parse_verdict("1") == "1" and parse_verdict("**2**\nbecause") == "2"
    assert parse_verdict("Tie.") == "tie" and parse_verdict("\n\nResponse 2 is better") == "2"
    assert parse_verdict("I cannot decide") is None and parse_verdict("") is None
    prompt = build_prompt("Q?", "first", "second", reference="ref")
    assert "[Reference answer]\nref" in prompt and prompt.index("first") < prompt.index("second")


def _sources():
    def src(sid, model, answers):
        items = {f"r{i}": {"prompt": f"What is {i}?", "response": ans, "reference": str(i), "category": "math",
                           "difficulty": "easy", "valid": True} for i, ans in enumerate(answers)}
        return PreferenceSource(sid, model, "1", "local_lookup", items)

    a = src("run_a", "model-a", ["0", "wrong", "2", "wrong but very long answer"])
    b = src("run_b", "model-b", ["no", "1", "2", "3"])
    return a, b


def _correctness_judge(prompt: str) -> str:
    """Picks the response that equals the reference number; tie if both or neither."""
    q = int(prompt.split("[Question]\nWhat is ")[1].split("?")[0])
    r1 = prompt.split("[Response 1]\n")[1].split("\n\n[Response 2]")[0].strip()
    r2 = prompt.split("[Response 2]\n")[1].split("\n\nReply with")[0].strip()
    ok1, ok2 = r1 == str(q), r2 == str(q)
    return "tie" if ok1 == ok2 else ("1" if ok1 else "2")


def test_judge_consistent_correctness_judge():
    a, b = _sources()
    judge = PairwiseJudge(LocalCallableAdapter("judge-x", _correctness_judge), resamples=200)
    out = judge.judge_sources(a, b).to_dict()
    verdicts = {t["record_id"]: t["verdict"] for t in out["tasks"]}
    assert verdicts == {"r0": "A", "r1": "B", "r2": "tie", "r3": "B"}
    assert out["position_consistency_rate"] == 1.0 and out["judge_calls"] == 8
    assert out["judge"]["prompt_version"] == "pairwise-v1" and len(out["judge"]["prompt_sha256"]) == 64
    assert out["win_rate_a"] == pytest.approx((1 + 0 + 0.5 + 0) / 4)


def test_judge_position_bias_is_detected_and_scored_as_tie():
    a, b = _sources()
    out = PairwiseJudge(LocalCallableAdapter("judge-y", lambda p: "1"), resamples=200).judge_sources(a, b).to_dict()
    assert out["position_consistency_rate"] == 0.0
    assert all(t["status"] == "inconsistent" and t["verdict"] == "tie" for t in out["tasks"])
    assert out["position_bias"]["share_first_shown_wins"] == 1.0
    assert "low_position_consistency" in {w["code"] for w in out["warnings"]}


def test_judge_self_preference_and_unparsed():
    a, b = _sources()
    out = PairwiseJudge(LocalCallableAdapter("model-a", lambda p: "maybe"), resamples=200).judge_sources(a, b).to_dict()
    codes = {w["code"] for w in out["warnings"]}
    assert {"self_preference_risk", "unparsed_verdicts"} <= codes
    assert out["win_rate_a"] is None


@pytest.mark.parametrize(
    ("filename", "format_args", "expected"),
    [
        ("judge.json", [], "json"),
        ("judge.csv", [], "csv"),
        ("judge.csv", ["--format", "json"], "json"),
        ("judge.txt", [], "markdown"),
        (None, [], "markdown"),
    ],
)
def test_cli_judge_output_format(tmp_path, capsys, monkeypatch, filename, format_args, expected):
    dataset = write_dataset(tmp_path / "dataset.jsonl", 1)
    a = evaluate(dataset, correct_answers(1), "model-a", tmp_path / "a.json")
    b = evaluate(dataset, correct_answers(1, wrong=(1,)), "model-b", tmp_path / "b.json")
    adapter = LocalCallableAdapter("offline", lambda prompt: "tie")
    monkeypatch.setattr("openbench.cli.create_provider", lambda *args, **kwargs: adapter)
    args = [
        "judge", "--results", str(a), str(b), "--judge-provider", "local_callable",
        "--judge-model", "offline", "--no-cache", "--resamples", "200", *format_args,
    ]
    if filename is None:
        assert main(args) == 0
        text = capsys.readouterr().out
    else:
        output = tmp_path / filename
        assert main([*args, "--output", str(output)]) == 0
        assert capsys.readouterr().out == f"Judge report ({expected}) written to: {output}\n"
        text = output.read_text(encoding="utf-8")
    if expected == "json":
        assert json.loads(text)["tasks"][0]["verdict"] == "tie"
    elif expected == "csv":
        assert next(csv.DictReader(text.splitlines()))["verdict"] == "tie"
    else:
        assert text.startswith("# OpenBench AI LLM-judge report\n")


class _JudgeServer(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        verdict = _correctness_judge(body["messages"][-1]["content"]) if "[Question]" in body["messages"][-1]["content"] else "tie"
        data = json.dumps({"choices": [{"message": {"content": verdict}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def test_cli_judge_with_human_agreement(tmp_path, capsys):
    ds = tmp_path / "ds.jsonl"
    ds.write_text("\n".join(json.dumps({"id": f"r{i}", "category": "math", "prompt": f"What is {i}?", "expected_answer": str(i)}) for i in range(6)) + "\n")
    a = evaluate(ds, {f"r{i}": (str(i) if i < 3 else "x") for i in range(6)}, "model-a", tmp_path / "a.json", metrics=("exact_match",))
    b = evaluate(ds, {f"r{i}": (str(i) if i >= 3 else "y") for i in range(6)}, "model-b", tmp_path / "b.json", metrics=("exact_match",))
    prefs_dir = tmp_path / "prefs"
    assert main(["preferences", "create", "--results", str(a), str(b), "--output", str(prefs_dir)]) == 0
    key = json.loads((prefs_dir / "tasks.key.json").read_text())
    rows = [{"annotation_id": f"h{n}", "task_id": t["task_id"], "annotator_id": "h1",
             "outcome": _display(key, t, "A" if int(t["record_id"][1:]) < 3 else "B")} for n, t in enumerate(key["tasks"])]
    ann = tmp_path / "ann.jsonl"
    ann.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    server = ThreadingHTTPServer(("127.0.0.1", 0), _JudgeServer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        capsys.readouterr()
        code = main(["judge", "--results", str(a), str(b), "--judge-provider", "openai_compatible",
                     "--judge-endpoint", f"http://127.0.0.1:{server.server_port}/v1", "--judge-model", "judge-z",
                     "--key", str(prefs_dir / "tasks.key.json"), "--annotations", str(ann),
                     "--cache-dir", str(tmp_path / "cache"), "--resamples", "200", "--format", "json"])
    finally:
        server.shutdown()
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["position_consistency_rate"] == 1.0
    assert out["human_agreement"]["percent_agreement"] == 1.0 and out["human_agreement"]["tasks_compared"] == 6
    assert out["judge"]["provider"] == "openai_compatible"


# ---------------------------------------------------------------------------------------------
# #30: a one-item stratum stays descriptive with stratum tests
# ---------------------------------------------------------------------------------------------
def test_single_item_stratum_stays_descriptive(tmp_path):
    ds = tmp_path / "ds.jsonl"
    records = [{"id": f"item_{i:03d}", "category": "solo" if i == 1 else "many", "prompt": f"Q{i}?", "expected_answer": f"ans{i}"} for i in range(1, 9)]
    ds.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    a = evaluate(ds, correct_answers(8, wrong=(2, 5)), "model-a", tmp_path / "a.json", metrics=("exact_match",))
    b = evaluate(ds, correct_answers(8, wrong=(1, 3)), "model-b", tmp_path / "b.json", metrics=("exact_match",))
    arms = [ComparisonArm.from_evaluation_file(p) for p in (a, b)]
    out = PairedComparisonEngine().compare(arms, ["exact_match"], ComparisonPolicy(stratum_tests=True, **FAST)).to_dict()
    cats = out["comparisons"][0]["breakdowns"]["by_category"]
    assert cats["solo"]["sample_count"] == 1
    assert cats["solo"]["descriptive_only"] is True and cats["solo"].get("test") is None
    assert "descriptive_only" not in cats["many"] and cats["many"]["test"] is not None and cats["many"]["p_value"] is not None



# ---------------------------------------------------------------------------------------------
# Bounded continuous equivalence (#20)
# ---------------------------------------------------------------------------------------------
def test_bernstein_p_value_inverts_the_bound():
    from openbench.stats.bernstein import _half_width, tost_empirical_bernstein

    rng = random.Random(4)
    diffs = [max(-1.0, min(1.0, rng.gauss(0.01, 0.1))) for _ in range(800)]
    res = tost_empirical_bernstein(diffs, 0.08, 0.05)
    scaled = [(d + 1) / 2 for d in diffs]
    m = sum(scaled) / len(scaled)
    var = sum((x - m) ** 2 for x in scaled) / (len(scaled) - 1)
    # At delta = p_upper the upper bound sits exactly on the margin.
    gap = (0.08 + 1) / 2 - m
    assert _half_width(var, len(diffs), res.p_upper) == pytest.approx(gap, rel=1e-9)
    assert res.p_tost == max(res.p_lower, res.p_upper)
    assert res.result_unadjusted == ("established" if res.p_tost <= 0.05 else "not_established")
    assert res.interval.lower < res.estimate < res.interval.upper


def test_bernstein_edge_cases():
    from openbench.stats.bernstein import tost_empirical_bernstein

    zero = tost_empirical_bernstein([0.0] * 500, 0.05)
    assert zero.result_unadjusted == "established" and zero.p_tost < 0.05
    assert tost_empirical_bernstein([0.0] * 30, 0.05).result_unadjusted == "not_established"
    assert tost_empirical_bernstein([0.5], 0.05).reason == "insufficient_items"
    assert tost_empirical_bernstein([0.5, 1.5], 0.05).reason == "differences_outside_bounds"
    far = tost_empirical_bernstein([0.4] * 300, 0.05)
    assert far.p_upper == 1.0 and far.result_unadjusted == "not_established"
    with pytest.raises(ValueError):
        tost_empirical_bernstein([0.0, 0.0], 0.0)


def test_analyze_cli_bounded_equivalence(tmp_path, capsys):
    ds = write_dataset(tmp_path / "ds.jsonl", 600)
    a = evaluate(ds, correct_answers(600), "model-a", tmp_path / "a.json", metrics=("keyword_coverage",))
    b = evaluate(ds, correct_answers(600), "model-b", tmp_path / "b.json", metrics=("keyword_coverage",))
    base = ["analyze", "--results", str(a), str(b), "--metrics", "keyword_coverage", "--equivalence-margin", "0.05",
            "--resamples", "200", "--permutations", "199"]
    assert main(base) == 0
    eq = json.loads(capsys.readouterr().out)["comparisons"][0]["equivalence"]
    assert eq["procedure"] == "bootstrap_t" or eq["descriptive_only"] is True
    assert main(base + ["--continuous-equivalence", "empirical_bernstein"]) == 0
    out = json.loads(capsys.readouterr().out)
    eq = out["comparisons"][0]["equivalence"]
    assert eq["procedure"] == "empirical_bernstein_tost" and eq["familywise_result"] == "established"
    assert out["parameters"]["continuous_equivalence"] == "empirical_bernstein"
