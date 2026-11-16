"""Renderers for v0.5.0 reports: model ranking, sample-size plans and LLM-judge results.

Same rules as :mod:`openbench.reporting_v040`: JSON is the source of truth; Markdown and CSV
are rendered from the same payload; CSV cells are protected against formula injection; model
text is escaped.
"""

from __future__ import annotations

from typing import Any

from openbench.exceptions import ConfigurationError
from openbench.reporting_v040 import (
    REPORT_DOC_MARKDOWN,
    _csv,
    _provenance_md,
    _table,
    fmt,
    fmt_ci,
    fmt_p,
    to_json,
)


def _check_format(fmt_name: str) -> str:
    if fmt_name not in ("json", "markdown", "csv"):
        raise ConfigurationError(f"Unsupported format '{fmt_name}'")
    return fmt_name


def render_ranking(payload: dict[str, Any], fmt_name: str) -> str:
    _check_format(fmt_name)
    if fmt_name == "json":
        return to_json(payload)
    headers = ["Rank", "Rank interval", "Model", "Mean", "95% CI", "Tied with top", "p(best), bootstrap"]
    rows = [
        [
            fmt(m["observed_rank"], 1),
            f"{m['rank_interval'][0]}-{m['rank_interval'][1]}",
            m["model"],
            fmt(m["mean"]),
            fmt_ci(m["mean_confidence_interval"]),
            fmt(m["tied_with_top"]),
            fmt(m["p_best_bootstrap"], 3),
        ]
        for m in payload["models"]
    ]
    if fmt_name == "csv":
        return _csv(headers, rows)
    lines = ["# OpenBench AI model ranking", ""]
    if payload.get("banner"):
        lines += [f"> **{payload['banner']}**", ""]
    lines += [
        f"- Metric: `{payload['metric']}` ({payload['score_kind']}); items: {payload['items']} "
        f"(excluded as incomplete: {payload.get('items_excluded_incomplete', 0)})",
        f"- Rank intervals: pairwise paired tests with Holm correction, alpha = {payload['alpha']}",
        "- Higher scores rank higher; equal means share the average rank.",
        "",
    ]
    lines += _table(headers, rows)
    lines += ["", payload["interpretation"], ""]
    lines += ["## Pairwise tests", ""]
    lines += _table(
        ["Model A", "Model B", "Mean difference (B - A)", "Test", "p", "Adjusted p (Holm)", "Significant"],
        [[t["model_a"], t["model_b"], fmt(t["mean_difference_b_minus_a"]), t["test"], fmt_p(t["p_value"]),
          fmt_p(t.get("adjusted_p_value")), fmt(t.get("significant"))] for t in payload["pairwise_tests"]],
    )
    if payload.get("warnings"):
        lines += ["", "## Warnings", ""] + [f"- `{w['code']}`: {w['message']}" for w in payload["warnings"]]
    lines += [""] + _provenance_md(payload.get("provenance"))
    lines.append(REPORT_DOC_MARKDOWN)
    return "\n".join(lines) + "\n"


def render_plan(payload: dict[str, Any], fmt_name: str) -> str:
    _check_format(fmt_name)
    if fmt_name == "json":
        return to_json(payload)
    rows = [[k, fmt(v) if isinstance(v, float) else v] for k, v in payload.items() if not isinstance(v, (dict, list))]
    if fmt_name == "csv":
        return _csv(["Field", "Value"], rows)
    lines = ["# OpenBench AI sample-size plan", ""]
    if payload.get("items_needed") is not None:
        lines += [f"**{payload['items_needed']} paired items** are needed to detect a difference of "
                  f"{fmt(payload.get('difference'))} with {fmt(payload.get('target_power'), 2)} power at alpha = {payload.get('alpha')}.", ""]
    elif payload.get("minimum_detectable_difference") is not None:
        lines += [f"With **{payload['items']} items**, the smallest difference detectable with "
                  f"{fmt(payload.get('target_power'), 2)} power is **{fmt(payload['minimum_detectable_difference'])}**.", ""]
    lines += _table(["Field", "Value"], rows)
    if payload.get("assumption"):
        lines += ["", f"Assumption: {payload['assumption']}."]
    lines += ["", "Plans are approximate: they assume the items you will use behave like the inputs given here.", ""]
    lines.append(REPORT_DOC_MARKDOWN)
    return "\n".join(lines) + "\n"


def render_judge(payload: dict[str, Any], fmt_name: str) -> str:
    _check_format(fmt_name)
    if fmt_name == "json":
        return to_json(payload)
    if fmt_name == "csv":
        return _csv(
            ["record_id", "category", "verdict_order_ab", "verdict_order_ba", "status", "verdict", "length_a", "length_b"],
            [[t["record_id"], t["category"], t["verdict_order_ab"], t["verdict_order_ba"], t["status"], t["verdict"],
              t["response_lengths"]["A"], t["response_lengths"]["B"]] for t in payload["tasks"]],
        )
    j = payload["judge"]
    lines = [
        "# OpenBench AI LLM-judge report",
        "",
        f"> {payload['limitations']}",
        "",
        f"- Judge: `{j['provider']}:{j['model_name']}` (prompt {j['prompt_version']}, sha256 {j['prompt_sha256'][:12]})",
        f"- Model A: `{payload['model_a']['source_id']}` ({payload['model_a']['model_name']}); "
        f"model B: `{payload['model_b']['source_id']}` ({payload['model_b']['model_name']})",
        f"- Pairs judged: {payload['pairs_judged']} ({payload['judge_calls']} judge calls, {payload['cached_judge_calls']} from cache)",
        "",
    ]
    lines += _table(
        ["Measure", "Value", "95% CI"],
        [
            ["Win rate of A (A = 1, tie = 0.5)", fmt(payload["win_rate_a"]), fmt_ci(payload["win_rate_a_confidence_interval"])],
            ["Position consistency (same verdict in both orders)", fmt(payload["position_consistency_rate"]), ""],
            ["Share of decisive verdicts for the first-shown response", fmt(payload["position_bias"]["share_first_shown_wins"]),
             fmt_ci(payload["position_bias"]["confidence_interval"])],
            ["Share of consistent decisive wins by the longer response", fmt(payload["length_bias"]["share_longer_wins"]),
             fmt_ci(payload["length_bias"]["confidence_interval"])],
        ],
    )
    c = payload["outcome_counts"]
    lines += ["", f"Outcomes: A {c['A']}, B {c['B']}, tie {c['tie']} (including {payload['inconsistent_pairs_scored_as_tie']} inconsistent pairs scored as ties).", ""]
    agreement = payload.get("human_agreement")
    if agreement:
        lines += ["## Agreement with human annotations", ""]
        lines += _table(
            ["Tasks compared", "Percent agreement", "95% CI", "Cohen's kappa vs human majority"],
            [[agreement["tasks_compared"], fmt(agreement["percent_agreement"]),
              fmt_ci(agreement["percent_agreement_confidence_interval"]), fmt(agreement["cohens_kappa_vs_human_majority"])]],
        )
        lines += ["", agreement["note"], ""]
    if payload.get("warnings"):
        lines += ["## Warnings", ""] + [f"- `{w['code']}`: {w['message']}" for w in payload["warnings"]] + [""]
    lines += ["## Method", "", payload["methodology"], ""]
    return "\n".join(lines) + "\n"


def render_agreement_md(agreement: dict[str, Any]) -> list[str]:
    lines = [f"## Agreement of `{agreement['automatic']}` with the human majority", ""]
    lines += _table(
        ["Tasks compared", "Percent agreement", "95% CI", "Cohen's kappa", "Decisive-task agreement"],
        [[agreement["tasks_compared"], fmt(agreement["percent_agreement"]), fmt_ci(agreement["percent_agreement_confidence_interval"]),
          fmt(agreement["cohens_kappa_vs_human_majority"]), fmt(agreement["decisive_agreement"])]],
    )
    lines += ["", agreement["note"], ""]
    return lines
