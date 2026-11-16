"""Self-contained HTML reports (v0.5.0, issue #18).

``openbench analyze --format html`` and ``openbench rank --format html`` render the same payload
as the JSON and Markdown reports into one HTML file: no external assets, no scripts, no new
dependencies. Every value from the payload, including model names and model output, is escaped
with :func:`html.escape`, and the page sets a restrictive Content-Security-Policy.

Differences with confidence intervals are drawn as inline SVG bars.
"""

from __future__ import annotations

import html
import math
from typing import Any

from openbench.reporting_v040 import REPORT_DOC_LINKS, fmt, fmt_ci, fmt_p

CSP = "default-src 'none'; style-src 'unsafe-inline'; img-src data:"

STYLE = """
:root { --bg:#ffffff; --fg:#1d1d1f; --muted:#5f6368; --line:#d9dce1; --accent:#2f6fde; --warn:#b26a00; --zero:#9aa0a6; --soft:#f4f6f9; }
@media (prefers-color-scheme: dark) { :root { --bg:#16181c; --fg:#e8eaed; --muted:#a0a6ad; --line:#3a3f45; --accent:#7aa7ff; --warn:#f0b35a; --zero:#6b7178; --soft:#1f2227; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width: 1040px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 24px; margin: 0 0 4px; } h2 { font-size: 18px; margin: 32px 0 8px; } h3 { font-size: 15px; margin: 20px 0 6px; }
.meta { color: var(--muted); font-size: 13px; }
.banner { border:1px solid var(--warn); color: var(--warn); padding: 10px 12px; border-radius: 6px; margin: 16px 0; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; margin: 8px 0; }
th, td { border-bottom: 1px solid var(--line); padding: 6px 8px; text-align: left; vertical-align: top; }
th { background: var(--soft); font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
ul { padding-left: 20px; } li { margin: 2px 0; }
code { font-size: 12px; background: var(--soft); padding: 1px 4px; border-radius: 3px; }
svg text { fill: var(--muted); font-size: 11px; }
.footer { color: var(--muted); font-size: 12px; margin-top: 32px; }
"""


def esc(value: Any) -> str:
    if value is None:
        return "n/a"
    return html.escape(str(value), quote=True)


def _table(headers: list[str], rows: list[list[Any]], numeric: set[int] | None = None) -> str:
    numeric = numeric or set()
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    num_attr = ' class="num"'
    body = "".join(
        "<tr>" + "".join(f"<td{num_attr if i in numeric else ''}>{esc(c)}</td>" for i, c in enumerate(row)) + "</tr>"
        for row in rows
    )
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _ci_chart(items: list[tuple[str, float | None, float | None, float | None]]) -> str:
    """Horizontal dot-and-whisker chart of estimates with CIs, with a zero line."""
    usable = [(label, est, lo, hi) for label, est, lo, hi in items if est is not None and lo is not None and hi is not None]
    if not usable:
        return ""
    lo_all = min(min(lo, est) for _, est, lo, _ in usable)
    hi_all = max(max(hi, est) for _, est, _, hi in usable)
    lo_all, hi_all = min(lo_all, 0.0), max(hi_all, 0.0)
    span = (hi_all - lo_all) or 1.0
    pad = span * 0.05
    lo_all, hi_all = lo_all - pad, hi_all + pad
    width, label_w, row_h = 760, 260, 26
    plot_w = width - label_w - 20
    height = row_h * len(usable) + 30

    def x(v: float) -> float:
        return label_w + (v - lo_all) / (hi_all - lo_all) * plot_w

    parts = [f'<svg role="img" aria-label="Differences with confidence intervals" viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg">']
    zx = x(0.0)
    parts.append(f'<line x1="{zx:.1f}" y1="4" x2="{zx:.1f}" y2="{height - 22}" stroke="var(--zero)" stroke-dasharray="3 3"/>')
    for i, (label, est, lo, hi) in enumerate(usable):
        y = 14 + i * row_h
        short = label if len(label) <= 38 else label[:37] + "…"
        parts.append(f'<text x="0" y="{y + 4}">{esc(short)}</text>')
        parts.append(f'<line x1="{x(lo):.1f}" y1="{y}" x2="{x(hi):.1f}" y2="{y}" stroke="var(--accent)" stroke-width="2"/>')
        parts.append(f'<circle cx="{x(est):.1f}" cy="{y}" r="4" fill="var(--accent)"/>')
    for v in (lo_all + pad, 0.0, hi_all - pad):
        parts.append(f'<text x="{x(v):.1f}" y="{height - 6}" text-anchor="middle">{v:.3g}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _page(title: str, body: str) -> str:
    documentation = " · ".join(
        f'<a href="{esc(url)}">{esc(label)}</a>' for label, url in REPORT_DOC_LINKS
    )
    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<meta http-equiv=\"Content-Security-Policy\" content=\"{CSP}\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{esc(title)}</title><style>{STYLE}</style></head><body><main>{body}"
        "<p class=\"footer\">Generated by OpenBench AI. Every value is computed from the analysis payload; no model was called to make this page.</p>"
        f"<p>{documentation}</p>"
        "</main></body></html>\n"
    )


def _provenance(prov: dict[str, Any] | None) -> str:
    if not prov:
        return ""
    rows = [["OpenBench version", prov.get("openbench_version")], ["Python", prov.get("python_version")],
            ["Platform", prov.get("platform")], ["Generated at", prov.get("generated_at")],
            ["Analysis seed", prov.get("analysis_seed")], ["Parameters SHA-256", prov.get("parameters_sha256")],
            ["Provider calls", prov.get("provider_calls")]]
    return "<h2>Provenance</h2>" + _table(["Field", "Value"], rows)


def render_paired_html(p: dict[str, Any]) -> str:
    params = p["parameters"]
    level = f"{round(params['confidence_level'] * 100, 3):g}%"
    out = ["<h1>Paired model comparison</h1>",
           f"<p class=\"meta\">Analysis <code>{esc(p['analysis_id'])}</code> · mode {esc(p['analysis_mode'])} · "
           f"missing-data policy {esc(params['missing_policy'])} · correction {esc(params['correction'])} · alpha {esc(params['alpha'])} · seed {esc(p['analysis_seed'])}</p>"]
    if p["analysis_mode"] == "exploratory":
        out.append(f"<div class=\"banner\"><strong>{esc(p.get('banner'))}</strong><br>Reason given: {esc(p.get('exploratory_reason'))}</div>")
    out.append("<p>Difference = challenger − baseline; higher is better for every metric. "
               "A difference that is not significant is not evidence that the models are equivalent.</p>")
    out.append("<h2>Arms</h2>" + _table(
        ["Arm", "Model", "Version", "Provider", "Source", "Repetitions", "Mock"],
        [[a["arm_id"], a["model_name"], a["model_version"], a["model_provider"], a["source"], a["repetitions"], fmt(a["is_mock_provider"])] for a in p["arms"]],
        numeric={5},
    ))
    out.append("<h2>Results</h2>")
    chart = _ci_chart([
        (f"{c['metric']['name']}: {c['baseline']} → {c['challenger']}", c.get("point_estimate"),
         (c.get("confidence_interval") or {}).get("lower"), (c.get("confidence_interval") or {}).get("upper"))
        for c in p["comparisons"]
    ])
    if chart:
        out.append(f"<p class=\"meta\">Differences with {esc(level)} CIs (not multiplicity-adjusted); dashed line = no difference.</p>{chart}")
    out.append(_table(
        ["Metric", "Pair", "n", "Baseline mean", "Challenger mean", "Difference", f"{level} CI", "Test", "p", "Adjusted p"],
        [[c["metric"]["name"], f"{c['baseline']} → {c['challenger']}", c["sample_count"],
          fmt(c["arm_means"][c["baseline"]]), fmt(c["arm_means"][c["challenger"]]), fmt(c["point_estimate"]),
          fmt_ci(c.get("confidence_interval")), (c.get("test") or {}).get("test") or "n/a",
          fmt_p(c.get("p_value")), fmt_p(c.get("adjusted_p_value"))] for c in p["comparisons"]],
        numeric={2, 3, 4, 5, 8, 9},
    ))
    for c in p["comparisons"]:
        items = [esc(c.get("interpretation", ""))]
        wlt = c["wins_losses_ties"]
        items.append(f"Wins / losses / ties (challenger vs baseline): {wlt['challenger_better']} / {wlt['baseline_better']} / {wlt['ties']}")
        eq = c.get("equivalence")
        if eq:
            items.append(esc(f"Equivalence (margin {fmt(eq.get('margin'))}, {eq.get('procedure')}): familywise {eq.get('familywise_result')}; "
                             f"unadjusted {eq.get('result_unadjusted')}" + (f"; reason {eq['reason']}" if eq.get("reason") else "")))
        for w in c.get("warnings", []):
            items.append(f"Warning <code>{esc(w['code'])}</code>: {esc(w['message'])}")
        for key, reason in sorted(c.get("unavailable", {}).items()):
            items.append(f"Unavailable <code>{esc(key)}</code>: {esc(reason)}")
        out.append(f"<h3>{esc(c['metric']['name'])}: {esc(c['baseline'])} → {esc(c['challenger'])}</h3><ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>")
        rows = c.get("top_disagreements") or c.get("disagreements") or []
        if rows:
            out.append(_table(["Record", "Baseline", "Challenger", "Difference"],
                              [[r.get("id"), fmt(r.get("baseline")), fmt(r.get("challenger")), fmt(r.get("difference"))] for r in rows[:20]],
                              numeric={1, 2, 3}))
    out.append("<h2>Compatibility</h2>")
    for pair in p["pairs"]:
        comp = pair["compatibility"]
        status = "compatible for inference" if comp["compatible_for_inference"] else "NOT compatible for inference"
        entries = [f"{label} <code>{esc(e['code'])}</code>: {esc(e['message'])}"
                   for label, key in (("Hard incompatibility", "hard_incompatibilities"), ("Warning", "warnings"), ("Info", "info"))
                   for e in comp[key]]
        out.append(f"<p><strong>{esc(pair['baseline'])} vs {esc(pair['challenger'])}</strong>: {esc(status)}</p>")
        if entries:
            out.append("<ul>" + "".join(f"<li>{e}</li>" for e in entries) + "</ul>")
    limitations = p.get("limitations") or []
    if limitations:
        out.append("<h2>Limitations</h2><ul>" + "".join(f"<li>{esc(x)}</li>" for x in limitations) + "</ul>")
    out.append(_provenance(p.get("provenance")))
    return _page("OpenBench paired comparison", "".join(out))


def render_ranking_html(p: dict[str, Any]) -> str:
    out = ["<h1>Model ranking</h1>",
           f"<p class=\"meta\">Metric <code>{esc(p['metric'])}</code> ({esc(p['score_kind'])}) · {esc(p['items'])} items · "
           f"Holm-corrected pairwise tests, alpha {esc(p['alpha'])}</p>"]
    if p.get("banner"):
        out.append(f"<div class=\"banner\"><strong>{esc(p['banner'])}</strong></div>")
    chart = _ci_chart([(m["model"], m["mean"], (m["mean_confidence_interval"] or {}).get("lower"),
                        (m["mean_confidence_interval"] or {}).get("upper")) for m in p["models"]])
    if chart:
        out.append(f"<p class=\"meta\">Mean score with {esc(round(p['confidence_level'] * 100))}% CI.</p>{chart}")
    out.append(_table(
        ["Rank", "Rank interval", "Model", "Mean", "CI", "Tied with top", "p(best), bootstrap"],
        [[fmt(m["observed_rank"], 1), f"{m['rank_interval'][0]}–{m['rank_interval'][1]}", m["model"], fmt(m["mean"]),
          fmt_ci(m["mean_confidence_interval"]), fmt(m["tied_with_top"]), fmt(m["p_best_bootstrap"], 3)] for m in p["models"]],
        numeric={0, 3, 6},
    ))
    out.append(f"<p>{esc(p['interpretation'])}</p>")
    out.append("<h2>Pairwise tests</h2>" + _table(
        ["Model A", "Model B", "Difference (B − A)", "Test", "p", "Adjusted p", "Significant"],
        [[t["model_a"], t["model_b"], fmt(t["mean_difference_b_minus_a"]), t["test"], fmt_p(t["p_value"]),
          fmt_p(t.get("adjusted_p_value")), fmt(t.get("significant"))] for t in p["pairwise_tests"]],
        numeric={2, 4, 5},
    ))
    if p.get("warnings"):
        out.append("<h2>Warnings</h2><ul>" + "".join(f"<li><code>{esc(w['code'])}</code>: {esc(w['message'])}</li>" for w in p["warnings"]) + "</ul>")
    out.append(_provenance(p.get("provenance")))
    return _page("OpenBench model ranking", "".join(out))


def render_html(payload: dict[str, Any]) -> str:
    kind = payload.get("analysis")
    if kind == "paired_comparison":
        return render_paired_html(payload)
    if kind == "model_ranking":
        return render_ranking_html(payload)
    if kind == "repeated_run_variability":
        return render_generic_html("Repeated-run variability", payload)
    return render_generic_html(str(kind or "OpenBench report"), payload)


def render_generic_html(title: str, payload: dict[str, Any]) -> str:
    """Escaped, readable dump of any payload (used for report types without a dedicated page)."""
    import json

    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    return _page(title, f"<h1>{esc(title)}</h1><div class=\"scroll\"><pre>{esc(text)}</pre></div>")


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
