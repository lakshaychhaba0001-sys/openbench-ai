"""``openbench serve``: a read-only local dashboard (v0.5.0, issue #36).

A small standard-library web server (``http.server``) that lists stored runs and repeat groups,
shows each run's scores and per-item results, a group's variability summary, and analysis
reports saved as JSON or HTML in a reports directory.

Safety rules:

* **Read-only.** It only reads the experiments and reports directories; it never writes,
  creates or deletes files (only ``GET`` and ``HEAD`` are answered).
* **Local by default.** It listens on ``127.0.0.1``. Any other address needs ``--allow-remote``.
* **Escaped output.** Dataset prompts and model answers are untrusted text; every value is
  HTML-escaped and every page carries a Content-Security-Policy that blocks scripts.
* **No path traversal.** Run, group and report names are validated, and report files must
  resolve inside the reports directory.
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlencode, urlparse

from openbench.experiments import ExperimentManager
from openbench.html_report import CSP, _page, _table, esc, render_html
from openbench.reporting_v040 import fmt

RUN_ID_RE = re.compile(r"^run_[A-Za-z0-9_-]{1,80}$")
GROUP_ID_RE = re.compile(r"^grp_[A-Za-z0-9_-]{1,80}$")
REPORT_RE = re.compile(r"^[A-Za-z0-9_.-]{1,160}\.(json|html)$")
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
MAX_ITEMS_SHOWN = 500
MAX_TEXT = 600


def _short(text: Any) -> str:
    s = "" if text is None else str(text)
    return s if len(s) <= MAX_TEXT else s[:MAX_TEXT] + " …"


class Dashboard:
    """Builds the pages; separate from the HTTP handler so it can be tested directly."""

    def __init__(self, experiments_dir: str | Path, reports_dir: str | Path | None = None) -> None:
        self.manager = ExperimentManager(experiments_root=experiments_dir)
        self.experiments_dir = Path(experiments_dir)
        self.reports_dir = Path(reports_dir) if reports_dir else None

    # -- pages ---------------------------------------------------------------------------------
    def index(self, status: str = "", provider: str = "", group: str = "") -> str:
        status = status.strip()
        provider = provider.strip()
        group = group.strip()
        if self.experiments_dir.exists():
            runs = self.manager.list_runs(status=status or None, provider=provider or None,
                                          repeat_group_id=group or None)
            all_runs = self.manager.list_runs()
        else:
            runs = []
            all_runs = []
        groups = sorted({r.repeat_group_id for r in runs if r.repeat_group_id})
        statuses = sorted({r.status for r in all_runs if r.status})
        providers = sorted({r.model_provider for r in all_runs if r.model_provider})
        all_groups = sorted({r.repeat_group_id for r in all_runs if r.repeat_group_id})

        def filter_url(**changes: str) -> str:
            params = {"status": status, "provider": provider, "group": group}
            params.update(changes)
            query = urlencode({k: v for k, v in params.items() if v})
            return f"/?{query}" if query else "/"

        def link_line(label: str, key: str, values: list[str]) -> str:
            links = " · ".join(
                f"<a href=\"{esc(filter_url(**{key: v}))}\">{esc(v)}</a>" for v in values
            )
            return f"<p class=\"meta\">{label}: {links}</p>"
        rows = []
        for r in runs:
            scores = ", ".join(f"{k}={fmt(v)}" for k, v in sorted((r.aggregate_scores or {}).items()))
            rows.append([r.run_id, r.status, f"{r.model_provider}:{r.model_name}" + (" [mock]" if r.is_mock_provider else ""),
                         f"{r.completed_items}/{r.total_items}", scores, r.repeat_group_id or "-", r.created_at])
        body = ["<h1>OpenBench AI dashboard</h1>",
                f"<p class=\"meta\">Read-only view of <code>{esc(self.experiments_dir)}</code>. Nothing here changes your files.</p>",
                f"<h2>Runs ({len(runs)})</h2>"]
        active = []
        if status:
            active.append(f"status={esc(status)}")
        if provider:
            active.append(f"provider={esc(provider)}")
        if group:
            active.append(f"group={esc(group)}")
        if active:
            body.append(f"<p class=\"meta\">Active filters: {', '.join(active)}."
                        f" <a href=\"/\">Show all runs</a></p>")
        if statuses:
            body.append(link_line("Status", "status", statuses))
        if providers:
            body.append(link_line("Provider", "provider", providers))
        if all_groups:
            body.append(link_line("Group", "group", all_groups))
        if rows:
            body.append(self._linked_table(["Run", "Status", "Model", "Items", "Scores", "Group", "Created"], rows,
                                           link_cols={0: "/runs/", 5: "/groups/"}))
        else:
            body.append("<p>No runs found.</p>")
        body.append(f"<h2>Repeat groups ({len(groups)})</h2>")
        if groups:
            body.append("<ul>"+"".join(
                f"<li><a href=\"/groups/{esc(g)}\">{esc(g)}</a>"
                f" (<a href=\"{esc(filter_url(group=g))}\">show only this group's runs</a>)</li>"
                for g in groups
            ) + "</ul>")
        else:
            body.append("<p>No repeat groups.</p>")
        if self.reports_dir is not None:
            reports = self._reports()
            body.append(f"<h2>Reports ({len(reports)})</h2>")
            body.append("<ul>" + "".join(f"<li><a href=\"/reports/{esc(n)}\">{esc(n)}</a></li>" for n in reports) + "</ul>"
                        if reports else f"<p>No .json or .html reports in <code>{esc(self.reports_dir)}</code>.</p>")
        return _page("OpenBench dashboard", "".join(body))

    def run(self, run_id: str, page: str = "1") -> str | None:
        if not RUN_ID_RE.match(run_id):
            return None
        try:
            rec = self.manager.load_run_record(run_id)
            ev = self.manager.load_evaluation(run_id)
        except (KeyError, FileNotFoundError, ValueError):
            return None
        page_count = max(1, (len(ev.individual_results) + MAX_ITEMS_SHOWN - 1) // MAX_ITEMS_SHOWN)
        try:
            page_number = int(page)
        except ValueError:
            page_number = 1
        if not 1 <= page_number <= page_count:
            page_number = 1
        start = (page_number - 1) * MAX_ITEMS_SHOWN
        meta = [["Status", rec.status], ["Model", f"{rec.model_provider}:{rec.model_name} ({rec.model_version})"],
                ["Mock provider", fmt(rec.is_mock_provider)], ["Dataset", f"{rec.dataset_identifier} v{rec.dataset_version}"],
                ["Dataset checksum", rec.dataset_checksum], ["Items", f"{rec.completed_items}/{rec.total_items} completed"],
                ["Repeat group", rec.repeat_group_id or "-"], ["Created", rec.created_at], ["OpenBench version", rec.openbench_version]]
        scores = [[k, fmt(v)] for k, v in sorted((rec.aggregate_scores or {}).items())]
        items = []
        for item in ev.individual_results[start:start + MAX_ITEMS_SHOWN]:
            metric_text = ", ".join(f"{k}={fmt(v.score)}" for k, v in sorted(item.metrics.items()))
            items.append([item.id, item.category, _short(item.prompt), _short(item.model_response), _short(item.expected_answer), metric_text])
        body = [f"<p><a href=\"/\">← all runs</a></p><h1>Run {esc(rec.run_id)}</h1>",
                _table(["Field", "Value"], meta), "<h2>Scores</h2>", _table(["Metric", "Score"], scores, numeric={1}),
                f"<h2>Items ({len(ev.individual_results)})</h2>",
                _table(["ID", "Category", "Prompt", "Response", "Reference", "Metrics"], items)]
        if page_count > 1:
            links = []
            if page_number > 1:
                links.append(f'<a href="/runs/{esc(run_id)}?page={page_number - 1}">Previous</a>')
            links.append(f"<span>Page {page_number} of {page_count}</span>")
            if page_number < page_count:
                links.append(f'<a href="/runs/{esc(run_id)}?page={page_number + 1}">Next</a>')
            body.append('<nav aria-label="Item pages">' + " ".join(links) + "</nav>")
        return _page(f"Run {rec.run_id}", "".join(body))

    def group(self, group_id: str) -> str | None:
        if not GROUP_ID_RE.match(group_id):
            return None
        try:
            evaluations = self.manager.load_group_evaluations(group_id)
            members = self.manager.list_runs(repeat_group_id=group_id)
        except (KeyError, FileNotFoundError, ValueError):
            return None
        from openbench.stats import decompose_variability

        analysis = decompose_variability(evaluations, resamples=1_000, arm_label=group_id)
        rows = []
        for metric, m in analysis["metrics"].items():
            vc = m.get("variance_components") or {}
            ci = m["item_variation"]["question_level_confidence_interval"]
            rows.append([metric, fmt(m["item_variation"]["mean_of_item_means"]), f"[{fmt(ci.get('lower'))}, {fmt(ci.get('upper'))}]",
                         fmt((vc.get("components") or {}).get("share_repetition")), vc.get("summary", "n/a")])
        member_rows = [[r.run_id, r.repetition_index, r.status, f"{r.completed_items}/{r.total_items}"]
                       for r in sorted(members, key=lambda r: r.repetition_index)]
        body = [f"<p><a href=\"/\">← all runs</a></p><h1>Repeat group {esc(group_id)}</h1>",
                f"<p class=\"meta\">{len(evaluations)} repetition(s). Variability computed on the fly with 1,000 bootstrap resamples (seed 0).</p>",
                "<h2>Variability</h2>",
                _table(["Metric", "Mean of item means", "95% CI (questions)", "Share from re-running", "Summary"], rows),
                "<h2>Members</h2>", self._linked_table(["Run", "Repetition", "Status", "Items"], member_rows, link_cols={0: "/runs/"})]
        return _page(f"Group {group_id}", "".join(body))

    def report(self, name: str) -> tuple[str, str] | None:
        """Return ``(content_type, body)`` for a report file, or ``None``."""
        if self.reports_dir is None or not REPORT_RE.match(name):
            return None
        root = self.reports_dir.resolve()
        path = (root / name).resolve()
        if path.parent != root or not path.is_file():
            return None
        text = path.read_text(encoding="utf-8", errors="replace")
        if name.endswith(".html"):
            return "text/html; charset=utf-8", text
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return "text/html; charset=utf-8", _page(name, f"<h1>{esc(name)}</h1><p>Not valid JSON.</p>")
        if not isinstance(payload, dict):
            return None
        return "text/html; charset=utf-8", render_html(payload)

    # -- helpers -------------------------------------------------------------------------------
    def _reports(self) -> list[str]:
        if self.reports_dir is None or not self.reports_dir.is_dir():
            return []
        return sorted(p.name for p in self.reports_dir.iterdir() if p.is_file() and REPORT_RE.match(p.name))

    @staticmethod
    def _linked_table(headers: list[str], rows: list[list[Any]], link_cols: dict[int, str]) -> str:
        head = "".join(f"<th>{esc(h)}</th>" for h in headers)
        body = []
        for row in rows:
            cells = []
            for i, c in enumerate(row):
                if i in link_cols and c not in (None, "-"):
                    cells.append(f"<td><a href=\"{link_cols[i]}{esc(c)}\">{esc(c)}</a></td>")
                else:
                    cells.append(f"<td>{esc(c)}</td>")
            body.append("<tr>" + "".join(cells) + "</tr>")
        return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def make_handler(dashboard: Dashboard) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "OpenBenchDashboard"
        sys_version = ""

        def log_message(self, fmt_str: str, *args: Any) -> None:  # quiet by default
            pass

        def _send(self, status: int, body: str, content_type: str = "text/html; charset=utf-8", head: bool = False) -> None:
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Content-Security-Policy", CSP)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if not head:
                self.wfile.write(data)

        def _route(self, head: bool = False) -> None:
            target = urlparse(self.path)
            path = unquote(target.path)
            try:
                if path in ("/", "/index.html"):
                    query = parse_qs(target.query, keep_blank_values=True)
                    return self._send(200, dashboard.index(
                        status=query.get("status", [""])[0],
                        provider=query.get("provider", [""])[0],
                        group=query.get("group", [""])[0],
                    ), head=head)
                if path.startswith("/runs/"):
                    requested_page = parse_qs(target.query, keep_blank_values=True).get("page", ["1"])[0]
                    page = dashboard.run(path[len("/runs/"):], requested_page)
                elif path.startswith("/groups/"):
                    page = dashboard.group(path[len("/groups/"):])
                elif path.startswith("/reports/"):
                    found = dashboard.report(path[len("/reports/"):])
                    if found:
                        return self._send(200, found[1], found[0], head=head)
                    page = None
                else:
                    page = None
            except Exception as exc:  # never leak a traceback to the browser
                return self._send(500, _page("Error", f"<h1>Error</h1><p>{esc(type(exc).__name__)}</p>"), head=head)
            if page is None:
                return self._send(404, _page("Not found", "<h1>Not found</h1><p><a href=\"/\">Back to the dashboard</a></p>"), head=head)
            return self._send(200, page, head=head)

        def do_GET(self) -> None:  # noqa: N802
            self._route()

        def do_HEAD(self) -> None:  # noqa: N802
            self._route(head=True)

        def _refuse(self) -> None:
            self._send(405, _page("Read-only", "<h1>Read-only</h1><p>This dashboard only answers GET requests.</p>"))

        do_POST = do_PUT = do_DELETE = do_PATCH = _refuse  # noqa: N815

    return Handler


def create_server(experiments_dir: str | Path, reports_dir: str | Path | None = None, host: str = "127.0.0.1",
                  port: int = 8765, allow_remote: bool = False) -> ThreadingHTTPServer:
    if host not in LOOPBACK and not allow_remote:
        raise ValueError(f"Refusing to listen on {host}: the dashboard is local-only unless --allow-remote is given")
    return ThreadingHTTPServer((host, port), make_handler(Dashboard(experiments_dir, reports_dir)))


def serve_in_thread(server: ThreadingHTTPServer) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread
