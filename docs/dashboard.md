# Results anyone can read: HTML reports and the local dashboard (v0.5.0)

## Self-contained HTML reports

`openbench analyze` and `openbench rank` can write a single HTML file that opens in any browser,
with no internet connection and nothing to install:

```bash
openbench analyze --runs run_<a> grp_<b> --format html --output comparison.html
openbench rank --runs run_<a> run_<b> run_<c> --metric exact_match --format html --output ranking.html
```

- It is rendered from the same payload as the JSON and Markdown reports, so the numbers match.
- Differences and means are drawn with their confidence intervals as inline SVG.
- There are no scripts or external files. Every value, including model names, is HTML-escaped,
  and the page sets a Content-Security-Policy that blocks scripts.

## `openbench serve`: a read-only local dashboard

```bash
openbench serve --experiments-dir experiments --reports-dir reports --open
# opens the dashboard in your default browser
```

With `--host 0.0.0.0 --allow-remote --open`, the browser opens `127.0.0.1` at the
server's actual port. The terminal still prints the wildcard bind address and the local URL.
Wildcard IPv6 bind addresses use `[::1]` for the browser; other hosts stay unchanged.

| Page | Shows |
| --- | --- |
| `/` | All runs (status, model, items, scores, group) and repeat groups; saved reports if `--reports-dir` is given. |
| `/runs/<run_id>` | Run details, scores and every item: prompt, response, reference and metric scores. |
| `/groups/<grp_id>` | Variability of a repeat group, including how much of the variation comes from re-running. |
| `/reports/<file>` | A saved `.json` analysis rendered as HTML, or a saved `.html` report. |

Safety:
- **Read-only.** It only reads files; it answers `GET` and `HEAD` and refuses everything else.
  It never writes to the experiments or reports directory.
- **Local by default.** It listens on `127.0.0.1`. Listening on another address needs
  `--allow-remote`. Do that only on a network you trust, because anyone who can reach it can read
  your prompts and model outputs.
- **Escaped.** Prompts and model outputs are untrusted text and are always escaped. Pages carry a
  Content-Security-Policy that blocks scripts.
- Only the standard library is used (`http.server`).
