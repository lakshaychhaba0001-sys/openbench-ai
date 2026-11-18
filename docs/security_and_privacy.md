# Security and Privacy Architecture

OpenBench AI `v0.3.1` provides layered security and privacy protections in `openbench.security`.

---

## 1. Key-Level and Value-Level Credential Redaction

`redact_sensitive_data()` and `sanitize_error_message()` protect sensitive credentials across both dictionary keys and string values:

1. **Key-Name Redaction**: Dictionary keys matching `api_key`, `access_token`, `auth_token`, `client_secret`, `private_key`, `secret`, `password`, `credential`, `authorization`, `bearer`, `token`, or `key` (while allowlisting legitimate metric/telemetry keys like `keyword_coverage`, `required_keys`, `cache_key`, and `token_usage`) are replaced with `"[REDACTED]"`.
2. **Value-Level String Redaction**: Even when a dictionary key is harmless (such as `{"endpoint": "https://example.com/api?token=abc123"}`), string values are scanned and redacted for:
   - URL query secrets: `?token=abc123`, `&api_key=abc123`, `&client_secret=...`, `&password=...`, `&sig=...`
   - Authorization headers: `Authorization: Bearer abc123`, `Basic ...`
   - Inline credential assignments: `password=example`, `client_secret=example`, `api_key=...`
   - Standard token prefixes: `sk-...`, `hf_...`, `ghp_...`, `github_pat_...`, `AKIA...`
   - URL-embedded credentials: `https://user:password@host`

### Runtime Execution vs. Stored Artifacts
- `redact_sensitive_data()` returns a new sanitized structure without mutating the caller's runtime dictionary in place, so live model providers continue to receive their valid runtime credentials during request execution.
- `ResponseCache` hashes and stores only the redacted generation configuration, ensuring raw credentials are never written to `.openbench_cache/` and rotating a token does not invalidate cached responses.
- **Limitation**: Pattern-based redaction covers common credential formats and key names, but cannot guarantee detection of arbitrary custom unformatted secret strings embedded in free-form text.


## v0.5.0 additions

- **API keys for hosted models.** The `openai_compatible` adapter reads the key from an
  environment variable (`--api-key-env`, default `OPENAI_API_KEY`) for each request. The key and
  the variable's name are never written to configs, caches, run directories, logs or reports.
- **Plugins are trusted code.** Installing an OpenBench plugin runs its code with your
  permissions. Set `OPENBENCH_DISABLE_PLUGINS=1` to turn plugin discovery off. See
  [plugins.md](plugins.md).
- **HTML reports** contain no scripts or external assets, escape every value, and set a
  Content-Security-Policy that blocks scripts.
- **The dashboard** (`openbench serve`) is read-only, listens on `127.0.0.1` unless
  `--allow-remote` is given, answers only `GET`/`HEAD`, escapes all prompts and outputs, sends a
  script-blocking Content-Security-Policy, and validates names to prevent path traversal.
- **The cache lock** (`cache_entries.lock`) is advisory and local; it coordinates OpenBench
  processes on one machine.
