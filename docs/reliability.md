# Reliable Execution, Retries, Recovery & Caching

OpenBench AI `v0.3.0` is engineered for resilient execution across flaky local endpoints or long benchmark datasets.

---

## 1. Retry Policies & Exponential Backoff (`RetryPolicy`)

When a request fails with a transient error (`connection_error`, `rate_limit`, or `timeout` when `retry_on_timeouts=True`), `ExecutionEngine` retries up to `max_attempts` using exponential backoff:

$$\text{delay}(k) = \min\left(\text{max\_backoff}, \; \text{initial\_backoff} \times \text{multiplier}^{k - 1}\right)$$

Validation and configuration errors (`validation_error`) fail immediately without wasting retries.

### Rate limits (v0.5.0)

- **Requests per minute.** `--max-requests-per-minute N` (profile: `execution.max_requests_per_minute`) spaces requests evenly, one every `60 / N` seconds, across all concurrent workers, so the limit holds over any one-minute window. The time spent waiting is recorded as `failure_info.rate_limit_wait_seconds`.
- **HTTP 429 and `Retry-After`.** When a provider rejects a request with HTTP 429 and sends `Retry-After` (seconds or an HTTP date), the engine waits that long before the next attempt, capped at `execution.max_retry_after_seconds` (default 60). Without the header it uses the normal backoff. Such retries are counted in `failure_info.rate_limited_retries`.
- Both are off by default, and runs without them record exactly the same fields as v0.4.0.

---

## 2. Failure Classification

Every failed request is classified into one of five diagnostic categories in `performance.json` and `logs.jsonl`:
- `timeout`
- `connection_error`
- `rate_limit`
- `validation_error`
- `provider_error`

---

## 3. Interrupted Run Recovery (`openbench runs resume`)

If an execution run is interrupted or experiences partial provider failures (`status = "partial"`):
- Completed responses are saved in `experiments/run_<id>/responses.json`.
- Running `openbench runs resume run_<id>` reloads all previously completed items and dispatches only the remaining failed/missing dataset records.

---

## 4. Configuration-Aware Deterministic Response Cache (`ResponseCache`)

- Cached entries are keyed by the SHA-256 digest of `(provider_name, model_name, model_version, dataset_name, dataset_version, record_id, prompt_sha256, generation_config_sha256)`.
- Responses are **never** reused across different models, model revisions, prompts, dataset versions, or generation parameters (such as a changed `temperature` or `seed`).
- Disable caching at any time with `--no-cache` or `use_cache: false`, or clear the cache with `openbench cache clear`.
- **v0.4.0:**
  - Runs in a repeat group add `(repeat_group_id, repetition_index)` to the key. Repetitions and independent groups therefore never share cached responses, and a resumed repetition reuses only its own entries. Single-run keys are unchanged.
  - A malformed cache file is reported with the offending line and can always be removed with `openbench cache clear`.
  - The cache is safe for concurrent threads in one process.
- **v0.5.0: several processes.** Loads, appends and clears take an advisory file lock (`cache_entries.lock` next to the cache file; `fcntl.flock` on Linux and macOS, `msvcrt.locking` on Windows). Several OpenBench processes can therefore share one cache directory without losing or corrupting entries. If the lock cannot be taken within 30 seconds, the run stops with a message naming the lock file. Two processes may still both generate and store the same entry; the cache is not a distributed exactly-once store.

  See [repeated_runs.md](repeated_runs.md).
- **Important**: In-flight duplicate protection prevents concurrent workers within the same run from dispatching duplicate requests for the same record ID, but OpenBench AI does not claim distributed exactly-once execution when an external HTTP provider times out mid-generation.
