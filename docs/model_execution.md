# Model Execution Engine

OpenBench AI `v0.3.0` introduces a provider-independent **Model Execution Engine** (`openbench.execution.ExecutionEngine`) that automates prompt dispatch, concurrency, retries, timeouts, deterministic caching, and telemetry collection while keeping evaluation completely decoupled.

---

## 1. Architectural Separation: Execution vs. Evaluation

- **`ExecutionEngine` (`openbench.execution`)**: Takes a `Dataset` and a `BaseModelAdapter`, sends prompts to the provider, handles transport errors/retries/caching, records per-request latency and token usage, and outputs a standard `ModelResponseSet` + `PerformanceReport`.
- **`Evaluator` (`openbench.evaluator`)**: Takes a `Dataset` and a `ModelResponseSet` (either produced live by `ExecutionEngine` or loaded from a static `.json` file via `ResponseLoader`) and computes deterministic evaluation metrics.

Users can always evaluate pre-generated JSON response files with `openbench evaluate` without invoking any model provider.

---

## 2. Execution Configuration (`ExecutionConfig`)

```python
from openbench.adapters import MockModelProvider
from openbench.dataset import DatasetLoader
from openbench.execution import ExecutionConfig, ExecutionEngine, RetryPolicy

dataset = DatasetLoader.load("datasets/reasoning/sample.jsonl")
provider = MockModelProvider(model_name="mock-v1")

exec_config = ExecutionConfig(
    timeout_seconds=30.0,
    concurrency=4,
    use_cache=True,
    estimate_missing_tokens=False,
    privacy_mode=False,
    retry_policy=RetryPolicy(
        max_attempts=3,
        initial_backoff_seconds=0.1,
        backoff_multiplier=2.0,
        max_backoff_seconds=2.0,
        retry_on_timeouts=True,
    ),
)

engine = ExecutionEngine(provider=provider, config=exec_config)
outcome = engine.execute_dataset(dataset)
print(f"Completed: {len(outcome.completed_record_ids)}, Failed: {len(outcome.failed_record_ids)}")
```

---

## 3. CLI Execution (`openbench run`)

```bash
# Execute using a YAML configuration profile
openbench run --config configs/mock_experiment.yaml

# Execute using direct CLI flags
openbench run   --dataset datasets/reasoning/sample.jsonl   --provider mock   --model mock-deterministic-v1   --concurrency 2   --timeout 15.0   --max-attempts 3
```

---

## 4. Managing the Response Cache

`openbench run` stores every response in a local cache (`.openbench_cache` by default, or `--cache-dir`), so repeated runs do not call the model again. These commands manage it:

```bash
openbench cache stats                                     # what is in the cache, and how old it is
openbench cache stats --format text                       # the same as a short readable summary
openbench cache prune --older-than 30 --dry-run           # report what would be removed
openbench cache prune --older-than 30                     # remove entries cached more than 30 days ago
openbench cache prune --model gpt-4o --dry-run            # preview removing one model's entries
openbench cache prune --provider openai --older-than 30   # combine filters (all must match)
openbench cache clear                                     # remove everything
```

`cache stats` prints JSON by default (`--format text` gives a short summary with one line per provider and model). It keeps the `enabled`, `cache_dir`, `stored_entries`, `session_hits` and `session_misses` keys and adds:

| Key | Meaning |
| --- | --- |
| `by_model` | Entry counts per provider and model, sorted by provider, then model. |
| `oldest_cached_at`, `newest_cached_at` | The earliest and latest readable `cached_at`; `null` when the cache is empty or no entry has a readable one. |
| `unknown_age_entries` | Entries whose `cached_at` is unreadable. |
| `file_size_bytes` | Size of the cache file; `0` when there is none. |

`stats` never creates the cache folder, and it reports a corrupted cache file as an error instead of a traceback. Run it before `prune` to see how much a given `--older-than` would affect.

`prune` requires at least one of `--older-than DAYS` (a positive integer), `--provider NAME`, or `--model NAME`. When several filters are given, an entry is removed only if it matches all of them. Entries whose `cached_at` is missing or unreadable are never removed when `--older-than` is active (the command reports how many it kept for that reason), but are removed if they match `--provider` and/or `--model` with no age filter. `--dry-run` only reports and changes nothing. All the cache commands accept `--cache-dir`.

`prune` takes the same cross-process lock as the run commands and swaps in the rewritten file in one step. If the cache file is corrupted, `prune` reports the first bad line and leaves the file untouched; use `openbench cache clear` to start over.
