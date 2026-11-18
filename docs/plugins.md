# Plugins: metrics, providers and dataset sources

A Python package can add metrics, model providers and dataset sources to OpenBench AI without changing OpenBench itself. Installing the package is enough: OpenBench finds it through [entry points](https://packaging.python.org/en/latest/specifications/entry-points/).

> **Plugins are trusted code.** Installing a plugin runs its code with your permissions, just like any other Python package. Only install plugins you trust.

## Entry point groups

| Group | The entry point must be | Used by |
| --- | --- | --- |
| `openbench.metrics` | a subclass of `openbench.metrics.base.BaseMetric` | `--metrics NAME`, profiles, `openbench metrics list` |
| `openbench.providers` | a subclass of `openbench.adapters.base.BaseModelAdapter` | `--provider NAME`, `openbench providers list` |
| `openbench.datasets` | a callable `load(argument: str) -> openbench.dataset.Dataset` | `--dataset plugin:NAME[:ARGUMENT]` |

## A minimal plugin

`pyproject.toml`:

```toml
[project]
name = "openbench-shout"
version = "0.1.0"
dependencies = ["openbench-ai>=0.5"]

[project.entry-points."openbench.metrics"]
shout_match = "openbench_shout:ShoutMetric"

[project.entry-points."openbench.providers"]
echo = "openbench_shout:EchoAdapter"

[project.entry-points."openbench.datasets"]
numbers = "openbench_shout:load_numbers"
```

`openbench_shout/__init__.py`:

```python
from openbench.adapters.base import BaseModelAdapter
from openbench.dataset import Dataset, DatasetRecord
from openbench.evaluator import ResponseItem
from openbench.metrics.base import BaseMetric, MetricResult


class ShoutMetric(BaseMetric):
    name = "shout_match"
    description = "1 when the response equals the expected answer, ignoring case."

    def compute_item(self, record, response_text):
        ok = response_text.strip().upper() == record.expected_answer.strip().upper()
        return MetricResult(metric_name=self.name, score=1.0 if ok else 0.0, passed=ok)


class EchoAdapter(BaseModelAdapter):
    provider_name = "echo"
    description = "Answers with the expected answer (for testing)."
    is_mock = True

    def generate_one(self, record):
        return ResponseItem(id=record.id, response=record.expected_answer, latency_ms=0.0)


def load_numbers(argument):
    n = int(argument or 10)
    records = [
        DatasetRecord(id=f"n{i}", category="numbers", prompt=f"Say {i}", expected_answer=str(i))
        for i in range(n)
    ]
    return Dataset(name="numbers", version="1.0", records=records, source_path="plugin:numbers")
```

After `pip install .`:

```bash
openbench plugins list
openbench run --dataset plugin:numbers:20 --provider echo --model e --metrics shout_match
```

## How loading works

- **Lazily.** Built-in names never import plugins. A plugin group is imported the first time a name is not found among the built-ins, or when a listing command (`plugins list`, `providers list`, `metrics list`) runs.
- **Clear errors.** A plugin that fails to import, or points to the wrong kind of object, raises `PluginError` naming the plugin, its entry point and its package.
- **Provenance.** Runs made with a plugin provider record `generation_configuration.plugin` (entry point, package and version), so results stay traceable. Listings show which package each metric or provider came from.
- **Turning plugins off.** Set `OPENBENCH_DISABLE_PLUGINS=1` to skip discovery entirely.
- **Name clashes.** A plugin that uses a built-in name replaces the built-in for that process. Avoid this; prefix plugin names with your package name.

## Testing a plugin

Use the `mock` provider and small fixture datasets so tests need no network. OpenBench's own tests (`tests/test_v050_foundations.py`) build a throwaway plugin package on disk and load it this way.
