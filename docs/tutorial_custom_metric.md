# Write your own metric

Use an environment with OpenBench AI 0.5.0 or newer and pytest installed.
Create the following files in a new directory. The tutorial scores stored
synthetic responses and needs no model provider.

## Implement a binary metric

Save this as `tutorial_metric.py`. The metric returns 1 when the answer
matches after collapsing whitespace and ignoring case, and 0 otherwise.
It cannot judge whether two answers mean the same thing.

```python title="tutorial_metric.py"
from openbench.dataset import DatasetRecord
from openbench.metrics.base import BaseMetric, MetricResult


class NormalizedMatch(BaseMetric):
    name = "tutorial_normalized_match"
    version = "0.1.0"
    description = "Match the reference after ignoring case and whitespace."
    limitation_note = "Checks text equality after normalization, not meaning."
    score_kind = "binary"

    def compute_item(self, record: DatasetRecord, response_text: str) -> MetricResult:
        reference = " ".join(record.expected_answer.casefold().split())
        response = " ".join(response_text.casefold().split())
        matches = bool(reference) and response == reference
        return MetricResult(metric_name=self.name, score=float(matches), passed=matches)

    def config_summary(self) -> dict[str, str]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "normalization": "casefold and collapse whitespace",
        }
```

The name is the registry key. Keep it distinct from built-in metrics. The
version and configuration allow readers to identify which rule produced a
report. `BaseMetric.aggregate()` already averages scores over the full
dataset and counts missing responses as zero.

## Test and register it locally

Save this as `test_tutorial_metric.py`:

```python title="test_tutorial_metric.py"
import pytest

from openbench.dataset import DatasetRecord
from openbench.metrics import create_metric, register_metric
from tutorial_metric import NormalizedMatch


@pytest.mark.parametrize(
    ("reference", "answer", "score"),
    [
        ("New York", " new   YORK ", 1.0),
        ("New York", "New York City", 0.0),
        ("", "  ", 0.0),
    ],
)
def test_normalized_match(reference, answer, score):
    record = DatasetRecord("one", "example", "Give the place name.", reference)
    metric = NormalizedMatch()
    result = metric.compute_item(record, answer)
    assert result.score == score
    assert result.passed == bool(score)
    assert result.metric_name == metric.name


def test_register_metric():
    register_metric(NormalizedMatch)
    metric = create_metric("tutorial_normalized_match")
    assert metric.get_metadata().version == "0.1.0"
    assert metric.config_summary()["normalization"] == "casefold and collapse whitespace"
```

```bash
python -m pytest -q test_tutorial_metric.py
```

`register_metric()` applies only in the current Python process. The test shows
the same call you would use before creating an evaluator that selects the metric
by name.

## Install it as a plugin

Save this as `pyproject.toml` alongside `tutorial_metric.py`. The entry point
names the class, which OpenBench can discover in a new process.

```toml title="pyproject.toml"
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "openbench-tutorial-metric"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["openbench-ai>=0.5"]

[project.entry-points."openbench.metrics"]
tutorial_normalized_match = "tutorial_metric:NormalizedMatch"

[tool.setuptools]
py-modules = ["tutorial_metric"]
```

```bash
python -m pip install .
openbench metrics list
```

Look for `tutorial_normalized_match` in the listing. See [Plugins](plugins.md)
for discovery, name clashes and trust considerations.

## Check the stored scores

Save this as `make_inputs.py` to write one dataset and two sets of saved
responses in the current directory:

```python title="make_inputs.py"
import json
from pathlib import Path

records = [
    {"id": "ny", "category": "example", "prompt": "Say New York.", "expected_answer": "New York"},
    {"id": "mt", "category": "example", "prompt": "Say Montreal.", "expected_answer": "Montreal"},
    {"id": "os", "category": "example", "prompt": "Say Oslo.", "expected_answer": "Oslo"},
]
Path("dataset.jsonl").write_text(
    "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
)

for name, answers in {
    "baseline": ["NEW YORK", "Toronto", "oslo"],
    "challenger": ["new  york", "montreal", "Oslo"],
}.items():
    responses = [
        {"id": record["id"], "response": answer}
        for record, answer in zip(records, answers)
    ]
    Path(f"{name}-responses.json").write_text(
        json.dumps({"model_name": name, "model_version": "1.0", "responses": responses}),
        encoding="utf-8",
    )
```

```bash
python make_inputs.py
openbench evaluate --dataset dataset.jsonl --responses baseline-responses.json --metrics tutorial_normalized_match --output baseline.json
openbench evaluate --dataset dataset.jsonl --responses challenger-responses.json --metrics tutorial_normalized_match --output challenger.json
openbench analyze --results baseline.json challenger.json --metrics tutorial_normalized_match --format json --output comparison.json
openbench rank --results baseline.json challenger.json --metric tutorial_normalized_match --format json --output ranking.json
```

The baseline matches two of three references and the challenger matches all
three. `analyze` and `rank` read the stored item scores without computing them
again. This fixture checks the workflow. Three items are too few to draw a
conclusion about comparative model performance.

### Use the declared statistical kind

`analyze` and `rank` read the `score_kind` that the installed plugin declares,
so the metric above is treated as `binary` without any extra step.

Use `binary` only for item scores that are always 0 or 1. A single-run paired
comparison can then use the exact McNemar test. Use `bounded_continuous` for
fractional scores bounded by 0 and 1, and `unknown` when neither classification
fits. The declared kind guides the choice of statistical procedure without
changing the stored item scores. See
[Statistical methodology](statistical_methodology.md) for assumptions and
interpretation.
