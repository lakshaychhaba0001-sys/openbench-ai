"""Unit tests for OpenBench AI evaluation metrics and normalization."""

from __future__ import annotations

import pytest

from openbench.dataset import DatasetRecord
from openbench.exceptions import MetricError
from openbench.metrics import (
    AccuracyMetric,
    BaseMetric,
    ExactMatchMetric,
    KeywordCoverageMetric,
    MetricResult,
    NormalizationConfig,
    create_metric,
    normalize_text,
    register_metric,
)


def _make_record(
    expected: str = "180",
    keywords: list[str] | None = None,
    disallow_substring: bool = False,
) -> DatasetRecord:
    meta: dict = {"difficulty": "easy"}
    if keywords is not None:
        meta["keywords"] = keywords
    if disallow_substring:
        meta["disallow_substring"] = True
    return DatasetRecord(
        id="test_001",
        category="reasoning",
        prompt="Sample prompt?",
        expected_answer=expected,
        metadata=meta,
    )


def test_normalization_default_preserves_punctuation_and_collapses_whitespace() -> None:
    """Default NormalizationConfig strips/collapses whitespace, lowercases, and keeps punctuation."""
    raw = "   Hello,   OpenBench-AI!  \n\t v0.1.0  "
    normalized = normalize_text(raw)
    assert normalized == "hello, openbench-ai! v0.1.0"


def test_normalization_configurable_case_and_punctuation() -> None:
    """NormalizationConfig respects case_sensitive=True and strip_punctuation=True."""
    cfg = NormalizationConfig(
        strip_whitespace=True,
        case_sensitive=True,
        collapse_whitespace=False,
        strip_punctuation=True,
    )
    assert normalize_text(" Answer: 42! ", cfg) == "Answer 42"


def test_normalization_unicode_nfkc() -> None:
    """NFKC normalizes full-width, Japanese, and compatibility characters."""
    cfg = NormalizationConfig(unicode_normalization="NFKC")
    assert normalize_text("\uff21\uff22\uff23\uff11\uff12\uff13", cfg) == "abc123"
    assert normalize_text("\uff83\uff7d\uff84", cfg) == "\u30c6\u30b9\u30c8"
    assert normalize_text("cafe\u0301 \ufb01", cfg) == "caf\u00e9 fi"


def test_normalization_unicode_nfc_hindi_and_latin() -> None:
    """NFC composes Hindi/Devanagari and accented Latin characters."""
    cfg = NormalizationConfig(unicode_normalization="NFC")
    assert normalize_text("cafe\u0301", cfg) == "caf\u00e9"
    assert normalize_text("\u0915\u093c", cfg) == "\u0958"
    assert (
        normalize_text("\u0928\u092e\u0938\u094d\u0924\u0947", cfg)
        == "\u0928\u092e\u0938\u094d\u0924\u0947"
    )


def test_normalization_strip_unicode_punctuation() -> None:
    """strip_unicode_punctuation removes Unicode P* punctuation when enabled."""
    cfg = NormalizationConfig(strip_punctuation=True, strip_unicode_punctuation=True)
    raw = "\u201cHello\u2014world!\u201d \u0928\u092e\u0938\u094d\u0924\u0947\u0964"
    assert normalize_text(raw, cfg) == "helloworld \u0928\u092e\u0938\u094d\u0924\u0947"


def test_normalization_config_to_dict_and_defaults() -> None:
    """to_dict includes Unicode fields and default config preserves prior behavior."""
    default_cfg = NormalizationConfig()
    assert default_cfg.unicode_normalization is None
    assert default_cfg.strip_unicode_punctuation is False
    assert (
        normalize_text("  \u201cHello\u2014OpenBench!\u201d \uff21  ")
        == "\u201chello\u2014openbench!\u201d \uff21"
    )

    custom_dict = NormalizationConfig(
        unicode_normalization="NFKC",
        strip_unicode_punctuation=True,
    ).to_dict()
    assert custom_dict["unicode_normalization"] == "NFKC"
    assert custom_dict["strip_unicode_punctuation"] is True


def test_normalization_invalid_unicode_normalization_raises() -> None:
    """Invalid unicode_normalization mode raises ValueError."""
    with pytest.raises(ValueError):
        NormalizationConfig(unicode_normalization="NFD")


def test_exact_match_metric_passes_and_fails_accurately() -> None:
    """ExactMatchMetric scores 1.0 on normalized exact match and 0.0 on extra framing."""
    record = _make_record(expected="180")
    metric = ExactMatchMetric()

    res_pass = metric.compute_item(record, "  180 \n")
    assert res_pass.passed is True
    assert res_pass.score == 1.0

    res_fail = metric.compute_item(record, "The answer is 180.")
    assert res_fail.passed is False
    assert res_fail.score == 0.0


def test_exact_match_metric_case_sensitive_option() -> None:
    """ExactMatchMetric respects case_sensitive=True when configured."""
    record = _make_record(expected="ACKNOWLEDGED")
    metric = ExactMatchMetric(
        normalization=NormalizationConfig(case_sensitive=True)
    )

    assert metric.compute_item(record, "ACKNOWLEDGED").passed is True
    assert metric.compute_item(record, "acknowledged").passed is False


def test_accuracy_metric_bounded_substring_and_false_prefix_prevention() -> None:
    """AccuracyMetric recognizes bounded answers in sentences while rejecting partial number overlaps."""
    record = _make_record(expected="18")
    metric = AccuracyMetric(rule="exact_or_substring")

    # Should match when '18' appears as a distinct word/number
    res_good = metric.compute_item(record, "The final value is 18.")
    assert res_good.passed is True
    assert res_good.score == 1.0
    assert res_good.details["match_mode"] == "bounded_substring"

    # Must NOT match when '18' is merely a prefix of '180'
    res_bad = metric.compute_item(record, "The final value is 180.")
    assert res_bad.passed is False
    assert res_bad.score == 0.0


def test_accuracy_metric_respects_disallow_substring_metadata() -> None:
    """When record metadata sets disallow_substring=True, AccuracyMetric requires exact match."""
    record = _make_record(expected="ACKNOWLEDGED", disallow_substring=True)
    metric = AccuracyMetric(rule="exact_or_substring")

    assert metric.compute_item(record, "ACKNOWLEDGED").passed is True
    assert metric.compute_item(record, "Status: ACKNOWLEDGED").passed is False


def test_accuracy_metric_invalid_rule_raises_metric_error() -> None:
    """Passing an unsupported rule name to AccuracyMetric raises MetricError."""
    with pytest.raises(MetricError, match="Invalid accuracy correctness rule"):
        AccuracyMetric(rule="fuzzy_magic")


def test_keyword_coverage_partial_and_full_matches() -> None:
    """KeywordCoverageMetric returns matched/missing keywords and accurate coverage percentage."""
    record = _make_record(
        expected="Use Argon2id or bcrypt with a salt.",
        keywords=["Argon2id", "bcrypt", "salt", "plaintext"],
    )
    metric = KeywordCoverageMetric(pass_threshold=0.75)

    res = metric.compute_item(
        record,
        "Always hash passwords with Argon2id or bcrypt using a unique salt.",
    )
    assert res.details["matched_keywords"] == ["Argon2id", "bcrypt", "salt"]
    assert res.details["missing_keywords"] == ["plaintext"]
    assert res.score == 0.75
    assert res.details["coverage_percentage"] == 75.0
    assert res.passed is True


def test_metric_aggregation_penalizes_missing_responses_transparently() -> None:
    """BaseMetric.aggregate divides by total_dataset_items so missing items count as 0.0."""
    metric = ExactMatchMetric()
    item_results = [
        MetricResult(metric_name="exact_match", score=1.0, passed=True),
        MetricResult(metric_name="exact_match", score=1.0, passed=True),
    ]
    agg = metric.aggregate(item_results, total_dataset_items=4)
    assert agg.total_items == 4
    assert agg.evaluated_items == 2
    assert agg.passed_items == 2
    assert agg.score == 0.5
    assert agg.details["evaluated_only_score"] == 1.0
    assert agg.details["missing_items"] == 2


def test_custom_metric_registration_and_lookup() -> None:
    """Developers can subclass BaseMetric, register it, and instantiate it via create_metric."""

    class LengthCheckMetric(BaseMetric):
        name = "non_empty_length"
        version = "0.1.0"
        description = "Checks if response has at least 1 character."

        def __init__(self, normalization: NormalizationConfig | None = None) -> None:
            self.normalization = normalization

        def compute_item(self, record: DatasetRecord, response_text: str) -> MetricResult:
            ok = len(response_text.strip()) > 0
            return MetricResult(metric_name=self.name, score=1.0 if ok else 0.0, passed=ok)

    register_metric(LengthCheckMetric)
    inst = create_metric("non_empty_length")
    rec = _make_record("42")
    assert inst.compute_item(rec, "42").passed is True

    with pytest.raises(MetricError, match="Unknown metric"):
        create_metric("non_existent_metric_xyz")
