from __future__ import annotations

import re
from typing import Any

from openbench.dataset import DatasetRecord
from openbench.metrics.base import BaseMetric, MetricResult
from openbench.metrics.normalization import NormalizationConfig, normalize_text


class RegexMatchMetric(BaseMetric):
    name = "regex_match"
    version = "0.1.0"
    score_kind = "binary"
    description = "Binary regex match against metadata answer_pattern with exact-match fallback."
    limitation_note = "Falls back to normalized exact match when no answer_pattern is provided in metadata."

    def __init__(
        self,
        normalization: NormalizationConfig | None = None,
        fullmatch: bool = False,
    ) -> None:
        self.normalization = normalization or NormalizationConfig()
        self.fullmatch = fullmatch
        self._compiled_cache: dict[str, re.Pattern[str]] = {}

    def _compile_pattern(self, pattern: str, item_id: str) -> re.Pattern[str]:
        compiled = self._compiled_cache.get(pattern)
        if compiled is None:
            try:
                compiled = re.compile(pattern)
            except re.error as exc:
                raise ValueError(
                    f"Invalid answer_pattern regex for item {item_id!r}: {exc}"
                ) from exc
            self._compiled_cache[pattern] = compiled
        return compiled

    def compute_item(self, record: DatasetRecord, response_text: str) -> MetricResult:
        metadata = record.metadata or {}
        pattern = metadata.get("answer_pattern")
        norm_response = normalize_text(response_text, self.normalization)

        if not pattern:
            norm_expected = normalize_text(record.expected_answer, self.normalization)
            matched = norm_expected == norm_response
            return MetricResult(
                metric_name=self.name,
                score=1.0 if matched else 0.0,
                passed=matched,
                details={
                    "fallback": "exact_match",
                    "fallback_reason": "No answer_pattern present in metadata",
                    "normalized_expected": norm_expected,
                    "normalized_response": norm_response,
                },
            )

        compiled = self._compile_pattern(str(pattern), record.id)
        match_obj = (
            compiled.fullmatch(norm_response)
            if self.fullmatch
            else compiled.search(norm_response)
        )
        matched = match_obj is not None
        return MetricResult(
            metric_name=self.name,
            score=1.0 if matched else 0.0,
            passed=matched,
            details={
                "pattern": str(pattern),
                "fullmatch": self.fullmatch,
                "matched_text": match_obj.group(0) if match_obj is not None else None,
                "normalized_response": norm_response,
            },
        )

    def config_summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "fullmatch": self.fullmatch,
            "normalization": self.normalization.to_dict(),
        }
