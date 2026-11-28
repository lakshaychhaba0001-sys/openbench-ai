"""Configurable text normalization utilities for deterministic evaluation."""

from __future__ import annotations

import re
import string
import unicodedata
from dataclasses import dataclass
from typing import Any

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class NormalizationConfig:
    """Configuration controlling text normalization prior to metric comparison.

    By default:
    - Leading and trailing whitespace is stripped (`strip_whitespace=True`).
    - Case is normalized to lowercase (`case_sensitive=False`).
    - Repeated internal whitespace is collapsed to a single space (`collapse_whitespace=True`).
    - Punctuation is preserved (`strip_punctuation=False`) so meaningful symbols
      in code, math, or structured formats are never silently discarded.
    """

    strip_whitespace: bool = True
    case_sensitive: bool = False
    collapse_whitespace: bool = True
    strip_punctuation: bool = False
    unicode_normalization: str | None = None
    strip_unicode_punctuation: bool = False

    def __post_init__(self) -> None:
        if self.unicode_normalization not in (None, "NFC", "NFKC"):
            raise ValueError(
                f"Unsupported unicode_normalization: {self.unicode_normalization!r}. "
                "Expected None, 'NFC', or 'NFKC'."
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a plain dictionary."""
        data: dict[str, Any] = {
            "strip_whitespace": self.strip_whitespace,
            "case_sensitive": self.case_sensitive,
            "collapse_whitespace": self.collapse_whitespace,
            "strip_punctuation": self.strip_punctuation,
        }
        if self.unicode_normalization is not None:
            data["unicode_normalization"] = self.unicode_normalization
        if self.strip_unicode_punctuation:
            data["strip_unicode_punctuation"] = self.strip_unicode_punctuation
        return data

def normalize_text(text: str, config: NormalizationConfig | None = None) -> str:
    """Normalize a string according to the provided NormalizationConfig."""
    cfg = config or NormalizationConfig()
    result = text

    if cfg.unicode_normalization is not None:
        result = unicodedata.normalize(cfg.unicode_normalization, result)

    if cfg.strip_whitespace:
        result = result.strip()

    if cfg.strip_punctuation:
        result = result.translate(str.maketrans("", "", string.punctuation))
        if cfg.strip_unicode_punctuation:
            result = "".join(
                ch for ch in result if not unicodedata.category(ch).startswith("P")
            )

    if cfg.collapse_whitespace:
        result = _WHITESPACE_RE.sub(" ", result)
        if cfg.strip_whitespace:
            result = result.strip()

    if not cfg.case_sensitive:
        result = result.lower()

    return result
