"""Deterministic, configuration-aware response cache for OpenBench AI."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from openbench.evaluator import ResponseItem
from openbench.exceptions import OpenBenchError
from openbench.filelock import FileLock
from openbench.security import redact_sensitive_data


@dataclass(frozen=True)
class CacheRecord:
    """Auditable cache entry storing response data and full request provenance."""

    cache_key: str
    provider_name: str
    model_name: str
    model_version: str
    dataset_name: str
    dataset_version: str
    record_id: str
    prompt_sha256: str
    generation_config_sha256: str
    response: str
    latency_ms: float | None
    token_usage: dict[str, int] | None
    cached_at: str
    repeat_group_id: str | None = None
    repetition_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "cache_key": self.cache_key,
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "record_id": self.record_id,
            "prompt_sha256": self.prompt_sha256,
            "generation_config_sha256": self.generation_config_sha256,
            "response": self.response,
            "latency_ms": self.latency_ms,
            "token_usage": self.token_usage,
            "cached_at": self.cached_at,
        }
        # v0.4.0: repetition fields are written only for repeat-group entries, so entries
        # written without a repetition context are byte-identical to v0.3.1 entries.
        if self.repeat_group_id is not None:
            payload["repeat_group_id"] = self.repeat_group_id
            payload["repetition_index"] = self.repetition_index
        return payload


@dataclass(frozen=True)
class PruneResult:
    """Outcome of :meth:`ResponseCache.prune`.

    ``removed`` entries matched all supplied filters (with ``dry_run`` they only *would* be
    removed). ``kept`` entries did not match and remain in the cache. ``unknown_age`` entries
    matched any provider/model filters when an age filter was active, but had a missing or
    unreadable ``cached_at``; they are always kept and are not part of ``kept``.
    """

    removed: int
    kept: int
    unknown_age: int
    dry_run: bool = False


def _parse_cached_at(value: Any) -> datetime | None:
    """Parse a ``cached_at`` timestamp; ``None`` when missing or unreadable. Naive means UTC."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _check_repetition_context(context: tuple[str, int]) -> tuple[str, int]:
    if not isinstance(context, tuple) or len(context) != 2:
        raise OpenBenchError("repetition_context must be a (repeat_group_id, repetition_index) tuple")
    group_id, rep_index = context
    if not isinstance(group_id, str) or not group_id.strip():
        raise OpenBenchError("repetition_context repeat_group_id must be a non-empty string")
    if isinstance(rep_index, bool) or not isinstance(rep_index, int) or rep_index < 0:
        raise OpenBenchError("repetition_context repetition_index must be a non-negative integer")
    return group_id.strip(), rep_index


class ResponseCache:
    """File-backed deterministic response cache preventing cross-model/cross-config collisions.

    Cache Key Isolation & Credential Hygiene Guarantee:
    - The SHA-256 `cache_key` incorporates `provider_name`, `model_name`, `model_version`,
      `dataset_name`, `dataset_version`, `record_id`, `prompt_sha256`, and the canonical JSON
      hash of `redact_sensitive_data(generation_config)`.
    - Redacting credentials before hashing and storing ensures that rotating an API key or
      token does not invalidate valid cached responses and that raw credentials are never
      persisted inside cache artifacts on disk.

    Repetition isolation (v0.4.0):
    - With ``repetition_context=(repeat_group_id, repetition_index)`` both values are added to
      the hashed payload, so repetitions of a group never share entries, independent groups
      never share entries, and resuming repetition i of group G reuses only (G, i) entries.
    - Without a context (single runs) the key is byte-identical to v0.3.1.
    - The cache is a local file. Since v0.5.0, loads, appends and clears take an advisory
      cross-process file lock (``cache_entries.lock``), so several OpenBench processes can share
      one cache directory without losing or corrupting entries. It is not a distributed
      exactly-once store: two processes may both generate (and append) the same entry.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        enabled: bool = True,
        lock_timeout_seconds: float = 30.0,
    ) -> None:
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.cwd() / ".openbench_cache"
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self._index_path = self.cache_dir / "cache_entries.jsonl"
        self._memory_map: dict[str, CacheRecord] | None = None
        # Serialises loads, lookups and appends when the execution engine runs concurrently.
        self._lock = threading.RLock()
        # v0.5.0: cross-process advisory lock around file reads and writes.
        self._file_lock = FileLock(self.cache_dir / "cache_entries.lock", timeout_seconds=lock_timeout_seconds)

    @staticmethod
    def compute_cache_key(
        provider_name: str,
        model_name: str,
        model_version: str,
        dataset_name: str,
        dataset_version: str,
        record_id: str,
        prompt: str,
        generation_config: dict[str, Any] | None = None,
        repetition_context: tuple[str, int] | None = None,
    ) -> tuple[str, str, str]:
        """Return `(cache_key, prompt_sha256, generation_config_sha256)`.

        ``repetition_context`` is ``(repeat_group_id, repetition_index)`` for repeat-group
        runs and ``None`` (v0.3.1 key) otherwise.
        """
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        safe_gen_cfg = redact_sensitive_data(generation_config or {})
        gen_canonical = json.dumps(safe_gen_cfg, sort_keys=True, ensure_ascii=False)
        gen_hash = hashlib.sha256(gen_canonical.encode("utf-8")).hexdigest()

        components: dict[str, Any] = {
            "provider_name": provider_name.strip(),
            "model_name": model_name.strip(),
            "model_version": model_version.strip(),
            "dataset_name": dataset_name.strip(),
            "dataset_version": dataset_version.strip(),
            "record_id": record_id.strip(),
            "prompt_sha256": prompt_hash,
            "generation_config_sha256": gen_hash,
        }
        if repetition_context is not None:
            group_id, rep_index = _check_repetition_context(repetition_context)
            components["repeat_group_id"] = group_id
            components["repetition_index"] = rep_index
        composite = json.dumps(components, sort_keys=True, ensure_ascii=False)
        key = hashlib.sha256(composite.encode("utf-8")).hexdigest()
        return key, prompt_hash, gen_hash

    def _ensure_loaded(self) -> dict[str, CacheRecord]:
        if self._memory_map is not None:
            return self._memory_map
        if not self._index_path.exists():
            self._memory_map = {}
            return self._memory_map
        loaded: dict[str, CacheRecord] = {}
        with self._file_lock:
            text = self._index_path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise OpenBenchError(f"Corrupted cache file {self._index_path}: {exc.msg}") from exc
            try:
                rec = self._record_from_dict(data)
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                raise OpenBenchError(
                    f"Corrupted cache file {self._index_path}: line {line_no} is not a valid cache "
                    f"entry ({type(exc).__name__}: {exc}). Clear the cache with 'openbench cache clear'."
                ) from exc
            loaded[rec.cache_key] = rec
        # Assigned only after a complete, valid load, so a corrupted file never leaves a
        # silently truncated in-memory cache behind (v0.4.0).
        self._memory_map = loaded
        return self._memory_map

    @staticmethod
    def _record_from_dict(data: dict[str, Any]) -> CacheRecord:
        if not isinstance(data, dict):
            raise TypeError("entry is not a JSON object")
        return CacheRecord(
            cache_key=str(data["cache_key"]),
            provider_name=str(data["provider_name"]),
            model_name=str(data["model_name"]),
            model_version=str(data["model_version"]),
            dataset_name=str(data["dataset_name"]),
            dataset_version=str(data["dataset_version"]),
            record_id=str(data["record_id"]),
            prompt_sha256=str(data["prompt_sha256"]),
            generation_config_sha256=str(data["generation_config_sha256"]),
            response=str(data["response"]),
            latency_ms=data.get("latency_ms"),
            token_usage=data.get("token_usage"),
            cached_at=str(data["cached_at"]),
            repeat_group_id=data.get("repeat_group_id"),
            repetition_index=data.get("repetition_index"),
        )

    def get(
        self,
        provider_name: str,
        model_name: str,
        model_version: str,
        dataset_name: str,
        dataset_version: str,
        record_id: str,
        prompt: str,
        generation_config: dict[str, Any] | None = None,
        repetition_context: tuple[str, int] | None = None,
    ) -> CacheRecord | None:
        """Look up a cached response for the exact request parameters."""
        if not self.enabled:
            self.misses += 1
            return None

        key, _, _ = self.compute_cache_key(
            provider_name=provider_name,
            model_name=model_name,
            model_version=model_version,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            record_id=record_id,
            prompt=prompt,
            generation_config=generation_config,
            repetition_context=repetition_context,
        )
        with self._lock:
            store = self._ensure_loaded()
            if key in store:
                self.hits += 1
                return store[key]
            self.misses += 1
            return None

    def put(
        self,
        provider_name: str,
        model_name: str,
        model_version: str,
        dataset_name: str,
        dataset_version: str,
        record_id: str,
        prompt: str,
        response_item: ResponseItem,
        generation_config: dict[str, Any] | None = None,
        repetition_context: tuple[str, int] | None = None,
    ) -> CacheRecord:
        """Store a response in the cache with full configuration provenance (redacting any secrets)."""
        key, prompt_hash, gen_hash = self.compute_cache_key(
            provider_name=provider_name,
            model_name=model_name,
            model_version=model_version,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            record_id=record_id,
            prompt=prompt,
            generation_config=generation_config,
            repetition_context=repetition_context,
        )
        now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        rec = CacheRecord(
            cache_key=key,
            provider_name=provider_name,
            model_name=model_name,
            model_version=model_version,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            record_id=record_id,
            prompt_sha256=prompt_hash,
            generation_config_sha256=gen_hash,
            response=response_item.response,
            latency_ms=response_item.latency_ms,
            token_usage=response_item.token_usage,
            cached_at=now_iso,
            repeat_group_id=repetition_context[0] if repetition_context else None,
            repetition_index=repetition_context[1] if repetition_context else None,
        )
        if not self.enabled:
            return rec

        with self._lock:
            store = self._ensure_loaded()
            store[key] = rec
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            line = json.dumps(rec.to_dict(), ensure_ascii=False) + "\n"
            with self._file_lock, self._index_path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
        return rec

    def clear(self) -> int:
        """Delete all cached entries on disk and in memory, returning the number of cleared entries.

        A corrupted cache file can still be cleared (v0.4.0); the count is then the number of
        non-empty lines.
        """
        with self._lock:
            try:
                count = len(self._ensure_loaded())
            except OpenBenchError:
                count = sum(1 for line in self._index_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())
            self._memory_map = {}
            if self._index_path.exists():
                with self._file_lock:
                    if self._index_path.exists():
                        self._index_path.unlink()
            return count

    def prune(
        self,
        older_than_days: int | None = None,
        dry_run: bool = False,
        now: datetime | None = None,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> PruneResult:
        """Remove entries matching all supplied filters (age, ``provider``, and/or ``model``).

        At least one of ``older_than_days``, ``provider``, or ``model`` is required. When
        ``older_than_days`` is set, entries with a missing or unreadable ``cached_at`` are never
        removed by the age filter and are counted separately in ``PruneResult.unknown_age``. When
        filtering only by ``provider`` and/or ``model``, entry age is ignored and matching entries
        are removed regardless of ``cached_at``. With ``dry_run`` nothing is changed. Like ``put``
        and ``clear``, this holds the in-process and the cross-process file lock; the file is
        rewritten to a temporary file and swapped in, so a failure never leaves a half-written
        cache. An entry is judged by its last line, the one a load uses; earlier lines for the same
        key are not counted, and are dropped whenever the file is rewritten. A corrupted cache file
        is reported and left untouched.
        """
        if older_than_days is not None and (
            isinstance(older_than_days, bool) or not isinstance(older_than_days, int) or older_than_days < 1
        ):
            raise OpenBenchError("older_than_days must be a positive integer")
        if provider is not None:
            if not isinstance(provider, str) or not provider.strip():
                raise OpenBenchError("provider must be a non-empty string")
            provider = provider.strip()
        if model is not None:
            if not isinstance(model, str) or not model.strip():
                raise OpenBenchError("model must be a non-empty string")
            model = model.strip()
        if older_than_days is None and provider is None and model is None:
            raise OpenBenchError("Specify at least one filter: older_than_days, provider, or model")

        cutoff: datetime | None = None
        if older_than_days is not None:
            reference = now or datetime.now(timezone.utc)
            if reference.tzinfo is None:
                reference = reference.replace(tzinfo=timezone.utc)
            cutoff = reference - timedelta(days=older_than_days)

        if not self._index_path.exists():
            return PruneResult(0, 0, 0, dry_run)  # nothing to prune; do not create the directory
        with self._lock, self._file_lock:
            if not self._index_path.exists():
                return PruneResult(0, 0, 0, dry_run)
            text = self._index_path.read_text(encoding="utf-8")
            latest: dict[str, tuple[int, str, Any, Any, Any]] = {}
            for line_no, line in enumerate(text.splitlines(), start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data = json.loads(stripped)
                    key = str(data["cache_key"])
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise OpenBenchError(
                        f"Corrupted cache file {self._index_path}: line {line_no} is not a valid cache "
                        "entry. Nothing was pruned; clear the cache with 'openbench cache clear'."
                    ) from exc
                latest[key] = (
                    line_no,
                    stripped,
                    data.get("cached_at"),
                    data.get("provider_name"),
                    data.get("model_name"),
                )
            keep: list[str] = []
            removed = kept = unknown = 0
            for _, stripped, cached_at, entry_provider, entry_model in sorted(latest.values()):
                if provider is not None and entry_provider != provider:
                    kept += 1
                    keep.append(stripped)
                    continue
                if model is not None and entry_model != model:
                    kept += 1
                    keep.append(stripped)
                    continue
                if cutoff is not None:
                    when = _parse_cached_at(cached_at)
                    if when is None:
                        unknown += 1
                        keep.append(stripped)
                        continue
                    if when >= cutoff:
                        kept += 1
                        keep.append(stripped)
                        continue
                removed += 1
            if removed and not dry_run:
                tmp = self._index_path.with_suffix(".jsonl.tmp")
                tmp.write_text("".join(f"{entry}\n" for entry in keep), encoding="utf-8")
                os.replace(tmp, self._index_path)
                self._memory_map = None
            return PruneResult(removed, kept, unknown, dry_run)

    def stats(self) -> dict[str, Any]:
        """Return a cache audit summary.

        Besides the entry count and the session hit counters, the summary says what is in the
        cache and how old it is: ``by_model`` (entry counts per provider and model, sorted by
        provider, then model), ``oldest_cached_at`` and ``newest_cached_at`` (``None`` when no
        entry has a readable ``cached_at``), ``unknown_age_entries`` (entries whose
        ``cached_at`` is unreadable) and ``file_size_bytes`` (``0`` when there is no cache file).
        Like ``prune``, it never creates the cache directory, and a corrupted cache file raises
        ``OpenBenchError``.
        """
        store = self._ensure_loaded()
        counts: dict[tuple[str, str], int] = {}
        timestamps: list[tuple[datetime, str]] = []
        unknown_age = 0
        for rec in store.values():
            counts[(rec.provider_name, rec.model_name)] = counts.get((rec.provider_name, rec.model_name), 0) + 1
            when = _parse_cached_at(rec.cached_at)
            if when is None:
                unknown_age += 1
            else:
                timestamps.append((when, rec.cached_at))
        try:
            file_size = self._index_path.stat().st_size
        except OSError:
            file_size = 0
        return {
            "enabled": self.enabled,
            "cache_dir": str(self.cache_dir),
            "stored_entries": len(store),
            "session_hits": self.hits,
            "session_misses": self.misses,
            "by_model": [
                {"provider": provider, "model": model, "entries": count}
                for (provider, model), count in sorted(counts.items())
            ],
            "oldest_cached_at": min(timestamps)[1] if timestamps else None,
            "newest_cached_at": max(timestamps)[1] if timestamps else None,
            "unknown_age_entries": unknown_age,
            "file_size_bytes": file_size,
        }
