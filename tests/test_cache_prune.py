"""`ResponseCache.prune` and `openbench cache prune`: remove cached responses older than N days."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from openbench.cache import CacheRecord, PruneResult, ResponseCache
from openbench.cli import main
from openbench.exceptions import OpenBenchError

NOW = datetime(2026, 6, 30, 12, 0, 0, tzinfo=timezone.utc)


def _entry(key: str, cached_at: str | None, response: str = "x") -> dict:
    data = CacheRecord(
        cache_key=key, provider_name="p", model_name="m", model_version="1", dataset_name="d",
        dataset_version="1", record_id=key, prompt_sha256="a" * 64, generation_config_sha256="b" * 64,
        response=response, latency_ms=None, token_usage=None, cached_at=cached_at or "",
    ).to_dict()
    if cached_at is None:
        del data["cached_at"]
    return data


def _write(cache_dir: Path, entries: list[dict], raw_lines: tuple[str, ...] = ()) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "cache_entries.jsonl"
    lines = [json.dumps(entry) for entry in entries] + list(raw_lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _keys(path: Path) -> list[str]:
    return [json.loads(line)["cache_key"] for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_prune_removes_old_entries_and_keeps_new_ones(tmp_path):
    path = _write(tmp_path, [
        _entry("old", "2026-05-01T12:00:00+00:00"),
        _entry("recent", "2026-06-25T12:00:00+00:00"),
        _entry("zulu-old", "2026-05-01T12:00:00Z"),
    ])
    cache = ResponseCache(tmp_path)
    assert cache.prune(30, now=NOW) == PruneResult(removed=2, kept=1, unknown_age=0)
    assert _keys(path) == ["recent"]
    assert ResponseCache(tmp_path).stats()["stored_entries"] == 1
    assert cache.stats()["stored_entries"] == 1  # the same object sees the pruned cache


def test_prune_cut_off_is_exact(tmp_path):
    path = _write(tmp_path, [
        _entry("just-inside", "2026-05-31T12:00:01+00:00"),
        _entry("exactly-30-days", "2026-05-31T12:00:00+00:00"),
        _entry("just-outside", "2026-05-31T11:59:59+00:00"),
    ])
    result = ResponseCache(tmp_path).prune(30, now=NOW)
    assert (result.removed, result.kept) == (1, 2)
    assert _keys(path) == ["just-inside", "exactly-30-days"]


def test_prune_keeps_entries_with_missing_or_unreadable_cached_at(tmp_path):
    path = _write(tmp_path, [
        _entry("old", "2026-01-01T00:00:00+00:00"),
        _entry("missing", None),
        _entry("garbage", "last tuesday"),
        _entry("empty", ""),
    ])
    result = ResponseCache(tmp_path).prune(1, now=NOW)
    assert result == PruneResult(removed=1, kept=0, unknown_age=3)
    assert _keys(path) == ["missing", "garbage", "empty"]


def test_prune_dry_run_changes_nothing(tmp_path):
    path = _write(tmp_path, [_entry("old", "2026-01-01T00:00:00+00:00"), _entry("new", "2026-06-29T00:00:00+00:00")])
    before = path.read_bytes()
    cache = ResponseCache(tmp_path)
    result = cache.prune(30, dry_run=True, now=NOW)
    assert result == PruneResult(removed=1, kept=1, unknown_age=0, dry_run=True)
    assert path.read_bytes() == before
    assert cache.stats()["stored_entries"] == 2


def test_prune_judges_a_repeated_key_by_its_last_line(tmp_path):
    path = _write(tmp_path, [
        _entry("a", "2026-01-01T00:00:00+00:00", response="old answer"),
        _entry("a", "2026-06-29T00:00:00+00:00", response="new answer"),
        _entry("b", "2026-06-29T00:00:00+00:00", response="b last line"),
        _entry("b", "2026-01-01T00:00:00+00:00", response="b earlier line"),
        _entry("c", "2026-01-01T00:00:00+00:00"),
    ])
    result = ResponseCache(tmp_path).prune(30, now=NOW)
    assert (result.removed, result.kept) == (2, 1)
    assert _keys(path) == ["a"]
    assert json.loads(path.read_text(encoding="utf-8"))["response"] == "new answer"


def test_prune_when_everything_is_removed_leaves_an_empty_cache(tmp_path):
    path = _write(tmp_path, [_entry("a", "2026-01-01T00:00:00+00:00")])
    cache = ResponseCache(tmp_path)
    assert cache.prune(30, now=NOW).removed == 1
    assert path.read_text(encoding="utf-8") == ""
    assert cache.stats()["stored_entries"] == 0
    assert not list(tmp_path.glob("*.tmp"))


def test_prune_without_a_cache_file_is_a_no_op(tmp_path):
    assert ResponseCache(tmp_path / "missing").prune(7, now=NOW) == PruneResult(0, 0, 0)
    assert not (tmp_path / "missing").exists()


def test_prune_refuses_a_corrupted_file_and_leaves_it_untouched(tmp_path):
    path = _write(tmp_path, [_entry("old", "2026-01-01T00:00:00+00:00")], raw_lines=("{broken",))
    before = path.read_bytes()
    with pytest.raises(OpenBenchError, match="line 2"):
        ResponseCache(tmp_path).prune(30, now=NOW)
    assert path.read_bytes() == before


@pytest.mark.parametrize("days", [0, -3, 1.5, "7", True])
def test_prune_rejects_invalid_day_counts(tmp_path, days):
    with pytest.raises(OpenBenchError, match="positive integer"):
        ResponseCache(tmp_path).prune(days)


def test_prune_after_put_keeps_fresh_entries(tmp_path):
    from openbench.evaluator import ResponseItem

    cache = ResponseCache(tmp_path)
    cache.put("p", "m", "1", "d", "1", "r1", "prompt", ResponseItem(id="r1", response="x"))
    assert cache.prune(1).removed == 0
    assert cache.stats()["stored_entries"] == 1


def test_cli_cache_prune_dry_run_then_real(tmp_path, capsys):
    cache_dir = tmp_path / "cache"
    old = "2020-01-01T00:00:00+00:00"
    path = _write(cache_dir, [_entry("old", old), _entry("no-date", None)])
    assert main(["cache", "prune", "--older-than", "30", "--dry-run", "--cache-dir", str(cache_dir)]) == 0
    out = capsys.readouterr().out
    assert "Would remove 1 cached response(s) older than 30 day(s)" in out
    assert "Kept 0 newer entries." in out and "Kept 1 entry with a missing or unreadable cached_at." in out
    assert _keys(path) == ["old", "no-date"]
    assert main(["cache", "prune", "--older-than", "30", "--cache-dir", str(cache_dir)]) == 0
    assert "Removed 1 cached response(s)" in capsys.readouterr().out
    assert _keys(path) == ["no-date"]


@pytest.mark.parametrize("value", ["0", "-5", "abc", "1.5"])
def test_cli_cache_prune_rejects_non_positive_days(tmp_path, capsys, value):
    with pytest.raises(SystemExit) as excinfo:
        main(["cache", "prune", "--older-than", value, "--cache-dir", str(tmp_path)])
    assert excinfo.value.code == 2
    assert "positive integer" in capsys.readouterr().err


def test_cli_cache_prune_requires_older_than(tmp_path, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["cache", "prune", "--cache-dir", str(tmp_path)])
    assert excinfo.value.code == 2
    assert "--older-than" in capsys.readouterr().err


def test_cli_cache_prune_reports_a_corrupted_cache(tmp_path, capsys):
    _write(tmp_path, [], raw_lines=("{broken",))
    assert main(["cache", "prune", "--older-than", "30", "--cache-dir", str(tmp_path)]) == 1
    assert "Corrupted cache file" in capsys.readouterr().err

def _entry_for(
    key: str,
    cached_at: str | None,
    *,
    provider: str = "p",
    model: str = "m",
    response: str = "x",
) -> dict:
    data = CacheRecord(
        cache_key=key,
        provider_name=provider,
        model_name=model,
        model_version="1",
        dataset_name="d",
        dataset_version="1",
        record_id=key,
        prompt_sha256="a" * 64,
        generation_config_sha256="b" * 64,
        response=response,
        latency_ms=None,
        token_usage=None,
        cached_at=cached_at or "",
    ).to_dict()
    if cached_at is None:
        del data["cached_at"]
    return data


def test_prune_by_model_only(tmp_path):
    path = _write(tmp_path, [
        _entry_for("a1", "2026-06-29T12:00:00+00:00", provider="openai", model="gpt-a"),
        _entry_for("a2", "2026-01-01T12:00:00+00:00", provider="azure", model="gpt-a"),
        _entry_for("b1", "2026-01-01T12:00:00+00:00", provider="openai", model="gpt-b"),
    ])
    cache = ResponseCache(tmp_path)
    result = cache.prune(model="gpt-a")
    assert result == PruneResult(removed=2, kept=1, unknown_age=0)
    assert _keys(path) == ["b1"]
    assert cache.stats()["stored_entries"] == 1


def test_prune_by_provider_only(tmp_path):
    path = _write(tmp_path, [
        _entry_for("o1", "2026-06-29T12:00:00+00:00", provider="openai", model="gpt-a"),
        _entry_for("o2", "2026-06-28T12:00:00+00:00", provider="openai", model="gpt-b"),
        _entry_for("l1", "2026-01-01T12:00:00+00:00", provider="ollama", model="llama3.2"),
    ])
    cache = ResponseCache(tmp_path)
    result = cache.prune(provider="openai")
    assert result == PruneResult(removed=2, kept=1, unknown_age=0)
    assert _keys(path) == ["l1"]


def test_prune_by_model_and_age_uses_and_semantics(tmp_path):
    path = _write(tmp_path, [
        _entry_for("a-old", "2026-05-01T12:00:00+00:00", provider="openai", model="gpt-a"),
        _entry_for("a-new", "2026-06-25T12:00:00+00:00", provider="openai", model="gpt-a"),
        _entry_for("b-old", "2026-05-01T12:00:00+00:00", provider="openai", model="gpt-b"),
    ])
    result = ResponseCache(tmp_path).prune(30, now=NOW, model="gpt-a")
    assert result == PruneResult(removed=1, kept=2, unknown_age=0)
    assert _keys(path) == ["a-new", "b-old"]


def test_prune_by_provider_and_model_uses_and_semantics(tmp_path):
    path = _write(tmp_path, [
        _entry_for("openai-a", "2026-06-25T12:00:00+00:00", provider="openai", model="gpt-a"),
        _entry_for("azure-a", "2026-06-25T12:00:00+00:00", provider="azure", model="gpt-a"),
        _entry_for("openai-b", "2026-06-25T12:00:00+00:00", provider="openai", model="gpt-b"),
    ])
    result = ResponseCache(tmp_path).prune(provider="openai", model="gpt-a")
    assert result == PruneResult(removed=1, kept=2, unknown_age=0)
    assert _keys(path) == ["azure-a", "openai-b"]


def test_prune_unreadable_cached_at_with_and_without_age_filter(tmp_path):
    path = _write(tmp_path, [
        _entry_for("a-old", "2026-01-01T00:00:00+00:00", provider="openai", model="gpt-a"),
        _entry_for("a-bad", "not-a-date", provider="openai", model="gpt-a"),
        _entry_for("b-bad", None, provider="openai", model="gpt-b"),
    ])
    cache = ResponseCache(tmp_path)
    # With age + model filter: unreadable cached_at for gpt-a is kept as unknown_age.
    with_age = cache.prune(30, now=NOW, model="gpt-a")
    assert with_age == PruneResult(removed=1, kept=1, unknown_age=1)
    assert _keys(path) == ["a-bad", "b-bad"]

    # With model filter only: age does not matter, so unreadable gpt-a is removed.
    without_age = cache.prune(model="gpt-a")
    assert without_age == PruneResult(removed=1, kept=1, unknown_age=0)
    assert _keys(path) == ["b-bad"]


def test_prune_model_dry_run_changes_nothing(tmp_path):
    path = _write(tmp_path, [
        _entry_for("a1", "2026-06-25T12:00:00+00:00", provider="openai", model="gpt-a"),
        _entry_for("b1", "2026-06-25T12:00:00+00:00", provider="openai", model="gpt-b"),
    ])
    before = path.read_bytes()
    result = ResponseCache(tmp_path).prune(model="gpt-a", dry_run=True)
    assert result == PruneResult(removed=1, kept=1, unknown_age=0, dry_run=True)
    assert path.read_bytes() == before


def test_prune_requires_at_least_one_filter(tmp_path):
    with pytest.raises(OpenBenchError, match="at least one filter"):
        ResponseCache(tmp_path).prune()


def test_prune_existing_positional_calls_behave_as_before(tmp_path):
    path = _write(tmp_path, [
        _entry("very-old", "2020-01-01T00:00:00+00:00"),
        _entry("far-future", "2099-01-01T00:00:00+00:00"),
    ])
    before = path.read_bytes()
    cache = ResponseCache(tmp_path)
    assert cache.prune(30, dry_run=True) == PruneResult(removed=1, kept=1, unknown_age=0, dry_run=True)
    assert path.read_bytes() == before
    assert cache.prune(30) == PruneResult(removed=1, kept=1, unknown_age=0, dry_run=False)
    assert _keys(path) == ["far-future"]


def test_cli_cache_prune_by_provider_model_and_age(tmp_path, capsys):
    cache_dir = tmp_path / "cache"
    path = _write(cache_dir, [
        _entry_for("a-old", "2020-01-01T00:00:00+00:00", provider="openai", model="gpt-a"),
        _entry_for("a-new", "2099-01-01T00:00:00+00:00", provider="openai", model="gpt-a"),
        _entry_for("b-old", "2020-01-01T00:00:00+00:00", provider="openai", model="gpt-b"),
        _entry_for("l-old", "2020-01-01T00:00:00+00:00", provider="ollama", model="llama3.2"),
    ])

    # Dry run with model + age
    assert main([
        "cache", "prune", "--model", "gpt-a", "--older-than", "30",
        "--dry-run", "--cache-dir", str(cache_dir),
    ]) == 0
    out = capsys.readouterr().out
    assert "Would remove 1 cached response(s) for model gpt-a older than 30 day(s)" in out
    assert len(_keys(path)) == 4

    # Real run with model + age
    assert main([
        "cache", "prune", "--model", "gpt-a", "--older-than", "30",
        "--cache-dir", str(cache_dir),
    ]) == 0
    out = capsys.readouterr().out
    assert "Removed 1 cached response(s) for model gpt-a older than 30 day(s)" in out
    assert _keys(path) == ["a-new", "b-old", "l-old"]

    # Real run with provider only
    assert main(["cache", "prune", "--provider", "openai", "--cache-dir", str(cache_dir)]) == 0
    out = capsys.readouterr().out
    assert "Removed 2 cached response(s) for provider openai" in out
    assert _keys(path) == ["l-old"]

    # Real run with model only
    assert main(["cache", "prune", "--model", "llama3.2", "--cache-dir", str(cache_dir)]) == 0
    out = capsys.readouterr().out
    assert "Removed 1 cached response(s) for model llama3.2" in out
    assert _keys(path) == []


def test_cli_cache_prune_no_filter_exits_with_code_2(tmp_path, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["cache", "prune", "--cache-dir", str(tmp_path)])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "--older-than" in err
    assert "--provider" in err
    assert "--model" in err
