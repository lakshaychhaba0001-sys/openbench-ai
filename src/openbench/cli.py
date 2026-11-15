"""Command-Line Interface for OpenBench AI (v0.3.0 commands preserved; v0.4.0 additions)."""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import webbrowser
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from openbench import __version__
from openbench.adapters import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_OLLAMA_ENDPOINT,
    DEFAULT_OPENAI_COMPATIBLE_ENDPOINT,
    GenerationConfig,
    create_provider,
    list_registered_providers,
)
from openbench.analytics import BenchmarkAnalytics
from openbench.cache import ResponseCache
from openbench.catalog import DatasetCatalog
from openbench.comparison import ComparisonResult, ModelComparator
from openbench.configuration import ProfileLoader
from openbench.dataset import DatasetLoader
from openbench.evaluator import (
    DEFAULT_METRIC_NAMES,
    EvaluationConfig,
    EvaluationResult,
    Evaluator,
    ResponseLoader,
)
from openbench.exceptions import OpenBenchError
from openbench.execution import ExecutionConfig, RetryPolicy
from openbench.experiments import ExperimentManager
from openbench.history import ResultHistoryStore
from openbench.reporting import ReportGenerator
from openbench.security import sanitize_error_message


def _slugify(value: str) -> str:
    """Create a filesystem-safe slug from a model name."""
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip()).strip("_").lower()
    return slug or "model"


def _version_string() -> str:
    """One-line version for bug reports: OpenBench, Python and platform (v0.5.0)."""
    import platform

    return f"openbench {__version__} (Python {platform.python_version()}, {platform.platform(terse=True)})"


def _examples(*lines: str) -> str:
    return "examples:\n" + "\n".join(f"  {line}" for line in lines)


def build_parser() -> argparse.ArgumentParser:
    """Construct the OpenBench AI CLI argument parser (preserving all v0.1.0 & v0.2.0 commands + v0.3.0 commands)."""
    parser = argparse.ArgumentParser(
        prog="openbench",
        description="OpenBench AI — Modular, open-source LLM experiment execution and benchmarking platform.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=_version_string(),
        help="Show the OpenBench AI version and exit.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available OpenBench commands")

    # 1. validate-dataset (v0.1.0 preserved)
    val_parser = subparsers.add_parser(
        "validate-dataset",
        help="Validate a JSONL dataset file or directory of JSONL files.",
    )
    val_parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to a JSONL dataset file or directory.",
    )
    val_parser.add_argument(
        "--quality-checks",
        action="store_true",
        help="v0.4.0: also run the read-only dataset quality audit (exit 1 only for error findings).",
    )
    val_parser.add_argument("--output", type=str, default=None, help="v0.4.0: write the audit report to this file.")
    val_parser.add_argument("--format", choices=["json", "markdown", "csv"], default="json", help="v0.4.0: audit report format.")
    val_parser.add_argument("--near-duplicate-threshold", type=float, default=0.85, help="v0.4.0: DQ003 Jaccard threshold.")
    val_parser.add_argument("--metrics", nargs="+", default=None, help="v0.4.0: metric names to check (DQ008).")
    val_parser.add_argument(
        "--near-duplicate-method",
        choices=["lexical", "semantic"],
        default="lexical",
        help="v0.5.0: 'semantic' adds an embedding pass that finds paraphrases (needs the 'semantic' extra).",
    )
    val_parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                            help="v0.5.0: local sentence-transformers model for --near-duplicate-method semantic.")
    val_parser.add_argument("--semantic-threshold", type=float, default=0.9, help="v0.5.0: cosine threshold for semantic DQ003.")

    # 2. evaluate (v0.1.0 & v0.2.0 preserved)
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate one or more pre-generated model response files against a dataset or YAML profile.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_examples(
            'openbench evaluate --dataset datasets/reasoning/sample.jsonl --responses examples/sample_responses.json',
            'openbench evaluate --dataset datasets/reasoning/sample.jsonl --responses examples/sample_responses.json --metrics exact_match f1 --output results/',
        ),
    )
    eval_parser.add_argument(
        "--dataset",
        required=False,
        type=str,
        help="Path to a JSONL dataset file or directory (required unless --config specifies dataset.path).",
    )
    eval_parser.add_argument(
        "--config",
        required=False,
        type=str,
        help="Path to a YAML evaluation configuration profile.",
    )
    eval_parser.add_argument(
        "--responses",
        required=True,
        type=str,
        help="Path to a JSON file containing model responses.",
    )
    eval_parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Space-separated list of metrics to compute.",
    )
    eval_parser.add_argument(
        "--category",
        nargs="+",
        default=None,
        help="Optional category filter(s) to apply to the dataset.",
    )
    eval_parser.add_argument(
        "--difficulty",
        nargs="+",
        default=None,
        help="Optional difficulty filter(s) (easy, medium, hard).",
    )
    eval_parser.add_argument(
        "--ids-file",
        required=False,
        default=None,
        type=str,
        help="Optional path to a file listing dataset record IDs to evaluate (like-for-like subset evaluation).",
    )
    eval_parser.add_argument(
        "--history-file",
        required=False,
        default=None,
        type=str,
        help="Optional path to local JSONL history file to record evaluation summary.",
    )
    eval_parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output directory or file path (.json, .md, or .csv) for evaluation reports.",
    )
    eval_parser.add_argument(
        "--confidence-intervals",
        action="store_true",
        help="v0.4.0: add a 'statistics' block with percentile bootstrap CIs over items "
        "(output is unchanged without this flag).",
    )
    eval_parser.add_argument(
        "--confidence-level", type=float, default=0.95, help="v0.4.0: CI level for --confidence-intervals (default 0.95)."
    )
    eval_parser.add_argument(
        "--resamples", type=int, default=10_000, help="v0.4.0: bootstrap resamples for --confidence-intervals (default 10000)."
    )
    eval_parser.add_argument(
        "--seed", type=int, default=0, help="v0.4.0: statistical analysis seed for --confidence-intervals (default 0)."
    )

    # 3. compare (v0.1.0 preserved)
    cmp_parser = subparsers.add_parser(
        "compare",
        help="Compare two or more JSON evaluation result files.",
    )
    cmp_parser.add_argument(
        "--results",
        action="append",
        required=True,
        nargs="+",
        help="Paths to evaluation result JSON files (provide at least two files).",
    )
    cmp_parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output file path (.md, .json, or .csv) or directory for the comparison report.",
    )

    # 4. list-datasets (v0.2.0 preserved)
    list_ds_parser = subparsers.add_parser(
        "list-datasets",
        help="List all built-in or directory JSONL datasets in the catalog.",
    )
    list_ds_parser.add_argument("--path", type=str, default=None)
    list_ds_parser.add_argument("--category", type=str, default=None)
    list_ds_parser.add_argument("--difficulty", type=str, default=None)
    list_ds_parser.add_argument("--format", choices=["text", "json"], default="text")

    # 5. inspect-dataset (v0.2.0 preserved)
    inspect_ds_parser = subparsers.add_parser(
        "inspect-dataset",
        help="Inspect metadata, schema breakdown, and sample records of a dataset.",
    )
    inspect_ds_parser.add_argument("dataset_path", type=str)
    inspect_ds_parser.add_argument("--category", type=str, default=None)
    inspect_ds_parser.add_argument("--difficulty", type=str, default=None)
    inspect_ds_parser.add_argument("--samples", type=int, default=2)
    inspect_ds_parser.add_argument("--format", choices=["text", "json"], default="text")

    # 6. dataset-stats (v0.2.0 preserved)
    stats_ds_parser = subparsers.add_parser(
        "dataset-stats",
        help="Display detailed statistics for a dataset file or directory.",
    )
    stats_ds_parser.add_argument("dataset_path", type=str)
    stats_ds_parser.add_argument("--category", type=str, default=None)
    stats_ds_parser.add_argument("--difficulty", type=str, default=None)
    stats_ds_parser.add_argument("--format", choices=["text", "json"], default="text")

    # 7. history (v0.2.0 preserved)
    history_parser = subparsers.add_parser(
        "history",
        help="List or inspect local historical evaluation runs.",
    )
    history_subparsers = history_parser.add_subparsers(dest="history_action")
    hist_list_parser = history_subparsers.add_parser("list")
    hist_list_parser.add_argument("--store", type=str, default=None)
    hist_list_parser.add_argument("--model", type=str, default=None)
    hist_list_parser.add_argument("--dataset", type=str, default=None)
    hist_list_parser.add_argument("--format", choices=["text", "json"], default="text")

    hist_show_parser = history_subparsers.add_parser("show")
    hist_show_parser.add_argument("evaluation_id", type=str)
    hist_show_parser.add_argument("--store", type=str, default=None)
    hist_show_parser.add_argument("--format", choices=["text", "json"], default="text")

    # 8. export (v0.2.0 preserved)
    export_parser = subparsers.add_parser(
        "export",
        help="Export an existing JSON evaluation or comparison result into CSV or Markdown.",
    )
    export_parser.add_argument("--input", required=True, type=str)
    export_parser.add_argument("--format", required=True, choices=["csv", "markdown", "json"])
    export_parser.add_argument("--mode", choices=["individual", "aggregate"], default="individual")
    export_parser.add_argument("--output", required=True, type=str)

    # 9. run (v0.3.0 new: Execute a model provider on a dataset and record an experiment run)
    run_parser = subparsers.add_parser(
        "run",
        help="Execute a model provider against a dataset, evaluate metrics, and create a verifiable experiment run.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_examples(
            'openbench run --dataset datasets/reasoning/sample.jsonl --provider mock --model demo',
            'openbench run --dataset datasets/reasoning/sample.jsonl --provider ollama --model llama3.2 --repetitions 3 --seed 7',
            'openbench run --dataset datasets/reasoning/sample.jsonl --provider openai_compatible --endpoint http://localhost:8000/v1 --model my-model',
        ),
    )
    run_parser.add_argument(
        "--config",
        required=False,
        type=str,
        help="Path to a YAML experiment configuration profile.",
    )
    run_parser.add_argument(
        "--dataset",
        required=False,
        type=str,
        help="Path to a JSONL dataset file or directory.",
    )
    run_parser.add_argument(
        "--provider",
        required=False,
        type=str,
        help="Model provider identifier (e.g. ollama, huggingface, mock).",
    )
    run_parser.add_argument(
        "--model",
        required=False,
        type=str,
        help="Model identifier (e.g. llama3.2, gpt2, mock-deterministic-v1).",
    )
    run_parser.add_argument(
        "--model-version",
        required=False,
        default=None,
        type=str,
        help="Model version or revision (default: latest or profile setting).",
    )
    run_parser.add_argument(
        "--endpoint",
        required=False,
        default=None,
        type=str,
        help="Optional provider HTTP endpoint (e.g. http://localhost:11434 for Ollama).",
    )
    run_parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Metrics to evaluate (default: exact_match accuracy keyword_coverage, or profile setting).",
    )
    run_parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Number of concurrent model requests (default: 1).",
    )
    run_parser.add_argument(
        "--max-requests-per-minute",
        type=float,
        default=None,
        help="v0.5.0: limit provider requests per minute across all workers (for hosted APIs).",
    )
    run_parser.add_argument(
        "--api-key-env",
        type=str,
        default=None,
        help="v0.5.0: environment variable holding the API key for openai_compatible "
        "(default: OPENAI_API_KEY). The key itself is never stored.",
    )
    run_parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-request timeout in seconds (default: 30.0).",
    )
    run_parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Maximum retry attempts per item (default: 3).",
    )
    run_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable deterministic response caching for this run.",
    )
    run_parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for response cache (default: .openbench_cache).",
    )
    run_parser.add_argument(
        "--experiments-dir",
        type=str,
        default=None,
        help="Directory for storing experiment runs (default: experiments).",
    )
    run_parser.add_argument(
        "--output",
        required=False,
        default=None,
        type=str,
        help="Optional output directory to also write top-level .json/.md/.csv reports.",
    )
    run_parser.add_argument(
        "--repetitions",
        type=int,
        default=None,
        help="v0.4.0: number of independent repetitions (>= 2 creates a repeat group; overrides "
        "options.repeat_count).",
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="v0.4.0: generation base seed (repetition i uses seed + i); overrides generation.seed.",
    )
    run_parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="v0.4.0: optional experiment ID recorded in run lineage.",
    )
    run_parser.add_argument(
        "--parent-experiment-id",
        type=str,
        default=None,
        help="v0.4.0: optional parent experiment ID recorded in run lineage.",
    )

    # 10. runs (v0.3.0 new: list, show, verify, resume, compare)
    runs_parser = subparsers.add_parser(
        "runs",
        help="Manage, inspect, verify, resume, and compare experiment runs.",
    )
    runs_subparsers = runs_parser.add_subparsers(dest="runs_action")

    runs_list_p = runs_subparsers.add_parser("list", help="List stored experiment runs.")
    runs_list_p.add_argument("--experiments-dir", type=str, default=None)
    runs_list_p.add_argument("--status", type=str, default=None)
    runs_list_p.add_argument("--provider", type=str, default=None)
    runs_list_p.add_argument("--format", choices=["text", "json"], default="text")
    runs_list_p.add_argument(
        "--group", type=str, default=None, help="v0.4.0: list only the runs of this repeat group."
    )
    runs_list_p.add_argument(
        "--wide",
        action="store_true",
        help="v0.5.0: print the full one-line-per-run format (all scores, origin, dataset).",
    )

    runs_show_p = runs_subparsers.add_parser("show", help="Inspect a specific experiment run.")
    runs_show_p.add_argument("run_id", type=str)
    runs_show_p.add_argument("--experiments-dir", type=str, default=None)
    runs_show_p.add_argument("--format", choices=["text", "json"], default="text")

    runs_verify_p = runs_subparsers.add_parser(
        "verify",
        help="Verify cryptographic SHA-256 artifact checksums and provenance of a run.",
    )
    runs_verify_p.add_argument("run_id", type=str)
    runs_verify_p.add_argument("--experiments-dir", type=str, default=None)
    runs_verify_p.add_argument("--format", choices=["text", "json"], default="text")
    runs_verify_p.add_argument(
        "--quality-checks", action="store_true", help="v0.4.0: also run the experiment quality audit (EQ checks)."
    )
    runs_verify_p.add_argument("--audit-output", type=str, default=None, help="v0.4.0: write the audit report (JSON) here.")

    runs_resume_p = runs_subparsers.add_parser(
        "resume",
        help="Resume an interrupted or partial provider-executed experiment run.",
    )
    runs_resume_p.add_argument("run_id", type=str)
    runs_resume_p.add_argument("--experiments-dir", type=str, default=None)
    runs_resume_p.add_argument("--cache-dir", type=str, default=None)

    runs_compare_p = runs_subparsers.add_parser(
        "compare",
        help="Compare two or more experiment runs by their RUN_IDs.",
    )
    runs_compare_p.add_argument("run_ids", nargs="+", help="Two or more experiment run IDs.")
    runs_compare_p.add_argument("--experiments-dir", type=str, default=None)
    runs_compare_p.add_argument("--output", type=str, default=None)
    runs_compare_p.add_argument("--format", choices=["text", "json", "markdown"], default="text")

    # v0.5.0: rank, plan, judge
    rank_parser = subparsers.add_parser(
        "rank",
        help="v0.5.0: rank several stored runs, repeat groups or evaluation files with uncertainty.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_examples(
            "openbench rank --runs run_<a> run_<b> grp_<c> --metric exact_match --format markdown",
            "openbench rank --results a.json b.json c.json --metric f1 --output ranking.json",
        ),
    )
    rank_src = rank_parser.add_mutually_exclusive_group(required=True)
    rank_src.add_argument("--runs", nargs="+", default=None, help="Run IDs (run_...) or repeat group IDs (grp_...).")
    rank_src.add_argument("--results", nargs="+", default=None, help="Evaluation result JSON files.")
    rank_parser.add_argument("--metric", required=True, type=str)
    rank_parser.add_argument("--experiments-dir", type=str, default=None)
    rank_parser.add_argument("--ids-file", type=str, default=None)
    rank_parser.add_argument("--alpha", type=float, default=0.05)
    rank_parser.add_argument("--confidence-level", type=float, default=0.95)
    rank_parser.add_argument("--resamples", type=int, default=10_000)
    rank_parser.add_argument("--permutations", type=int, default=9_999)
    rank_parser.add_argument("--seed", type=int, default=0)
    rank_parser.add_argument("--exploratory", action="store_true", help="Rank inputs that fail hard compatibility checks (descriptive only).")
    rank_parser.add_argument("--output", type=str, default=None)
    rank_parser.add_argument("--format", choices=["json", "markdown", "csv", "html"], default="markdown")

    plan_parser = subparsers.add_parser(
        "plan",
        help="v0.5.0: how many questions are needed to detect a difference (or what n can detect).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_examples(
            "openbench plan --baseline-accuracy 0.7 --difference 0.05",
            "openbench plan --discordance 0.2 --difference 0.1 --power 0.9",
            "openbench plan --items 200 --baseline-accuracy 0.7       # smallest detectable difference",
            "openbench plan --pilot-runs run_<a> run_<b> --metric exact_match --difference 0.05",
            "openbench plan --continuous --sd-difference 0.3 --difference 0.05",
        ),
    )
    plan_parser.add_argument("--difference", type=float, default=None, help="Smallest difference worth detecting (metric units).")
    plan_parser.add_argument("--items", type=int, default=None, help="Reverse mode: report the smallest detectable difference at this n.")
    plan_parser.add_argument("--baseline-accuracy", type=float, default=None)
    plan_parser.add_argument("--discordance", type=float, default=None, help="Share of items where the two models disagree.")
    plan_parser.add_argument("--continuous", action="store_true", help="Plan for a continuous metric (sign-flip test).")
    plan_parser.add_argument("--sd-difference", type=float, default=None, help="SD of per-item differences (continuous).")
    plan_parser.add_argument("--pilot-runs", nargs=2, default=None, metavar=("RUN_A", "RUN_B"), help="Estimate inputs from two stored runs.")
    plan_parser.add_argument("--metric", type=str, default="exact_match", help="Metric for --pilot-runs (default exact_match).")
    plan_parser.add_argument("--experiments-dir", type=str, default=None)
    plan_parser.add_argument("--alpha", type=float, default=0.05)
    plan_parser.add_argument("--power", type=float, default=0.8)
    plan_parser.add_argument("--output", type=str, default=None)
    plan_parser.add_argument("--format", choices=["json", "markdown", "csv"], default="markdown")

    judge_parser = subparsers.add_parser(
        "judge",
        help="v0.5.0: pairwise LLM-as-judge with order swapping and bias checks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_examples(
            "openbench judge --runs run_<a> run_<b> --judge-provider ollama --judge-model qwen2.5:7b --format markdown",
            "openbench judge --runs run_<a> run_<b> --judge-provider openai_compatible --judge-endpoint http://localhost:8000/v1 "
            "--judge-model judge --key prefs/tasks.key.json --annotations ann.jsonl",
        ),
    )
    judge_src = judge_parser.add_mutually_exclusive_group(required=True)
    judge_src.add_argument("--runs", nargs=2, default=None, metavar=("RUN_A", "RUN_B"))
    judge_src.add_argument("--results", nargs=2, default=None, metavar=("EVAL_A", "EVAL_B"))
    judge_parser.add_argument("--judge-provider", required=True, type=str)
    judge_parser.add_argument("--judge-model", required=True, type=str)
    judge_parser.add_argument("--judge-endpoint", type=str, default=None)
    judge_parser.add_argument("--api-key-env", type=str, default=None)
    judge_parser.add_argument("--criteria", type=str, default=None, help="Text file with judging criteria (replaces the default).")
    judge_parser.add_argument("--show-reference", action="store_true", help="Show the reference answer to the judge.")
    judge_parser.add_argument("--experiments-dir", type=str, default=None)
    judge_parser.add_argument("--cache-dir", type=str, default=None)
    judge_parser.add_argument("--no-cache", action="store_true")
    judge_parser.add_argument("--concurrency", type=int, default=1)
    judge_parser.add_argument("--max-requests-per-minute", type=float, default=None)
    judge_parser.add_argument("--timeout", type=float, default=60.0)
    judge_parser.add_argument("--key", type=str, default=None, help="Preference task key for agreement with people.")
    judge_parser.add_argument("--annotations", nargs="+", default=None, help="Validated human annotation files for that key.")
    judge_parser.add_argument("--seed", type=int, default=0)
    judge_parser.add_argument("--resamples", type=int, default=10_000)
    judge_parser.add_argument("--output", type=str, default=None)
    judge_parser.add_argument("--format", choices=["json", "markdown", "csv"], default="markdown")

    # v0.5.0: benchmarks and reports
    imp_parser = subparsers.add_parser(
        "import-dataset",
        help="v0.5.0: convert a local benchmark file (QA or multiple choice) into an OpenBench JSONL dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_examples(
            "openbench import-dataset mmlu_math.csv --format mcq-csv --category math --license MIT --output datasets/mmlu_math.jsonl",
            "openbench import-dataset qa.jsonl --format qa-jsonl --category trivia --question-field q --answer-field a --output trivia.jsonl",
        ),
    )
    imp_parser.add_argument("input", type=str)
    imp_parser.add_argument("--format", required=True, choices=["qa-jsonl", "qa-csv", "mcq-jsonl", "mcq-csv"], dest="import_format")
    imp_parser.add_argument("--category", required=True, type=str)
    imp_parser.add_argument("--output", required=True, type=str)
    imp_parser.add_argument("--id-prefix", type=str, default=None)
    imp_parser.add_argument("--source", type=str, default=None, help="Where the benchmark comes from (recorded in metadata).")
    imp_parser.add_argument("--license", type=str, default=None, dest="license_name", help="The benchmark's licence (recorded in metadata).")
    imp_parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default=None)
    imp_parser.add_argument("--question-field", type=str, default="question")
    imp_parser.add_argument("--answer-field", type=str, default="answer")
    imp_parser.add_argument("--id-field", type=str, default="id")
    imp_parser.add_argument("--overwrite", action="store_true")

    cont_parser = subparsers.add_parser(
        "audit-contamination",
        help="v0.5.0: read-only check of word n-gram overlap between a dataset and a reference corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_examples(
            "openbench audit-contamination datasets/reasoning/sample.jsonl --corpus my_finetune_data/ --format markdown",
            "openbench audit-contamination my.jsonl --corpus public_benchmark.jsonl --ngram 8 --threshold 0.3 --output contamination.json",
        ),
    )
    cont_parser.add_argument("dataset_path", type=str)
    cont_parser.add_argument("--corpus", required=True, type=str, help="A text, Markdown, JSON, JSONL or CSV file, or a directory of them.")
    cont_parser.add_argument("--ngram", type=int, default=13)
    cont_parser.add_argument("--threshold", type=float, default=0.5)
    cont_parser.add_argument("--min-words", type=int, default=8)
    cont_parser.add_argument("--output", type=str, default=None)
    cont_parser.add_argument("--format", choices=["json", "markdown", "csv"], default="markdown")

    card_parser = subparsers.add_parser(
        "dataset-card",
        help="v0.5.0: generate a dataset card (facts computed from the data, author sections left as TODO).",
    )
    card_parser.add_argument("dataset_path", type=str)
    card_parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    card_parser.add_argument("--output", type=str, default=None)

    serve_parser = subparsers.add_parser(
        "serve",
        help="v0.5.0: read-only local web dashboard for runs, repeat groups and saved reports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_examples("openbench serve --experiments-dir experiments --reports-dir reports", "openbench serve --port 9000"),
    )
    serve_parser.add_argument("--experiments-dir", type=str, default="experiments")
    serve_parser.add_argument("--reports-dir", type=str, default=None, help="Directory of saved .json/.html analysis reports.")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument("--allow-remote", action="store_true", help="Allow listening on a non-loopback address (exposes your results).")
    serve_parser.add_argument("--open", action="store_true", help="Open the dashboard in your default browser.")
    # v0.5.0: metrics and plugins listings
    met_parser = subparsers.add_parser("metrics", help="v0.5.0: list built-in and plugin metrics.")
    met_sub = met_parser.add_subparsers(dest="metrics_action")
    met_list = met_sub.add_parser("list", help="List registered metrics and where they come from.")
    met_list.add_argument("--format", choices=["text", "json"], default="text")
    plug_parser = subparsers.add_parser("plugins", help="v0.5.0: list installed OpenBench plugins.")
    plug_sub = plug_parser.add_subparsers(dest="plugins_action")
    plug_list = plug_sub.add_parser("list", help="Discover and list installed plugins (this imports them).")
    plug_list.add_argument("--format", choices=["text", "json"], default="text")

    # 11. providers (v0.3.0 new: list, check)
    prov_parser = subparsers.add_parser(
        "providers",
        help="List registered model providers or check provider connectivity.",
    )
    prov_subparsers = prov_parser.add_subparsers(dest="providers_action")

    prov_list_p = prov_subparsers.add_parser("list", help="List all registered model provider adapters.")
    prov_list_p.add_argument("--format", choices=["text", "json"], default="text")

    prov_check_p = prov_subparsers.add_parser("check", help="Check connectivity/availability of a provider.")
    prov_check_p.add_argument("provider_name", type=str, help="Provider name (e.g. ollama, huggingface, mock).")
    prov_check_p.add_argument("--model", type=str, default=None)
    prov_check_p.add_argument("--endpoint", type=str, default=None)
    prov_check_p.add_argument("--format", choices=["text", "json"], default="text")

    # 12. cache (v0.3.0 new: clear, stats)
    cache_parser = subparsers.add_parser(
        "cache",
        help="Manage or clear the deterministic response cache.",
    )
    cache_subparsers = cache_parser.add_subparsers(dest="cache_action")

    cache_clear_p = cache_subparsers.add_parser("clear", help="Clear all cached responses.")
    cache_clear_p.add_argument("--cache-dir", type=str, default=None)

    cache_stats_p = cache_subparsers.add_parser("stats", help="Show cache entry count and location.")
    cache_stats_p.add_argument("--cache-dir", type=str, default=None)

    # 13. analyze (v0.4.0 new: paired comparison / repeated-run variability of stored results)
    ana_parser = subparsers.add_parser(
        "analyze",
        help="v0.4.0: statistically compare stored runs, repeat groups or evaluation files "
        "(no provider calls), or analyse one repeat group's variability.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_examples(
            'openbench analyze --runs run_<baseline> grp_<challenger> --format markdown --output comparison.md',
            'openbench analyze --runs grp_<id>    # variability of one repeat group',
            'openbench analyze --results eval_a.json eval_b.json --metrics exact_match --equivalence-margin 0.05',
        ),
    )
    ana_src = ana_parser.add_mutually_exclusive_group(required=True)
    ana_src.add_argument("--runs", nargs="+", default=None, help="Run IDs (run_...) or repeat group IDs (grp_...).")
    ana_src.add_argument("--results", nargs="+", default=None, help="Evaluation result JSON files.")
    ana_parser.add_argument("--experiments-dir", type=str, default=None)
    ana_parser.add_argument("--metrics", nargs="+", default=None)
    ana_parser.add_argument("--baseline", type=str, default=None, help="Arm ID or model name of the baseline.")
    ana_parser.add_argument("--ids-file", type=str, default=None, help="Restrict the analysis to these record IDs.")
    ana_parser.add_argument("--confidence-level", type=float, default=0.95)
    ana_parser.add_argument("--resamples", type=int, default=10_000)
    ana_parser.add_argument("--permutations", type=int, default=9_999)
    ana_parser.add_argument("--seed", type=int, default=0, help="Statistical analysis seed (default 0).")
    ana_parser.add_argument("--missing-policy", choices=["complete_case", "missing_as_failure"], default="complete_case")
    ana_parser.add_argument("--correction", choices=["holm", "bh", "none"], default="holm")
    ana_parser.add_argument("--min-practical-difference", type=float, default=None)
    ana_parser.add_argument("--equivalence-margin", type=float, default=None, help="Pre-specified TOST margin (metric units).")
    ana_parser.add_argument("--alpha", type=float, default=0.05)
    ana_parser.add_argument(
        "--continuous-equivalence",
        choices=["descriptive", "empirical_bernstein"],
        default="descriptive",
        help="v0.5.0: equivalence method for continuous metrics. 'empirical_bernstein' can establish "
        "equivalence (finite-sample guarantee, needs many items); 'descriptive' reports v0.4.0 values only.",
    )
    ana_parser.add_argument("--stratified-bootstrap", action="store_true")
    ana_parser.add_argument("--stratum-tests", action="store_true")
    ana_parser.add_argument("--exploratory", action="store_true", help="Descriptive-only comparison of incompatible inputs.")
    ana_parser.add_argument("--exploratory-reason", type=str, default=None)
    ana_parser.add_argument("--output", type=str, default=None)
    ana_parser.add_argument("--format", choices=["json", "markdown", "csv", "html"], default="json",
                            help="Report format; v0.5.0 adds 'html' (one self-contained file).")

    # 14. preferences (v0.4.0 new: create, validate, analyze)
    pref_parser = subparsers.add_parser(
        "preferences",
        help="v0.4.0: local blinded pairwise human preference evaluation.",
    )
    pref_sub = pref_parser.add_subparsers(dest="preferences_action")
    pc = pref_sub.add_parser(
        "create",
        help="Create blind A/B tasks (tasks.jsonl) and a separate key file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_examples(
            'openbench preferences create --runs run_<a> run_<b> --output prefs/',
            'openbench preferences create --results eval_a.json eval_b.json --output prefs/ --csv --show-reference',
        ),
    )
    pc_src = pc.add_mutually_exclusive_group(required=True)
    pc_src.add_argument("--runs", nargs=2, default=None, metavar=("RUN_A", "RUN_B"))
    pc_src.add_argument("--results", nargs=2, default=None, metavar=("EVAL_A", "EVAL_B"))
    pc.add_argument("--experiments-dir", type=str, default=None)
    pc.add_argument("--seed", type=int, default=0, help="Preference-order seed (default 0).")
    pc.add_argument("--output", required=True, type=str, help="Output directory for tasks.jsonl.")
    pc.add_argument("--key-output", type=str, default=None, help="Key file path (default OUTPUT/tasks.key.json).")
    pc.add_argument("--rubric", type=str, default=None, help="JSON rubric file.")
    pc.add_argument("--show-reference", action="store_true", help="Include reference answers in tasks.")
    pc.add_argument("--csv", action="store_true", help="Also write tasks.csv.")
    pc.add_argument("--overwrite", action="store_true", help="Allow overwriting existing task files.")
    pv = pref_sub.add_parser("validate", help="Validate completed annotation files against a task key.")
    pv.add_argument("--key", required=True, type=str)
    pv.add_argument("--annotations", required=True, nargs="+")
    pv.add_argument("--annotators", type=str, default=None, help="Optional roster file (one annotator ID per line).")
    pv.add_argument("--format", choices=["text", "json"], default="text")
    pa = pref_sub.add_parser("analyze", help="Analyse validated annotations (win rates, agreement, optional Elo).")
    pa.add_argument("--key", required=True, nargs="+")
    pa.add_argument("--annotations", required=True, nargs="+")
    pa.add_argument("--annotators", type=str, default=None)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--confidence-level", type=float, default=0.95)
    pa.add_argument("--resamples", type=int, default=10_000)
    pa.add_argument("--elo", action="store_true", help="Also report Elo ratings (relative; see limitations).")
    pa.add_argument(
        "--compare-metric",
        type=str,
        default=None,
        help="v0.5.0: report how often this metric picks the same winner as the human majority.",
    )
    pa.add_argument("--experiments-dir", type=str, default=None, help="v0.5.0: where the key's runs are stored.")
    pa.add_argument(
        "--metric-results",
        nargs=2,
        default=None,
        metavar=("EVAL_A", "EVAL_B"),
        help="v0.5.0: evaluation files for models A and B when the key was built from --results.",
    )
    pa.add_argument("--output", type=str, default=None)
    pa.add_argument("--format", choices=["json", "markdown", "csv"], default="json")

    return parser


def _instantiate_provider_from_args_or_profile(
    provider_name: str,
    model_name: str,
    model_version: str,
    endpoint: str | None,
    gen_cfg: GenerationConfig,
    timeout_seconds: float,
    extra_prov_cfg: dict[str, Any],
):
    key = provider_name.strip().lower()
    if key == "ollama":
        return create_provider(
            "ollama",
            model_name=model_name,
            model_version=model_version,
            endpoint=endpoint or extra_prov_cfg.get("endpoint", DEFAULT_OLLAMA_ENDPOINT),
            generation_config=gen_cfg,
            timeout_seconds=timeout_seconds,
            allow_remote_http=bool(extra_prov_cfg.get("allow_remote_http", False)),
        )
    if key in {"openai_compatible", "openai", "openai-compatible"}:
        api_key_env = extra_prov_cfg.get("api_key_env", DEFAULT_API_KEY_ENV)
        if api_key_env == "[REDACTED]":  # stored runs never keep the variable name
            api_key_env = DEFAULT_API_KEY_ENV
        return create_provider(
            "openai_compatible",
            model_name=model_name,
            model_version=model_version,
            endpoint=endpoint or extra_prov_cfg.get("endpoint", DEFAULT_OPENAI_COMPATIBLE_ENDPOINT),
            api_key_env=api_key_env,
            system_prompt=extra_prov_cfg.get("system_prompt"),
            generation_config=gen_cfg,
            timeout_seconds=timeout_seconds,
            allow_remote_http=bool(extra_prov_cfg.get("allow_remote_http", False)),
        )
    if key in {"huggingface", "huggingface_transformers", "hf"}:
        return create_provider(
            "huggingface_transformers",
            model_id=model_name,
            revision=model_version,
            allow_download=bool(extra_prov_cfg.get("allow_download", False)),
            device=str(extra_prov_cfg.get("device", "cpu")),
            max_new_tokens=gen_cfg.max_tokens,
            temperature=gen_cfg.temperature,
            generation_config=gen_cfg,
            timeout_seconds=timeout_seconds,
        )
    if key == "mock":
        return create_provider(
            "mock",
            model_name=model_name,
            model_version=model_version,
            use_expected_answer=bool(extra_prov_cfg.get("use_expected_answer", True)),
            simulated_latency_ms=float(extra_prov_cfg.get("simulated_latency_ms", 2.0)),
            generation_config=gen_cfg,
            timeout_seconds=timeout_seconds,
        )
    return create_provider(
        key,
        model_name=model_name,
        model_version=model_version,
        generation_config=gen_cfg,
        timeout_seconds=timeout_seconds,
    )


def _handle_validate_dataset(dataset_path: str) -> int:
    dataset = DatasetLoader.load(dataset_path)
    summary = dataset.summary()
    categories_str = ", ".join(
        f"{cat}: {cnt}" for cat, cnt in summary["categories"].items()
    )
    print(
        f"Dataset valid: '{dataset.name}' ({len(dataset)} records; categories: {categories_str})"
    )
    return 0


def _print_audit_summary(report_dict: dict[str, Any], label: str) -> None:
    summary = report_dict["summary"]
    print(f"Quality audit [{label}]: {summary['errors']} error(s), {summary['warnings']} warning(s), {summary['info']} info")
    for f in report_dict["findings"]:
        ids = ", ".join(f["record_ids"][:5] or f["run_ids"][:5])
        more = "" if len(f["record_ids"]) <= 5 else f" (+{len(f['record_ids']) - 5} more)"
        cls = f" [{f['classification']}]" if f.get("classification") else ""
        print(f"  {f['severity'].upper():7} {f['check_id']:6} {f['message']}{cls} {('(' + ids + more + ')') if ids else ''}".rstrip())


def _handle_dataset_audit(args: argparse.Namespace) -> int:
    """`validate-dataset PATH --quality-checks` (v0.4.0). Read-only."""
    from openbench.quality import DatasetAuditor
    from openbench.reporting_v040 import render_audit
    from openbench.security import validate_safe_output_path

    embedder = None
    if getattr(args, "near_duplicate_method", "lexical") == "semantic":
        from openbench.metrics.semantic_similarity import HuggingFaceEmbeddingProvider

        try:
            embedder = HuggingFaceEmbeddingProvider(model_id=args.embedding_model)
            embedder._load_model()  # fail early, before the audit starts
        except Exception as exc:  # missing optional dependency or model
            print(
                f"Error: semantic near-duplicate detection needs the 'semantic' extra "
                f"(pip install 'openbench-ai[semantic]') and a local model: {sanitize_error_message(str(exc))}",
                file=sys.stderr,
            )
            return 2
    report = DatasetAuditor(
        near_duplicate_threshold=args.near_duplicate_threshold, metrics=args.metrics,
        embedding_provider=embedder, semantic_threshold=getattr(args, "semantic_threshold", 0.9),
    ).audit(args.dataset_path)
    payload = report.to_dict()
    try:
        dataset = DatasetLoader.load(args.dataset_path)
        print(f"Dataset loads: '{dataset.name}' ({len(dataset)} records)")
    except OpenBenchError as exc:
        print(f"Dataset does not load: {exc}", file=sys.stderr)
    _print_audit_summary(payload, args.dataset_path)
    if args.output:
        out = validate_safe_output_path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_audit(payload, args.format), encoding="utf-8")
        print(f"Audit report written to: {out}")
    return report.exit_code


def _handle_evaluate(args: argparse.Namespace) -> int:
    profile = None
    output_formats = ["json", "markdown"]

    if args.config:
        profile = ProfileLoader.load_file(args.config)
        output_formats = list(profile.output_formats)

    dataset_path = args.dataset or (profile.dataset_path if profile else None)
    if not dataset_path:
        print(
            "Error: Either --dataset or a --config profile specifying dataset.path is required.",
            file=sys.stderr,
        )
        return 2

    if profile is not None:
        config = profile.to_evaluation_config(override_metrics=args.metrics)
        if args.category:
            config.categories = list(args.category)
        if args.difficulty:
            config.difficulties = list(args.difficulty)
        if getattr(args, "ids_file", None):
            config.ids_file = args.ids_file
    else:
        metric_names = list(args.metrics) if args.metrics else list(DEFAULT_METRIC_NAMES)
        config = EvaluationConfig(
            metric_names=metric_names,
            categories=list(args.category) if args.category else None,
            difficulties=list(args.difficulty) if args.difficulty else None,
            ids_file=getattr(args, "ids_file", None),
        )

    dataset = DatasetLoader.load(dataset_path)
    response_sets = ResponseLoader.load_all(args.responses)
    evaluator = Evaluator(config=config)
    results = evaluator.evaluate_multiple(dataset, response_sets)

    if getattr(args, "confidence_intervals", False):
        from openbench.evalstats import compute_evaluation_statistics

        for res in results:
            res.statistics = compute_evaluation_statistics(
                res,
                confidence_level=args.confidence_level,
                resamples=args.resamples,
                analysis_seed=args.seed,
            )

    for res in results:
        for warning_msg in res.warnings:
            print(f"Warning [{res.model_name}]: {warning_msg}", file=sys.stderr)

    out_path = Path(args.output)
    is_single_file = out_path.suffix.lower() in {".json", ".md", ".csv"} and len(results) == 1

    history_store: ResultHistoryStore | None = None
    if args.history_file:
        history_store = ResultHistoryStore(args.history_file)
    elif profile and profile.record_history:
        base_hist_dir = out_path.parent if is_single_file else out_path
        history_store = ResultHistoryStore(base_hist_dir / "history.jsonl")

    if is_single_file:
        res = results[0]
        suffix = out_path.suffix.lower()
        if suffix == ".json":
            ReportGenerator.save_json(res, out_path)
            md_companion = out_path.with_suffix(".md")
            ReportGenerator.save_markdown(res, md_companion)
            if history_store:
                history_store.record(res, report_path=out_path)
            print(f"Saved evaluation reports: {out_path}, {md_companion}")
        elif suffix == ".md":
            ReportGenerator.save_markdown(res, out_path)
            json_companion = out_path.with_suffix(".json")
            ReportGenerator.save_json(res, json_companion)
            if history_store:
                history_store.record(res, report_path=json_companion)
            print(f"Saved evaluation reports: {out_path}, {json_companion}")
        else:
            ReportGenerator.save_csv(res, out_path)
            json_companion = out_path.with_suffix(".json")
            ReportGenerator.save_json(res, json_companion)
            if history_store:
                history_store.record(res, report_path=json_companion)
            print(f"Saved evaluation reports: {out_path}, {json_companion}")
        return 0

    out_dir = out_path
    out_dir.mkdir(parents=True, exist_ok=True)

    written_files: list[str] = []
    for res in results:
        slug = _slugify(res.model_name)
        json_file = out_dir / f"{slug}.json"
        md_file = out_dir / f"{slug}.md"
        csv_file = out_dir / f"{slug}.csv"

        if "json" in output_formats:
            ReportGenerator.save_json(res, json_file)
            written_files.append(str(json_file))
        if "markdown" in output_formats:
            ReportGenerator.save_markdown(res, md_file)
            written_files.append(str(md_file))
        if "csv" in output_formats:
            ReportGenerator.save_csv(res, csv_file)
            written_files.append(str(csv_file))

        if history_store:
            history_store.record(res, report_path=json_file if "json" in output_formats else None)

    print(
        f"Evaluated {len(results)} model(s) on '{dataset.name}' ({len(results[0].individual_results)} items). "
        f"Reports written to: {', '.join(written_files)}"
    )
    return 0


def _handle_compare(results_args: list[list[str]], output_arg: str) -> int:
    flat_paths: list[str] = [p for group in results_args for p in group]
    if len(flat_paths) < 2:
        print(
            "Error: At least two result files must be provided via --results for comparison.",
            file=sys.stderr,
        )
        return 2

    eval_results: list[EvaluationResult] = []
    for path_str in flat_paths:
        target = Path(path_str)
        if not target.exists():
            raise FileNotFoundError(f"Evaluation result file does not exist: {target}")
        raw_text = target.read_text(encoding="utf-8")
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise OpenBenchError(
                f"Malformed JSON in result file {target}: {exc.msg}"
            ) from exc
        eval_results.append(EvaluationResult.from_dict(payload))

    comparison = ModelComparator.compare(eval_results)
    out_path = Path(output_arg)

    if out_path.suffix.lower() == ".json":
        ReportGenerator.save_json(comparison, out_path)
        print(f"Comparison JSON report written to: {out_path}")
    elif out_path.suffix.lower() == ".csv":
        ReportGenerator.save_csv(comparison, out_path)
        print(f"Comparison CSV report written to: {out_path}")
    elif out_path.suffix.lower() == ".md":
        ReportGenerator.save_markdown(comparison, out_path)
        json_companion = out_path.with_suffix(".json")
        ReportGenerator.save_json(comparison, json_companion)
        print(f"Comparison reports written to: {out_path}, {json_companion}")
    else:
        out_path.mkdir(parents=True, exist_ok=True)
        md_file = out_path / "comparison.md"
        json_file = out_path / "comparison.json"
        ReportGenerator.save_markdown(comparison, md_file)
        ReportGenerator.save_json(comparison, json_file)
        print(f"Comparison reports written to: {md_file}, {json_file}")

    return 0


def _handle_list_datasets(args: argparse.Namespace) -> int:
    catalog = DatasetCatalog(args.path)
    entries = catalog.list_datasets(category=args.category, difficulty=args.difficulty)
    if args.format == "json":
        print(json.dumps([e.to_dict() for e in entries], indent=2))
        return 0

    total_records = sum(e.total_records for e in entries)
    print(f"Found {len(entries)} dataset(s) ({total_records} total records):")
    for entry in entries:
        cats = ", ".join(f"{k}:{v}" for k, v in entry.categories.items())
        diffs = ", ".join(f"{k}:{v}" for k, v in entry.difficulties.items())
        print(
            f"  - {entry.name} (v{entry.version}): {entry.total_records} records "
            f"[categories: {cats}] [difficulties: {diffs}] ({entry.relative_path})"
        )
    return 0


def _handle_inspect_dataset(args: argparse.Namespace) -> int:
    info = DatasetCatalog.inspect_dataset(
        args.dataset_path,
        category=args.category,
        difficulty=args.difficulty,
        sample_limit=args.samples,
    )
    if args.format == "json":
        print(json.dumps(info, indent=2))
        return 0

    print(f"Dataset Inspection: {info['name']} (v{info['version']})")
    print(f"  Source Path   : {info['source_path']}")
    print(f"  SHA-256       : {info['checksum']}")
    print(f"  Total Records : {info['total_records']}")
    print(f"  Categories    : {info['categories']}")
    print(f"  Difficulties  : {info['difficulties']}")
    print(f"  Languages     : {info['languages']}")
    if info["sample_records"]:
        print(f"  Sample Records ({len(info['sample_records'])}):")
        for sample in info["sample_records"]:
            print(f"    * [{sample['id']}] ({sample['category']}): {sample['prompt']}")
    return 0


def _handle_dataset_stats(args: argparse.Namespace) -> int:
    stats = DatasetCatalog.compute_stats(
        args.dataset_path,
        category=args.category,
        difficulty=args.difficulty,
    )
    if args.format == "json":
        print(json.dumps(stats, indent=2))
        return 0

    print(f"Dataset Statistics: {stats['name']} (v{stats['version']})")
    print(f"  Files Count      : {stats['files_count']}")
    print(f"  Total Records    : {stats['total_records']}")
    print(f"  SHA-256 Checksum : {stats['checksum']}")
    print(f"  Categories       : {stats['categories']}")
    print(f"  Difficulties     : {stats['difficulties']}")
    print(f"  Languages        : {stats['languages']}")
    print(f"  Prompt Length    : {stats['prompt_char_length']}")
    print(f"  Answer Length    : {stats['expected_answer_char_length']}")
    return 0


def _handle_history(args: argparse.Namespace) -> int:
    if not args.history_action:
        print("Error: Specify a history action ('list' or 'show').", file=sys.stderr)
        return 2

    store = ResultHistoryStore(args.store)
    if args.history_action == "list":
        entries = store.list_entries(model_name=args.model, dataset_name=args.dataset)
        if args.format == "json":
            print(json.dumps([e.to_dict() for e in entries], indent=2))
            return 0
        if not entries:
            print("No historical evaluation runs found.")
            return 0
        print(f"Recorded Evaluation Runs ({len(entries)}):")
        for e in entries:
            scores_str = ", ".join(f"{k}={v:.4f}" for k, v in e.result_summary.items())
            print(
                f"  - {e.evaluation_id} | {e.timestamp} | {e.model_name} (v{e.model_version}) "
                f"| dataset={e.dataset_name} | {scores_str}"
            )
        return 0

    if args.history_action == "show":
        try:
            entry = store.get_entry(args.evaluation_id)
        except KeyError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        if args.format == "json":
            print(json.dumps(entry.to_dict(), indent=2))
            return 0
        print(f"Evaluation Run: {entry.evaluation_id}")
        print(f"  Timestamp     : {entry.timestamp}")
        print(f"  Model         : {entry.model_name} (v{entry.model_version})")
        print(f"  Dataset       : {entry.dataset_name} (v{entry.dataset_version})")
        print(f"  Checksum      : {entry.dataset_checksum}")
        print(f"  Evaluated     : {entry.evaluated_items} / {entry.total_items} (missing: {entry.missing_responses_count})")
        print(f"  Scores        : {entry.result_summary}")
        if entry.report_path:
            print(f"  Report Path   : {entry.report_path}")
        return 0

    return 2


def _handle_export(args: argparse.Namespace) -> int:
    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input result file does not exist: {in_path}")

    raw_text = in_path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise OpenBenchError(f"Malformed JSON in {in_path}: {exc.msg}") from exc

    out_path = Path(args.output)
    if "models" in payload and "highest_by_metric" in payload:
        from openbench.comparison import MetricLeader, ModelComparisonEntry

        entries = [
            ModelComparisonEntry(
                model_name=m["model_name"],
                model_version=m["model_version"],
                dataset_name=m.get("dataset_name", "unknown"),
                total_dataset_items=int(m.get("total_dataset_items", 0)),
                evaluated_items=int(m.get("evaluated_items", 0)),
                missing_responses_count=int(m.get("missing_responses_count", 0)),
                exact_match_score=m.get("exact_match"),
                accuracy_score=m.get("accuracy"),
                keyword_coverage_score=m.get("keyword_coverage"),
                metric_scores=dict(m.get("metric_scores", {})),
                weighted_score=m.get("weighted_score"),
            )
            for m in payload["models"]
        ]
        leaders = {
            k: MetricLeader(
                metric_name=v["metric_name"],
                highest_score=float(v["highest_score"]),
                leading_models=list(v["leading_models"]),
                runner_up_score=v.get("runner_up_score"),
                score_margin=float(v.get("score_margin", 0.0)),
            )
            for k, v in payload.get("highest_by_metric", {}).items()
        }
        target_obj: EvaluationResult | ComparisonResult = ComparisonResult(
            comparison_timestamp=str(payload.get("comparison_timestamp", "")),
            dataset_name=str(payload.get("dataset_name", "unknown")),
            total_dataset_items=int(payload.get("total_dataset_items", 0)),
            models=entries,
            metrics_compared=list(payload.get("metrics_compared", [])),
            highest_by_metric=leaders,
            warnings=list(payload.get("warnings", [])),
            methodology_note=str(payload.get("methodology_note", "")),
            weights_used=payload.get("weights_used"),
        )
    else:
        target_obj = EvaluationResult.from_dict(payload)

    if args.format == "csv":
        ReportGenerator.save_csv(target_obj, out_path, mode=args.mode)
    elif args.format == "markdown":
        ReportGenerator.save_markdown(target_obj, out_path)
    else:
        ReportGenerator.save_json(target_obj, out_path)

    print(f"Exported {args.format.upper()} report to: {out_path}")
    return 0


def _handle_run(args: argparse.Namespace) -> int:
    """Execute `openbench run` via ExecutionEngine and ExperimentManager."""
    profile = ProfileLoader.load_file(args.config) if args.config else None

    dataset_path = args.dataset or (profile.dataset_path if profile else None)
    if not dataset_path:
        print(
            "Error: Specify --dataset or provide a --config profile with dataset.path.",
            file=sys.stderr,
        )
        return 2

    prov_cfg = dict(profile.provider_config) if profile else {}
    if getattr(args, "api_key_env", None):
        prov_cfg["api_key_env"] = args.api_key_env
    provider_name = args.provider or prov_cfg.get("name")
    if not provider_name:
        print(
            "Error: Specify --provider (e.g. ollama, huggingface, mock) or set provider.name in --config.",
            file=sys.stderr,
        )
        return 2

    model_name = args.model or prov_cfg.get("model")
    if not model_name:
        print(
            "Error: Specify --model or set provider.model in --config.",
            file=sys.stderr,
        )
        return 2

    model_version = (
        args.model_version
        or prov_cfg.get("model_version")
        or prov_cfg.get("revision")
        or "1.0"
    )
    gen_cfg = profile.generation_config if profile else GenerationConfig()

    base_exec = profile.execution_config if profile else ExecutionConfig()
    timeout_sec = args.timeout if args.timeout is not None else base_exec.timeout_seconds
    concurrency = args.concurrency if args.concurrency is not None else base_exec.concurrency
    max_attempts = (
        args.max_attempts
        if args.max_attempts is not None
        else base_exec.retry_policy.max_attempts
    )
    use_cache = False if args.no_cache else base_exec.use_cache

    retry_policy = RetryPolicy(
        max_attempts=max_attempts,
        initial_backoff_seconds=base_exec.retry_policy.initial_backoff_seconds,
        backoff_multiplier=base_exec.retry_policy.backoff_multiplier,
        max_backoff_seconds=base_exec.retry_policy.max_backoff_seconds,
        retry_on_timeouts=base_exec.retry_policy.retry_on_timeouts,
    )
    exec_cfg = ExecutionConfig(
        timeout_seconds=timeout_sec,
        concurrency=concurrency,
        retry_policy=retry_policy,
        use_cache=use_cache,
        estimate_missing_tokens=base_exec.estimate_missing_tokens,
        privacy_mode=base_exec.privacy_mode,
        pricing=base_exec.pricing,
        max_requests_per_minute=(
            args.max_requests_per_minute
            if getattr(args, "max_requests_per_minute", None) is not None
            else base_exec.max_requests_per_minute
        ),
        max_retry_after_seconds=base_exec.max_retry_after_seconds,
    )

    provider = _instantiate_provider_from_args_or_profile(
        provider_name=str(provider_name),
        model_name=str(model_name),
        model_version=str(model_version),
        endpoint=args.endpoint,
        gen_cfg=gen_cfg,
        timeout_seconds=timeout_sec,
        extra_prov_cfg=prov_cfg,
    )

    if profile is not None:
        eval_cfg = profile.to_evaluation_config(override_metrics=args.metrics)
    else:
        eval_cfg = EvaluationConfig(
            metric_names=list(args.metrics) if args.metrics else list(DEFAULT_METRIC_NAMES)
        )

    dataset = DatasetLoader.load(dataset_path)
    cache_dir = args.cache_dir or (profile.cache_dir if profile else None)
    cache = ResponseCache(cache_dir=cache_dir, enabled=use_cache)

    exp_dir = args.experiments_dir or (profile.experiments_dir if profile else None)
    manager = ExperimentManager(experiments_root=exp_dir)

    repeat_count = profile.repeat_count if profile else 1
    if getattr(args, "repetitions", None) is not None:
        if args.repetitions < 1:
            print("Error: --repetitions must be >= 1.", file=sys.stderr)
            return 2
        repeat_count = args.repetitions
    base_seed = args.seed if getattr(args, "seed", None) is not None else gen_cfg.seed

    def _first_error(rec) -> str | None:
        """v0.5.0: the first failure reason recorded in the run's logs, for the console."""
        logs = Path(rec.run_dir) / "logs.jsonl"
        try:
            for line in logs.read_text(encoding="utf-8").splitlines():
                entry = json.loads(line) if line.strip() else {}
                if entry.get("event") == "failed":
                    return f"[{entry.get('error_type')}] {entry.get('error_message')}"
        except (OSError, json.JSONDecodeError):
            return None
        return None

    def _report(rec) -> None:
        group_note = (
            f" [group {rec.repeat_group_id} repetition {rec.repetition_index + 1}/{repeat_count}]"
            if rec.repeat_group_id
            else ""
        )
        print(
            f"Run '{rec.run_id}' [{rec.status}]{group_note}: {rec.completed_items}/{rec.total_items} completed "
            f"(provider={rec.model_provider}, model={rec.model_name}, scores={rec.aggregate_scores}) -> {rec.run_dir}"
        )
        if rec.failed_items:
            reason = _first_error(rec)
            if reason:
                print(f"  {rec.failed_items} item(s) failed. First error: {reason}", file=sys.stderr)

    group = manager.execute_repeated(
        dataset=dataset,
        provider=provider,
        repetitions=repeat_count,
        generation_base_seed=base_seed,
        evaluation_config=eval_cfg,
        execution_config=exec_cfg,
        cache=cache,
        experiment_id=getattr(args, "experiment_id", None),
        parent_experiment_id=getattr(args, "parent_experiment_id", None),
        on_repetition=_report,
    )
    run_records = group.records
    eval_results = group.evaluations
    if group.repeat_group_id:
        print(
            f"Repeat group '{group.repeat_group_id}': {group.completed_repetitions}/{group.planned_repetitions} "
            f"repetitions completed; index: {group.index_path}"
        )
    seed_status = run_records[0].lineage.get("seed_status")
    if seed_status and seed_status != "not_supplied":
        print(f"Generation seed status: {seed_status} (a seed does not guarantee deterministic output)")

    if repeat_count > 1:
        variance_summary = BenchmarkAnalytics.analyze_repeated_runs(eval_results)
        var_file = Path(run_records[-1].run_dir) / "repeated_runs_analysis.json"
        var_file.write_text(json.dumps(variance_summary, indent=2) + "\n", encoding="utf-8")
        print(f"Repeated-run variance analysis (n={repeat_count}) saved to: {var_file}")

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        last_eval = eval_results[-1]
        slug = _slugify(last_eval.model_name)
        ReportGenerator.save_json(last_eval, out_dir / f"{slug}.json")
        ReportGenerator.save_markdown(last_eval, out_dir / f"{slug}.md")

    return 0 if all(r.status == "completed" for r in run_records) else 1


def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: max(width - 1, 1)] + "…"


def _compact_runs_table(runs: list[Any]) -> list[str]:
    """Aligned table for `openbench runs list` that fits in about 120 columns (v0.5.0)."""
    header = ("RUN ID", "STATUS", "MODEL", "ITEMS", "SCORE", "GROUP")
    rows: list[tuple[str, ...]] = []
    for r in runs:
        model = f"{r.model_provider}:{r.model_name}" + (" [mock]" if r.is_mock_provider else "")
        scores = r.aggregate_scores or {}
        if scores:
            first = sorted(scores)[0]
            value = scores[first]
            score = f"{first}={value:.3f}" if isinstance(value, (int, float)) else f"{first}={value}"
        else:
            score = "-"
        group = f"{r.repeat_group_id}#{r.repetition_index}" if r.repeat_group_id else "-"
        rows.append(
            (
                r.run_id,
                str(r.status),
                _truncate(model, 34),
                f"{r.completed_items}/{r.total_items}",
                _truncate(score, 24),
                group,
            )
        )
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(header)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = ["  " + fmt.format(*header).rstrip()]
    lines.extend("  " + fmt.format(*row).rstrip() for row in rows)
    lines.append("  (SCORE shows the first metric alphabetically; use --wide or --format json for all scores.)")
    return lines


def _handle_runs(args: argparse.Namespace) -> int:
    """Execute `openbench runs list|show|verify|resume|compare`."""
    if not args.runs_action:
        print("Error: Specify a runs subcommand (list, show, verify, resume, compare).", file=sys.stderr)
        return 2

    manager = ExperimentManager(experiments_root=getattr(args, "experiments_dir", None))

    from openbench.lineage import is_repeat_group_id

    run_id_arg = getattr(args, "run_id", None)
    if run_id_arg and is_repeat_group_id(run_id_arg) and args.runs_action in {"show", "verify", "resume"}:
        return _handle_group_action(args, manager)

    if args.runs_action == "list":
        runs = manager.list_runs(
            status=args.status, provider=args.provider, repeat_group_id=getattr(args, "group", None)
        )
        if args.format == "json":
            print(json.dumps([r.to_dict() for r in runs], indent=2))
            return 0
        if not runs:
            print("No experiment runs found.")
            return 0
        print(f"Experiment Runs ({len(runs)}):")
        if getattr(args, "wide", False):
            for r in runs:
                mock_tag = " [MOCK]" if r.is_mock_provider else ""
                group_tag = (
                    f" | group={r.repeat_group_id}#{r.repetition_index}" if r.repeat_group_id else ""
                )
                print(
                    f"  - {r.run_id} | {r.status} | origin={r.execution_origin} | "
                    f"{r.model_provider}:{r.model_name}{mock_tag} | dataset={r.dataset_identifier} "
                    f"({r.completed_items}/{r.total_items}) | scores={r.aggregate_scores}{group_tag}"
                )
            return 0
        for line in _compact_runs_table(runs):
            print(line)
        return 0

    if args.runs_action == "show":
        try:
            details = manager.inspect_run(args.run_id)
        except KeyError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        if args.format == "json":
            print(json.dumps(details, indent=2))
            return 0
        exp = details["experiment"]
        integ = details["integrity"]
        print(f"Experiment Run: {exp['run_id']} ({exp['status']})")
        print(f"  Origin        : {exp['execution_origin']}")
        print(f"  Provider      : {exp['model_provider']} (is_mock={exp['is_mock_provider']})")
        print(f"  Model         : {exp['model_name']} (v{exp['model_version']})")
        print(f"  Dataset       : {exp['dataset_identifier']} (checksum={exp['dataset_checksum']})")
        print(f"  Items         : {exp['completed_items']} / {exp['total_items']} completed ({exp['failed_items']} failed)")
        print(f"  Scores        : {exp['aggregate_scores']}")
        print(f"  Integrity     : {'VALID' if integ['valid'] else 'INVALID'}")
        print(f"  Run Directory : {exp['run_dir']}")
        return 0

    if args.runs_action == "verify":
        try:
            check = manager.verify_run(args.run_id)
        except KeyError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        if getattr(args, "quality_checks", False):
            return _verify_with_audit(args, manager, check.to_dict(), check.valid)
        if args.format == "json":
            print(json.dumps(check.to_dict(), indent=2))
            return 0 if check.valid else 1
        status_label = "VALID" if check.valid else "INVALID"
        print(
            f"Run Integrity Verification [{check.run_id}]: {status_label} "
            f"(origin={check.execution_origin}, schema={check.schema_version})"
        )
        if check.missing_artifacts:
            print(f"  Missing Artifacts      : {check.missing_artifacts}")
        if check.changed_artifacts:
            print(f"  Changed Artifacts      : {check.changed_artifacts}")
        if check.inconsistent_metadata:
            print(f"  Inconsistent Metadata  : {check.inconsistent_metadata}")
        for w in check.warnings:
            print(f"  Note: {w}")
        return 0 if check.valid else 1

    if args.runs_action == "resume":
        try:
            info = manager.inspect_run(args.run_id)["experiment"]
        except KeyError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        provider = _provider_from_stored_run(info)
        cache = ResponseCache(cache_dir=args.cache_dir, enabled=_stored_use_cache(info))
        rec, _, _ = manager.resume_run(args.run_id, provider=provider, cache=cache)
        print(
            f"Resumed run '{rec.run_id}' -> status={rec.status} "
            f"({rec.completed_items}/{rec.total_items} completed, scores={rec.aggregate_scores})"
        )
        return 0 if rec.status == "completed" else 1

    if args.runs_action == "compare":
        if len(args.run_ids) < 2:
            print("Error: Provide at least two RUN_IDs to compare.", file=sys.stderr)
            return 2
        try:
            comparison = manager.compare_runs(args.run_ids)
        except KeyError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        if args.output:
            out_p = Path(args.output)
            if out_p.suffix.lower() == ".json":
                ReportGenerator.save_json(comparison, out_p)
            elif out_p.suffix.lower() == ".csv":
                ReportGenerator.save_csv(comparison, out_p)
            else:
                ReportGenerator.save_markdown(comparison, out_p)
            print(f"Saved run comparison report to: {out_p}")
            return 0

        if args.format == "json":
            print(ReportGenerator.to_json(comparison))
        else:
            print(ReportGenerator.comparison_to_markdown(comparison))
        return 0

    return 2


def _generation_config_from_stored(gen: dict[str, Any]) -> GenerationConfig:
    """Rebuild a GenerationConfig from a stored (redacted) generation configuration."""
    stops = gen.get("stop_sequences")
    seed = gen.get("seed")
    return GenerationConfig(
        temperature=float(gen.get("temperature", 0.0)),
        max_tokens=int(gen.get("max_tokens", 256)),
        top_p=float(gen.get("top_p", 1.0)),
        seed=int(seed) if isinstance(seed, int) and not isinstance(seed, bool) else None,
        stop_sequences=[str(x) for x in stops] if isinstance(stops, list) else [],
        stream=bool(gen.get("stream", False)),
    )


def _provider_from_stored_run(info: dict[str, Any]):
    gen = info.get("generation_configuration") or {}
    return _instantiate_provider_from_args_or_profile(
        provider_name=info["model_provider"],
        model_name=info["model_name"],
        model_version=info["model_version"],
        endpoint=gen.get("endpoint"),
        gen_cfg=_generation_config_from_stored(gen),
        timeout_seconds=float((info.get("execution_configuration") or {}).get("timeout_seconds", 30.0)),
        extra_prov_cfg=gen,
    )


def _stored_use_cache(info: dict[str, Any]) -> bool:
    """v0.4.0 runs honour their recorded cache setting on resume; v0.3.x runs keep using the cache."""
    lineage = info.get("lineage") or {}
    if lineage and not lineage.get("legacy_run"):
        return bool((info.get("execution_configuration") or {}).get("use_cache", True))
    return True


def _handle_group_action(args: argparse.Namespace, manager: ExperimentManager) -> int:
    """`runs show|verify|resume grp_...` (v0.4.0 repeat groups)."""
    group_id = args.run_id
    if args.runs_action == "show":
        try:
            group = manager.load_group(group_id)
        except KeyError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        payload = {
            "index": group["index"],
            "members": [m.to_dict() for m in group["members"]],
        }
        if args.format == "json":
            print(json.dumps(payload, indent=2))
            return 0
        idx = group["index"]
        print(f"Repeat Group: {group_id} ({idx['status']})")
        print(f"  Repetitions   : {idx['completed_repetitions']} completed / {idx['planned_repetitions']} planned")
        print(f"  Experiment ID : {idx.get('experiment_id')}")
        print(f"  Base seed     : {idx.get('generation_base_seed')}")
        for m in group["members"]:
            print(
                f"  - #{m.repetition_index} {m.run_id} [{m.status}] seed={m.lineage.get('repetition_seed')} "
                f"seed_status={m.lineage.get('seed_status')} scores={m.aggregate_scores}"
            )
        if idx.get("missing_repetition_indices"):
            print(f"  Missing repetitions: {idx['missing_repetition_indices']}")
        for w in idx.get("rebuild_warnings", []):
            print(f"  Warning: {w}")
        return 0
    if args.runs_action == "verify":
        result = manager.verify_group(group_id)
        if getattr(args, "quality_checks", False):
            return _verify_with_audit(args, manager, result, result["valid"], group=True)
        if args.format == "json":
            print(json.dumps(result, indent=2))
            return 0 if result["valid"] else 1
        print(
            f"Repeat Group Verification [{group_id}]: {'VALID' if result['valid'] else 'INVALID'} "
            f"({result['member_count']} member(s), {result['planned_repetitions']} planned)"
        )
        for problem in result["problems"]:
            print(f"  Problem: {problem}")
        for w in result["warnings"]:
            print(f"  Note: {w}")
        return 0 if result["valid"] else 1
    # resume
    try:
        group = manager.load_group(group_id)
    except KeyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if not group["members"]:
        print(f"Error: repeat group '{group_id}' has no member runs to resume from.", file=sys.stderr)
        return 1
    info = group["members"][0].to_dict()
    provider = _provider_from_stored_run(info)
    cache = ResponseCache(cache_dir=args.cache_dir, enabled=_stored_use_cache(info))
    result = manager.resume_group(group_id, provider=provider, cache=cache)
    for rec in result.records:
        print(f"  - #{rec.repetition_index} {rec.run_id} [{rec.status}] scores={rec.aggregate_scores}")
    print(
        f"Resumed repeat group '{group_id}': {result.completed_repetitions}/{result.planned_repetitions} "
        "repetitions completed"
    )
    return 0 if result.is_complete else 1


def _verify_with_audit(args: argparse.Namespace, manager: ExperimentManager, check: dict[str, Any], valid: bool, group: bool = False) -> int:
    """`runs verify ... --quality-checks` (v0.4.0): integrity plus EQ audit; read-only."""
    from openbench.quality import ExperimentAuditor
    from openbench.reporting_v040 import render_audit
    from openbench.security import validate_safe_output_path

    auditor = ExperimentAuditor(manager)
    report = auditor.audit_group(args.run_id) if group else auditor.audit_run(args.run_id)
    payload = report.to_dict()
    if args.format == "json":
        print(json.dumps({"integrity": check, "quality_audit": payload}, indent=2))
    else:
        print(f"Integrity [{args.run_id}]: {'VALID' if valid else 'INVALID'}")
        _print_audit_summary(payload, args.run_id)
    if args.audit_output:
        out = validate_safe_output_path(args.audit_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_audit(payload, "json"), encoding="utf-8")
    return 0 if valid and report.exit_code == 0 else 1


def _handle_analyze(args: argparse.Namespace) -> int:
    """`openbench analyze` (v0.4.0): paired comparison or repeated-run variability; no provider calls."""
    from openbench.dataset import load_ids_file
    from openbench.paired import ComparisonArm, ComparisonPolicy, PairedComparisonEngine
    from openbench.reporting_v040 import render_analysis
    from openbench.security import redact_sensitive_data, validate_safe_output_path
    from openbench.stats import decompose_variability

    if args.exploratory_reason and not args.exploratory:
        print("Error: --exploratory-reason requires --exploratory.", file=sys.stderr)
        return 2
    if args.exploratory and not (args.exploratory_reason and args.exploratory_reason.strip()):
        print("Error: --exploratory requires a non-empty --exploratory-reason.", file=sys.stderr)
        return 2

    arms = []
    if args.runs:
        manager = ExperimentManager(experiments_root=args.experiments_dir)
        for identifier in args.runs:
            try:
                arms.append(ComparisonArm.from_experiments(manager, identifier))
            except KeyError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1
    else:
        for path in args.results:
            arms.append(ComparisonArm.from_evaluation_file(path))
        labels = [a.arm_id for a in arms]
        if len(set(labels)) != len(labels):
            for i, arm in enumerate(arms):
                arm.arm_id = f"{arm.arm_id}#{i + 1}"

    selected = None
    if args.ids_file:
        selected = tuple(load_ids_file(args.ids_file).selected_ids)

    if len(arms) == 1:
        arm = arms[0]
        if arm.repetitions < 2:
            print(
                "Error: analyze needs at least two arms to compare, or one repeat group "
                "(grp_...) for a variability analysis.",
                file=sys.stderr,
            )
            return 2
        payload = decompose_variability(
            arm.evaluations,
            args.metrics,
            analysis_seed=args.seed,
            confidence_level=args.confidence_level,
            resamples=args.resamples,
            arm_label=arm.arm_id,
        )
        payload["arm"] = arm.summary()
        payload["provider_calls"] = 0
    else:
        policy = ComparisonPolicy(
            missing_policy=args.missing_policy,
            correction=args.correction,
            confidence_level=args.confidence_level,
            resamples=args.resamples,
            permutations=args.permutations,
            analysis_seed=args.seed,
            alpha=args.alpha,
            min_practical_difference=args.min_practical_difference,
            equivalence_margin=args.equivalence_margin,
            stratified_bootstrap=args.stratified_bootstrap,
            stratum_tests=args.stratum_tests,
            exploratory=args.exploratory,
            exploratory_reason=args.exploratory_reason,
            baseline=args.baseline,
            selected_ids=selected,
            continuous_equivalence=getattr(args, "continuous_equivalence", "descriptive"),
        )
        result = PairedComparisonEngine().compare(arms, args.metrics, policy)
        payload = result.to_dict()
        payload["variability"] = {
            arm.arm_id: decompose_variability(
                arm.evaluations,
                args.metrics,
                analysis_seed=args.seed,
                confidence_level=args.confidence_level,
                resamples=args.resamples,
                arm_label=arm.arm_id,
            )
            for arm in arms
            if arm.repetitions >= 2
        }

    payload = redact_sensitive_data(payload)
    if args.format == "html":
        from openbench.html_report import render_html

        text = render_html(payload)
    else:
        text = render_analysis(payload, args.format)
    if args.output:
        out = validate_safe_output_path(args.output)
        for parent in out.resolve().parents:
            if parent.name.startswith("run_") and (parent / "reproducibility.json").exists():
                print("Error: analyses are never written into run directories; choose another --output.", file=sys.stderr)
                return 2
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"Analysis ({payload.get('analysis')}, {args.format}) written to: {out}")
        if payload.get("analysis_mode") == "exploratory":
            print(f"Warning: {payload.get('banner')}", file=sys.stderr)
    else:
        print(text)
    return 0


def _read_roster(path: str | None) -> list[str] | None:
    if not path:
        return None
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def _metric_human_agreement(args: argparse.Namespace, keys: list[dict[str, Any]], annotations: list[Any]) -> dict[str, Any]:
    """v0.5.0 (#46): agreement between a metric's per-task preference and the human majority."""
    from openbench.agreement import agreement_report, metric_preference
    from openbench.itemscores import collect_item_scores
    from openbench.paired import ComparisonArm
    from openbench.preferences import MODEL_SPACE

    metric = args.compare_metric
    tasks = {t["task_id"]: t for k in keys for t in k["tasks"]}
    evals: dict[str, Any] = {}
    for key in keys:
        for side_index, side in enumerate(("A", "B")):
            source_id = key["models"][side]["source_id"]
            if source_id in evals:
                continue
            if args.metric_results:
                if len(keys) != 1:
                    raise OpenBenchError("--metric-results supports a single key; use runs for several keys")
                evals[source_id] = ComparisonArm.from_evaluation_file(args.metric_results[side_index]).evaluations
            else:
                manager = ExperimentManager(experiments_root=args.experiments_dir)
                evals[source_id] = [manager.load_evaluation(source_id)]
    scores = {sid: collect_item_scores(ev, metric) for sid, ev in evals.items()}
    human: dict[str, list[str]] = {}
    for a in annotations:
        task = tasks[a.task_id]
        human.setdefault(a.task_id, []).append(MODEL_SPACE[task["display_order"]].get(a.outcome, a.outcome))
    automatic: dict[str, str | None] = {}
    categories: dict[str, str] = {}
    for tid, task in tasks.items():
        obs_a = scores[task["model_a_id"]].get(task["record_id"])
        obs_b = scores[task["model_b_id"]].get(task["record_id"])
        automatic[tid] = metric_preference(obs_a.mean if obs_a else None, obs_b.mean if obs_b else None)
        categories[tid] = task.get("category", "unknown")
    return agreement_report(
        human, automatic, categories=categories, automatic_name=f"metric:{metric}",
        analysis_seed=args.seed, confidence_level=args.confidence_level, resamples=args.resamples,
    )


def _handle_preferences(args: argparse.Namespace) -> int:
    """`openbench preferences create|validate|analyze` (v0.4.0)."""
    from openbench import preferences as prefs
    from openbench.reporting_v040 import render_preferences
    from openbench.security import validate_safe_output_path

    action = getattr(args, "preferences_action", None)
    if not action:
        print("Error: Specify a preferences subcommand (create, validate, analyze).", file=sys.stderr)
        return 2
    if action == "create":
        sources = []
        if args.runs:
            manager = ExperimentManager(experiments_root=args.experiments_dir)
            for rid in args.runs:
                try:
                    rec = manager.load_run_record(rid)
                    ev = manager.load_evaluation(rid)
                except KeyError as exc:
                    print(f"Error: {exc}", file=sys.stderr)
                    return 1
                sources.append(prefs.PreferenceSource.from_evaluation(rec.run_id, ev))
        else:
            from openbench.paired import ComparisonArm

            for path in args.results:
                ev = ComparisonArm.from_evaluation_file(path).evaluations[0]
                sources.append(prefs.PreferenceSource.from_evaluation(Path(path).stem, ev))
            if sources[0].source_id == sources[1].source_id:
                sources[0].source_id += "#1"
                sources[1].source_id += "#2"
        rubric, instructions = ([], None)
        if args.rubric:
            rubric, instructions = prefs.load_rubric(args.rubric)
        builder = prefs.PreferenceTaskBuilder(args.seed, rubric, instructions, args.show_reference)
        task_set = builder.build(sources[0], sources[1])
        written = prefs.write_task_set(task_set, args.output, args.key_output, write_csv=args.csv, overwrite=args.overwrite)
        print(f"Created {len(task_set.tasks)} blind task(s): {written['tasks']}")
        print(f"Unblinded key (keep away from annotators): {written['key']}")
        for w in task_set.leakage_warnings:
            print(f"Warning: possible identity leak in task text ({len(w['task_ids'])} task(s)): {', '.join(w['task_ids'][:10])}", file=sys.stderr)
        return 0

    roster = _read_roster(getattr(args, "annotators", None))
    if action == "validate":
        key = prefs.load_key(args.key)
        report = prefs.AnnotationImporter(key, annotator_roster=roster).load(args.annotations)
        if args.format == "json":
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(f"Annotation validation: {'VALID' if report.valid else 'INVALID'} ({report.records_read} record(s) read)")
            for e in report.errors:
                print(f"  Error [{e['code']}] {e.get('file', '')}:{e.get('line')}: {e['message']}")
            for w in report.warnings:
                print(f"  Warning [{w['code']}] line {w.get('line')}: {w['message']}")
        return 0 if report.valid else 1

    keys = [prefs.load_key(k) for k in args.key]
    merged = {"tasks": [t for k in keys for t in k["tasks"]], "creation_parameters": keys[0].get("creation_parameters", {})}
    report = prefs.AnnotationImporter(merged, annotator_roster=roster).load(args.annotations)
    if not report.valid:
        print(f"Error: {len(report.errors)} annotation error(s); run 'preferences validate' for details.", file=sys.stderr)
        for e in report.errors[:20]:
            print(f"  [{e['code']}] {e.get('file', '')}:{e.get('line')}: {e['message']}", file=sys.stderr)
        return 1
    analysis = prefs.PreferenceAnalyzer(keys, analysis_seed=args.seed, confidence_level=args.confidence_level,
                                        resamples=args.resamples).analyze(report.annotations, elo=args.elo)
    analysis["validation_warnings"] = report.warnings
    if getattr(args, "compare_metric", None):
        try:
            analysis["metric_agreement"] = _metric_human_agreement(args, keys, report.annotations)
        except (KeyError, OpenBenchError, FileNotFoundError) as exc:
            print(f"Error: {sanitize_error_message(str(exc))}", file=sys.stderr)
            return 1
    from openbench.integrity import compute_file_sha256

    analysis["provenance"]["inputs"] += [
        {"annotation_file": str(a), "sha256": compute_file_sha256(a)} for a in args.annotations
    ] + [{"key_file": str(k), "sha256": compute_file_sha256(k)} for k in args.key]
    text = render_preferences(analysis, args.format)
    if args.output:
        out = validate_safe_output_path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"Preference analysis ({args.format}) written to: {out}")
    else:
        print(text)
    return 0


def _handle_providers(args: argparse.Namespace) -> int:
    """Execute `openbench providers list|check`."""
    if not args.providers_action:
        print("Error: Specify a providers subcommand ('list' or 'check').", file=sys.stderr)
        return 2

    if args.providers_action == "list":
        providers = list_registered_providers()
        if args.format == "json":
            print(json.dumps(providers, indent=2))
            return 0
        print(f"Registered Model Providers ({len(providers)}):")
        for p in providers:
            flags = []
            if p["is_mock"]:
                flags.append("mock")
            if p["is_experimental"]:
                flags.append("experimental")
            source = p.get("source")
            if isinstance(source, dict):
                flags.append(f"plugin: {source.get('distribution') or source.get('entry_point')}")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            print(f"  - {p['provider_name']}{flag_str}: {p['description']}")
        return 0

    if args.providers_action == "check":
        p_name = args.provider_name.strip().lower()
        default_model = args.model or ("llama3.2" if p_name == "ollama" else "test-model")
        provider = _instantiate_provider_from_args_or_profile(
            provider_name=p_name,
            model_name=default_model,
            model_version="latest",
            endpoint=args.endpoint,
            gen_cfg=GenerationConfig(),
            timeout_seconds=5.0,
            extra_prov_cfg={},
        )
        health = provider.check_connection()
        if args.format == "json":
            print(json.dumps(health.to_dict(), indent=2))
            return 0 if health.available else 1
        state = "AVAILABLE" if health.available else "UNAVAILABLE"
        print(f"Provider Check [{health.provider_name}]: {state} — {health.message}")
        return 0 if health.available else 1

    return 2


def _write_or_print(text: str, output: str | None, label: str) -> None:
    from openbench.security import validate_safe_output_path

    if output:
        out = validate_safe_output_path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"{label} written to: {out}")
    else:
        print(text, end="" if text.endswith("\n") else "\n")


def _load_arms(args: argparse.Namespace) -> list[Any]:
    from openbench.paired import ComparisonArm

    arms = []
    if getattr(args, "runs", None):
        manager = ExperimentManager(experiments_root=args.experiments_dir)
        for identifier in args.runs:
            arms.append(ComparisonArm.from_experiments(manager, identifier))
    else:
        for path in args.results:
            arms.append(ComparisonArm.from_evaluation_file(path))
        labels = [a.arm_id for a in arms]
        if len(set(labels)) != len(labels):
            for i, arm in enumerate(arms):
                arm.arm_id = f"{arm.arm_id}#{i + 1}"
    return arms


def _handle_rank(args: argparse.Namespace) -> int:
    """`openbench rank` (v0.5.0): rank stored arms with uncertainty; no provider calls."""
    from openbench.dataset import load_ids_file
    from openbench.ranking import rank_arms
    from openbench.reporting_v050 import render_ranking
    from openbench.security import redact_sensitive_data

    try:
        arms = _load_arms(args)
    except KeyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    selected = tuple(load_ids_file(args.ids_file).selected_ids) if args.ids_file else None
    payload = rank_arms(
        arms, args.metric, alpha=args.alpha, confidence_level=args.confidence_level, resamples=args.resamples,
        permutations=args.permutations, analysis_seed=args.seed, selected_ids=selected, allow_incompatible=args.exploratory,
    )
    if args.format == "html":
        from openbench.html_report import render_html

        text = render_html(redact_sensitive_data(payload))
    else:
        text = render_ranking(redact_sensitive_data(payload), args.format)
    _write_or_print(text, args.output, f"Ranking ({args.format})")
    if payload.get("banner"):
        print(f"Warning: {payload['banner']}", file=sys.stderr)
    return 0


def _pilot_inputs(args: argparse.Namespace) -> dict[str, Any]:
    from openbench.itemscores import collect_item_scores, metric_score_kind
    from openbench.stats._numeric import stdev

    manager = ExperimentManager(experiments_root=args.experiments_dir)
    ev_a = manager.load_evaluation(args.pilot_runs[0])
    ev_b = manager.load_evaluation(args.pilot_runs[1])
    obs_a = collect_item_scores([ev_a], args.metric)
    obs_b = collect_item_scores([ev_b], args.metric)
    pairs = [(obs_a[i].mean, obs_b[i].mean) for i in sorted(set(obs_a) & set(obs_b))
             if obs_a[i].mean is not None and obs_b[i].mean is not None]
    if len(pairs) < 2:
        raise OpenBenchError("The pilot runs share fewer than 2 scored items")
    kind = metric_score_kind(args.metric)
    out: dict[str, Any] = {"pilot_items": len(pairs), "pilot_runs": list(args.pilot_runs), "metric": args.metric, "score_kind": kind}
    if kind == "binary" and not args.continuous:
        n = len(pairs)
        out["p01"] = sum(1 for a, b in pairs if a == 0 and b == 1) / n
        out["p10"] = sum(1 for a, b in pairs if a == 1 and b == 0) / n
        out["baseline_accuracy"] = sum(a for a, _ in pairs) / n
    else:
        out["sd_difference"] = stdev([b - a for a, b in pairs])
    return out


def _handle_plan(args: argparse.Namespace) -> int:
    """`openbench plan` (v0.5.0): sample-size and power planning; no provider calls."""
    from openbench.reporting_v050 import render_plan
    from openbench.stats import power as pw

    pilot: dict[str, Any] = {}
    if args.pilot_runs:
        try:
            pilot = _pilot_inputs(args)
        except (KeyError, OpenBenchError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    continuous = args.continuous or (pilot and "sd_difference" in pilot)
    try:
        if args.items is not None:
            if continuous:
                sd = args.sd_difference or pilot.get("sd_difference")
                if not sd:
                    print("Error: --items with --continuous needs --sd-difference or --pilot-runs.", file=sys.stderr)
                    return 2
                payload = {"metric_kind": "continuous", "items": args.items, "sd_difference": sd, "alpha": args.alpha,
                           "target_power": args.power, "method": "normal_approximation",
                           "minimum_detectable_difference": pw.minimum_detectable_difference_continuous(args.items, sd_difference=sd, alpha=args.alpha, power=args.power)}
            else:
                base = args.baseline_accuracy if args.baseline_accuracy is not None else pilot.get("baseline_accuracy")
                if base is None:
                    print("Error: --items needs --baseline-accuracy (or --pilot-runs).", file=sys.stderr)
                    return 2
                payload = {"metric_kind": "binary", "test": "mcnemar_exact", "items": args.items, "baseline_accuracy": base,
                           "alpha": args.alpha, "target_power": args.power,
                           "assumption": "independent errors (usually overstates disagreement, so the detectable difference is cautious)",
                           "minimum_detectable_difference": pw.minimum_detectable_difference_binary(args.items, baseline_accuracy=base, alpha=args.alpha, power=args.power)}
        else:
            if args.difference is None:
                print("Error: give --difference (the smallest difference worth detecting) or --items.", file=sys.stderr)
                return 2
            if continuous:
                sd = args.sd_difference or pilot.get("sd_difference")
                if not sd:
                    print("Error: --continuous needs --sd-difference or --pilot-runs.", file=sys.stderr)
                    return 2
                payload = pw.plan_continuous(delta=args.difference, sd_difference=sd, alpha=args.alpha, power=args.power)
            elif "p01" in pilot:
                psi = pilot["p01"] + pilot["p10"]
                payload = pw.plan_binary(delta=args.difference, discordance=max(psi, abs(args.difference)), alpha=args.alpha, power=args.power)
                payload["assumption"] = "discordance rate estimated from the pilot runs"
            else:
                payload = pw.plan_binary(delta=args.difference, baseline_accuracy=args.baseline_accuracy,
                                         discordance=args.discordance, alpha=args.alpha, power=args.power)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    if pilot:
        payload["pilot"] = pilot
    payload["provider_calls"] = 0
    _write_or_print(render_plan(payload, args.format), args.output, f"Plan ({args.format})")
    return 0


def _handle_judge(args: argparse.Namespace) -> int:
    """`openbench judge` (v0.5.0): pairwise LLM-as-judge with bias controls."""
    from openbench import preferences as prefs
    from openbench.agreement import agreement_report
    from openbench.judge import DEFAULT_CRITERIA, PairwiseJudge
    from openbench.reporting_v050 import render_judge
    from openbench.security import redact_sensitive_data

    sources = []
    if args.runs:
        manager = ExperimentManager(experiments_root=args.experiments_dir)
        for rid in args.runs:
            try:
                rec = manager.load_run_record(rid)
                ev = manager.load_evaluation(rid)
            except KeyError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1
            sources.append(prefs.PreferenceSource.from_evaluation(rec.run_id, ev))
    else:
        from openbench.paired import ComparisonArm

        for path in args.results:
            ev = ComparisonArm.from_evaluation_file(path).evaluations[0]
            sources.append(prefs.PreferenceSource.from_evaluation(Path(path).stem, ev))
        if sources[0].source_id == sources[1].source_id:
            sources[0].source_id += "#1"
            sources[1].source_id += "#2"
    extra = {"api_key_env": args.api_key_env} if args.api_key_env else {}
    judge_adapter = _instantiate_provider_from_args_or_profile(
        provider_name=args.judge_provider, model_name=args.judge_model, model_version="latest",
        endpoint=args.judge_endpoint, gen_cfg=GenerationConfig(temperature=0.0, max_tokens=8),
        timeout_seconds=args.timeout, extra_prov_cfg=extra,
    )
    criteria = Path(args.criteria).read_text(encoding="utf-8") if args.criteria else DEFAULT_CRITERIA
    exec_cfg = ExecutionConfig(timeout_seconds=args.timeout, concurrency=args.concurrency, use_cache=not args.no_cache,
                               max_requests_per_minute=args.max_requests_per_minute)
    cache = ResponseCache(cache_dir=args.cache_dir, enabled=not args.no_cache)
    judge = PairwiseJudge(judge_adapter, criteria=criteria, show_reference=args.show_reference, execution_config=exec_cfg,
                          cache=cache, analysis_seed=args.seed, resamples=args.resamples)
    result = judge.judge_sources(sources[0], sources[1]).to_dict()
    if args.key or args.annotations:
        if not (args.key and args.annotations):
            print("Error: --key and --annotations must be given together.", file=sys.stderr)
            return 2
        key = prefs.load_key(args.key)
        ids = {key["models"]["A"]["source_id"], key["models"]["B"]["source_id"]}
        if ids != {sources[0].source_id, sources[1].source_id}:
            print("Error: the preference key was built from different runs than the ones judged.", file=sys.stderr)
            return 2
        swap = key["models"]["A"]["source_id"] != sources[0].source_id
        report = prefs.AnnotationImporter(key).load(args.annotations)
        if not report.valid:
            print("Error: annotations are invalid; run 'openbench preferences validate'.", file=sys.stderr)
            return 1
        tasks = {t["task_id"]: t for t in key["tasks"]}
        human: dict[str, list[str]] = {}
        for a in report.annotations:
            task = tasks[a.task_id]
            human.setdefault(task["record_id"], []).append(prefs.MODEL_SPACE[task["display_order"]].get(a.outcome, a.outcome))
        flip = {"A": "B", "B": "A", "tie": "tie", None: None}
        automatic = {t["record_id"]: (flip[t["verdict"]] if swap else t["verdict"]) for t in result["tasks"]}
        result["human_agreement"] = agreement_report(
            human, automatic, categories={t["record_id"]: t["category"] for t in result["tasks"]},
            automatic_name=f"judge:{args.judge_model}", analysis_seed=args.seed, resamples=args.resamples,
        )
    _write_or_print(render_judge(redact_sensitive_data(result), args.format), args.output, f"Judge report ({args.format})")
    for w in result["warnings"]:
        print(f"Warning [{w['code']}]: {w['message']}", file=sys.stderr)
    return 0


def _handle_import_dataset(args: argparse.Namespace) -> int:
    """`openbench import-dataset` (v0.5.0): convert a local benchmark file."""
    from openbench.importers import import_benchmark
    from openbench.security import validate_safe_output_path

    out = validate_safe_output_path(args.output)
    if out.exists() and not args.overwrite:
        print(f"Error: {out} exists; use --overwrite to replace it.", file=sys.stderr)
        return 2
    result = import_benchmark(
        args.input, args.import_format, category=args.category, id_prefix=args.id_prefix, source=args.source,
        license_name=args.license_name, difficulty=args.difficulty, question_field=args.question_field,
        answer_field=args.answer_field, id_field=args.id_field,
    )
    if not result.records:
        print(f"Error: no records could be converted ({len(result.skipped)} skipped).", file=sys.stderr)
        for s in result.skipped[:20]:
            print(f"  line {s['line']}: {s['reason']}", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.to_jsonl(), encoding="utf-8")
    dataset = DatasetLoader.load(out)  # validate with the normal loader
    print(f"Imported {len(dataset)} record(s) into {out} (checksum {dataset.checksum[:12]}).")
    if result.skipped:
        print(f"Warning: skipped {len(result.skipped)} record(s):", file=sys.stderr)
        for s in result.skipped[:20]:
            print(f"  line {s['line']}: {s['reason']}", file=sys.stderr)
    print("Check the benchmark's licence before sharing results or the converted file.")
    return 0


def _handle_audit_contamination(args: argparse.Namespace) -> int:
    """`openbench audit-contamination` (v0.5.0): read-only n-gram overlap check."""
    from openbench.contamination import audit_contamination, render_contamination

    dataset = DatasetLoader.load(args.dataset_path)
    payload = audit_contamination(dataset, args.corpus, ngram=args.ngram, threshold=args.threshold, min_words=args.min_words)
    _write_or_print(render_contamination(payload, args.format), args.output, f"Contamination audit ({args.format})")
    s = payload["summary"]
    print(f"{s['records_flagged']} of {len(payload['records'])} record(s) flagged (overlap >= {args.threshold}). "
          "A low overlap does not prove the dataset is clean.", file=sys.stderr)
    return 0


def _handle_dataset_card(args: argparse.Namespace) -> int:
    """`openbench dataset-card` (v0.5.0)."""
    from openbench.datacard import build_card, render_card

    card = build_card(args.dataset_path)
    _write_or_print(render_card(card, args.format), args.output, f"Dataset card ({args.format})")
    return 0


def _handle_serve(args: argparse.Namespace) -> int:
    """`openbench serve` (v0.5.0): read-only local dashboard."""
    from openbench.serve import create_server

    try:
        server = create_server(args.experiments_dir, args.reports_dir, args.host, args.port, args.allow_remote)
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    host, port = server.server_address[:2]
    bound_host = f"[{host}]" if ":" in host else host
    bound_url = f"http://{bound_host}:{port}/"
    browser_host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
    browser_host = f"[{browser_host}]" if ":" in browser_host else browser_host
    url = f"http://{browser_host}:{port}/"
    if args.open:
        threading.Thread(
            target=webbrowser.open,
            args=(url,),
            daemon=True,
        ).start()
    print(f"OpenBench dashboard (read-only) at {bound_url}  -  press Ctrl+C to stop.")
    if url != bound_url:
        print(f"Local dashboard URL: {url}")
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("Warning: listening on a non-loopback address; anyone who can reach it can read your results.", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _handle_metrics(args: argparse.Namespace) -> int:
    """Execute `openbench metrics list` (v0.5.0)."""
    from openbench.metrics import METRIC_REGISTRY, list_registered_metrics
    from openbench.plugins import plugin_source

    if getattr(args, "metrics_action", None) != "list":
        print("Error: Specify a metrics subcommand ('list').", file=sys.stderr)
        return 2
    metas = list_registered_metrics()
    rows = []
    for meta in metas:
        cls = METRIC_REGISTRY.get(meta.name)
        source = plugin_source(cls) if cls is not None else None
        rows.append({**meta.to_dict(), "source": source or "built-in"})
    if args.format == "json":
        print(json.dumps(rows, indent=2))
        return 0
    print(f"Registered Metrics ({len(rows)}):")
    for row in rows:
        src = row["source"]
        tag = f" [plugin: {src.get('distribution') or src.get('entry_point')}]" if isinstance(src, dict) else ""
        print(f"  - {row['name']}{tag}: {row.get('description', '')}")
    return 0


def _handle_plugins(args: argparse.Namespace) -> int:
    """Execute `openbench plugins list` (v0.5.0)."""
    from openbench.plugins import load_plugins, plugins_disabled

    if getattr(args, "plugins_action", None) != "list":
        print("Error: Specify a plugins subcommand ('list').", file=sys.stderr)
        return 2
    infos = [i.to_dict() for i in load_plugins()]
    if args.format == "json":
        print(json.dumps({"disabled": plugins_disabled(), "plugins": infos}, indent=2))
        return 0
    if plugins_disabled():
        print("Plugin discovery is disabled (OPENBENCH_DISABLE_PLUGINS is set).")
        return 0
    if not infos:
        print("No OpenBench plugins installed.")
        return 0
    print(f"Installed plugins ({len(infos)}):")
    for i in infos:
        pkg = f" from {i['distribution']} {i['version']}" if i["distribution"] else ""
        print(f"  - {i['kind']}: {i['name']} -> {i['entry_point']}{pkg}")
    return 0


def _handle_cache(args: argparse.Namespace) -> int:
    """Execute `openbench cache clear|stats`."""
    if not args.cache_action:
        print("Error: Specify a cache subcommand ('clear' or 'stats').", file=sys.stderr)
        return 2

    cache = ResponseCache(cache_dir=getattr(args, "cache_dir", None), enabled=True)
    if args.cache_action == "clear":
        cleared = cache.clear()
        print(f"Cleared {cleared} cached response(s) from {cache.cache_dir}")
        return 0
    if args.cache_action == "stats":
        print(json.dumps(cache.stats(), indent=2))
        return 0
    return 2


def _is_closed_pipe_error(exc: OSError) -> bool:
    """True when `exc` means stdout's reader went away (EPIPE, or EINVAL on Windows)."""
    import errno

    if isinstance(exc, BrokenPipeError):
        return True
    # Windows reports writes to a closed pipe as EINVAL (pip handles it the same way).
    return sys.platform == "win32" and exc.errno == errno.EINVAL


def _exit_quietly_after_closed_stdout() -> int:
    """Point stdout at the null device so the interpreter's final flush cannot fail, then exit 1."""
    import os

    try:
        fileno = sys.stdout.fileno()
    except (AttributeError, OSError, ValueError):
        # stdout has no real file descriptor (e.g. replaced by a StringIO); nothing to redirect.
        return 1
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, fileno)
    finally:
        # dup2 duplicated the descriptor onto stdout; the original is no longer needed (#6).
        os.close(devnull)
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point for the `openbench` CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 2

    try:
        if args.command == "validate-dataset":
            if getattr(args, "quality_checks", False):
                return _handle_dataset_audit(args)
            return _handle_validate_dataset(args.dataset_path)
        if args.command == "evaluate":
            return _handle_evaluate(args)
        if args.command == "compare":
            return _handle_compare(
                results_args=args.results,
                output_arg=args.output,
            )
        if args.command == "list-datasets":
            return _handle_list_datasets(args)
        if args.command == "inspect-dataset":
            return _handle_inspect_dataset(args)
        if args.command == "dataset-stats":
            return _handle_dataset_stats(args)
        if args.command == "history":
            return _handle_history(args)
        if args.command == "export":
            return _handle_export(args)
        if args.command == "run":
            return _handle_run(args)
        if args.command == "runs":
            return _handle_runs(args)
        if args.command == "providers":
            return _handle_providers(args)
        if args.command == "cache":
            return _handle_cache(args)
        if args.command == "metrics":
            return _handle_metrics(args)
        if args.command == "rank":
            return _handle_rank(args)
        if args.command == "plan":
            return _handle_plan(args)
        if args.command == "judge":
            return _handle_judge(args)
        if args.command == "import-dataset":
            return _handle_import_dataset(args)
        if args.command == "audit-contamination":
            return _handle_audit_contamination(args)
        if args.command == "dataset-card":
            return _handle_dataset_card(args)
        if args.command == "serve":
            return _handle_serve(args)
        if args.command == "plugins":
            return _handle_plugins(args)
        if args.command == "analyze":
            return _handle_analyze(args)
        if args.command == "preferences":
            return _handle_preferences(args)
    except BrokenPipeError:
        # v0.4.0: output piped into a command that closed early (e.g. `| head`); exit quietly.
        return _exit_quietly_after_closed_stdout()
    except FileNotFoundError as exc:
        print(f"Error: {sanitize_error_message(str(exc))}", file=sys.stderr)
        return 1
    except OpenBenchError as exc:
        print(f"Validation/Evaluation Error: {sanitize_error_message(str(exc))}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover
        if isinstance(exc, OSError) and _is_closed_pipe_error(exc):
            # Windows reports a closed stdout reader as OSError(EINVAL), not BrokenPipeError.
            return _exit_quietly_after_closed_stdout()
        # v0.4.0: messages are scrubbed of embedded credentials before printing.
        print(f"Unexpected Error: {sanitize_error_message(str(exc))}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
