# Cookbook: Did my fine-tune actually help?

A common question after fine-tuning is whether the new model actually improved over its base model, or whether an apparent gain is just sampling noise, too small a test set, or training-data overlap.

This walkthrough runs the full OpenBench workflow end to end using the offline `mock` provider (no GPU or API key required), then shows how to swap in a real local or hosted provider.

## 1. Plan the evaluation size (`openbench plan`)

Before running models, check how many evaluation items you need to detect a meaningful difference. Neither `plan` nor `analyze` calls a model.

Suppose your base model scores around 70% accuracy and you want 80% power (at `alpha = 0.05`) to detect a 10-point improvement:

```bash
openbench plan --baseline-accuracy 0.7 --difference 0.1
```

This reports **313 paired items** (assuming independent errors, which is cautious). If you already have a fixed benchmark—such as OpenBench's 200-item built-in suite under `datasets/`—you can run `plan` in reverse to see what difference 200 items can reliably detect:

```bash
openbench plan --items 200 --baseline-accuracy 0.7
```

With 200 items at 70% baseline accuracy, the minimum detectable difference at 80% power is **0.1237** (~12.4 percentage points). If you have already run two small pilot evaluations, pass `--pilot-runs <base_run> <tuned_run> --metric exact_match --difference 0.05` to estimate disagreement from real data instead of the independence assumption. See [Ranking and planning](ranking_and_planning.md) for details.

## 2. Run the base and fine-tuned models (`openbench run`)

Run both models on the **same dataset** so items align by record ID. Run at least one arm (ideally both, if budget allows) with `--repetitions` and `--seed` so OpenBench can separate run-to-run generation noise from item-to-item variation:

```bash
# Base model (single run)
openbench run --dataset datasets/ --provider mock --model base-mock \
  --experiment-id finetune-base

# Fine-tuned model (3 repetitions with base seed 100)
openbench run --dataset datasets/ --provider mock --model finetuned-mock \
  --repetitions 3 --seed 100 --experiment-id finetune-tuned
```

Note the IDs printed at the end of each command:

- The single run prints a run ID like `run_5b8c9869dccf`.
- The repeated run prints a repeat group ID like `grp_ff810888d3c0` (where repetition *i* uses seed `100 + i` and an isolated cache key).

> **Note on the `mock` provider:** `mock` echoes each item's `expected_answer`, so both arms score identically (`1.0` on `exact_match` and `accuracy`). Use it to verify your pipeline before spending compute; never treat mock scores as model quality.

### Swapping in real providers

Replace `--provider mock` and the mock model names with your real provider while keeping `--dataset`, `--repetitions`, `--seed`, and `--metrics` the same:

```bash
# Local Ollama models
openbench run --dataset datasets/ --provider ollama --model llama3.2 \
  --experiment-id finetune-base
openbench run --dataset datasets/ --provider ollama --model llama3.2-finetuned \
  --repetitions 3 --seed 100 --experiment-id finetune-tuned

# OpenAI-compatible server (vLLM, LM Studio, llama.cpp server, or hosted API)
openbench run --dataset datasets/ --provider openai_compatible \
  --endpoint http://localhost:8000/v1 --model my-base-model \
  --experiment-id finetune-base
openbench run --dataset datasets/ --provider openai_compatible \
  --endpoint http://localhost:8000/v1 --model my-finetuned-model \
  --repetitions 3 --seed 100 --experiment-id finetune-tuned
```

Check provider readiness beforehand with `openbench providers check ollama --model llama3.2` or `openbench providers check openai_compatible --endpoint http://localhost:8000/v1 --model my-base-model`. See [Providers](providers.md) and [Repeated runs](repeated_runs.md) for more options.

## 3. Compare the runs (`openbench analyze`)

Pass the base run ID and fine-tuned group ID (substitute the IDs printed in step 2) to `openbench analyze`:

```bash
openbench analyze --runs <base_run_id> <tuned_group_id> \
  --baseline base-mock --format markdown --output comparison.md
```

### Reading the result

Open `comparison.md` and check these sections in order:

1. **Direction (`Difference = challenger - baseline`):** A positive difference favours the fine-tuned model (`finetuned-mock`); a negative difference favours `base-mock`.
2. **Sample size and test:** For a repeat group, OpenBench averages each item's score across its valid repetitions first, so the unit of analysis is the **200 aligned items** (not `200 × 3`). Because one arm has repetitions, OpenBench uses the paired `sign_flip` permutation test and applies Holm multiplicity correction across metrics (`Adjusted p`).
3. **Repeated-run variability:** The `Repeated-run variability` section reports within-item variance, flip rate, and run-to-run range separately from the question-level 95% confidence interval.
4. **Compatibility and warnings:** Check for `mock_provider`, `repetition_counts_differ`, or `generation_configuration_differs` warnings before drawing conclusions.

### What "not significant" does and doesn't mean

With the mock runs, `comparison.md` reports `Difference = 0.0000` and `Adjusted p = 1.0000` (`No statistically significant difference was detected`). On real models, a non-significant result (`Adjusted p >= 0.05`) is common when a fine-tune brings modest gains or when the test set is small:

- **What it does mean:** Under this test and sample size, the data do not provide enough evidence to rule out zero difference at `alpha = 0.05`.
- **What it does *not* mean:** It does **not** prove the fine-tune had no effect or that the two models are equivalent. Look at the **95% CI**—if the interval is `[-0.02, +0.10]`, the data are compatible with both a slight regression and a 10-point improvement.
- **Equivalence vs. practical importance:** To formally test equivalence, you must supply a pre-specified margin with `--equivalence-margin` (TOST). To classify whether an observed difference is large enough to matter in practice, supply `--min-practical-difference`. See [Reading a comparison report](reading_a_report.md).

## 4. Check per-category breakdowns and contamination risk

Overall accuracy can hide regressions—for example, a coding fine-tune might improve `coding` and `debugging` while hurting `safety` or `instruction_following`.

### Per-category breakdowns

Under each metric in `comparison.md`, OpenBench automatically prints a **By category (descriptive)** table (and **By difficulty (descriptive)**) showing per-group `n`, means, differences, and CIs:

```text
By category (descriptive):

| Group | n | Baseline mean | Challenger mean | Difference | CI | p | Adjusted p |
|---|---|---|---|---|---|---|---|
| coding | 20 | 1.0000 | 1.0000 | 0.0000 | [0.0000, 0.0000] | n/a | n/a |
...
```

By default, category rows are descriptive (`p` is `n/a`). If you want formal per-category hypothesis tests (with multiplicity adjustment), re-run `analyze` with `--stratum-tests`:

```bash
openbench analyze --runs <base_run_id> <tuned_group_id> \
  --baseline base-mock --stratum-tests --format markdown --output comparison_strata.md
```

### Audit for fine-tuning data contamination (`openbench audit-contamination`)

Before celebrating a real gain, verify that your evaluation questions did not leak into your fine-tuning corpus. `audit-contamination` is read-only and compares word n-grams (13-grams by default) between your evaluation dataset and any `.txt`, `.md`, `.json`, `.jsonl`, or `.csv` file (or directory of files):

```bash
# Create a tiny example fine-tuning file (or point --corpus at your real training data)
printf '{"prompt": "Explain binary search.", "completion": "Halve a sorted array each step."}\n' > finetune_train.jsonl

openbench audit-contamination datasets/ --corpus finetune_train.jsonl \
  --format markdown --output contamination.md
```

Records with prompt or reference-answer n-gram overlap at or above `--threshold` (default `0.5`, skipping fields shorter than `--min-words 8`) are flagged for review.

Keep the caveat in mind: **a low overlap does not prove the evaluation set is clean**—paraphrases, translations, and copies shorter than `--ngram` (13 words) are not detected—and a high overlap is a risk signal to inspect manually rather than proof of memorization. See [Benchmark tools](benchmark_tools.md) for all options.
