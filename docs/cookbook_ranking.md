# Rank three models and plan a sample size

This walkthrough uses OpenBench AI 0.5.0 or newer and its offline mock
provider. It needs no GPU or network access. The mock provider returns the
dataset's expected answer for every model, so these runs demonstrate the
commands and report format; they do not compare real model quality.

## Save a small dataset

In a fresh working directory, save these six synthetic arithmetic questions
as `benchmark.jsonl`:

```jsonl title="benchmark.jsonl"
{"id":"q1","category":"arithmetic","prompt":"What is 2 + 2? Reply with a number.","expected_answer":"4"}
{"id":"q2","category":"arithmetic","prompt":"What is 3 + 5? Reply with a number.","expected_answer":"8"}
{"id":"q3","category":"arithmetic","prompt":"What is 10 - 6? Reply with a number.","expected_answer":"4"}
{"id":"q4","category":"arithmetic","prompt":"What is 3 × 4? Reply with a number.","expected_answer":"12"}
{"id":"q5","category":"arithmetic","prompt":"What is 18 ÷ 3? Reply with a number.","expected_answer":"6"}
{"id":"q6","category":"arithmetic","prompt":"What is 7 + 2? Reply with a number.","expected_answer":"9"}
```

## Run three mock models

These commands store the experiment records under `experiments/`. They save
each generated run ID so the ranking commands can use it. The mock provider
uses the expected answer for all three model names:

```bash
cedar_output=$(openbench run --dataset benchmark.jsonl --provider mock --model cedar --metrics exact_match --experiments-dir experiments --no-cache)
printf '%s\n' "$cedar_output"
cedar_id=$(printf '%s\n' "$cedar_output" | sed -n "s/^Run '\([^']*\)'.*/\1/p")

maple_output=$(openbench run --dataset benchmark.jsonl --provider mock --model maple --metrics exact_match --experiments-dir experiments --no-cache)
printf '%s\n' "$maple_output"
maple_id=$(printf '%s\n' "$maple_output" | sed -n "s/^Run '\([^']*\)'.*/\1/p")

birch_output=$(openbench run --dataset benchmark.jsonl --provider mock --model birch --metrics exact_match --experiments-dir experiments --no-cache)
printf '%s\n' "$birch_output"
birch_id=$(printf '%s\n' "$birch_output" | sed -n "s/^Run '\([^']*\)'.*/\1/p")
```

Observed run summaries (the generated IDs vary):

```text
Run 'run_e1e81697bfed' [completed]: 6/6 completed (provider=mock, model=cedar, scores={'exact_match': 1.0}) -> experiments/run_e1e81697bfed
Run 'run_379c411ea7e2' [completed]: 6/6 completed (provider=mock, model=maple, scores={'exact_match': 1.0}) -> experiments/run_379c411ea7e2
Run 'run_25c0408aaef3' [completed]: 6/6 completed (provider=mock, model=birch, scores={'exact_match': 1.0}) -> experiments/run_25c0408aaef3
```

## Rank in Markdown and HTML

Pass the format explicitly so each command is clear regardless of output
filename:

```bash
openbench rank --runs "$cedar_id" "$maple_id" "$birch_id" --metric exact_match --format markdown --output ranking.md
openbench rank --runs "$cedar_id" "$maple_id" "$birch_id" --metric exact_match --format html --output ranking.html
```

The Markdown command reported `Ranking (markdown) written to: ranking.md`;
the HTML command reported `Ranking (html) written to: ranking.html`. The
observed Markdown table was:

| Rank | Rank interval | Model | Mean | 95% CI | Tied with top | p(best), bootstrap |
|---|---|---|---|---|---|---|
| 2.0 | 1-3 | run_25c0408aaef3 | 1.0000 | [1.0000, 1.0000] | yes | 0.333 |
| 2.0 | 1-3 | run_379c411ea7e2 | 1.0000 | [1.0000, 1.0000] | yes | 0.333 |
| 2.0 | 1-3 | run_e1e81697bfed | 1.0000 | [1.0000, 1.0000] | yes | 0.333 |

In that captured report, the first ID is the birch run, the second is maple,
and the third is cedar; generated IDs vary. Each `1-3`
rank interval means the data do not rule out any of the three positions.
“Tied with top” means the paired tests did not distinguish that run from
the best observed run; it does not establish that the models are equal. In
this deliberately uniform mock example all pairwise Holm-adjusted p-values
were `1`. On real data, rank intervals and indistinguishable groups express
uncertainty: a single sorted leaderboard can imply a precision that the
items do not support. Here, the identical perfect mock answers also mean
the displayed confidence intervals are not evidence about real model
performance. See the [glossary](glossary.md#rank-interval) and
[statistical methodology](statistical_methodology.md#18-ranking-several-models-openbench-rank-v050)
for definitions and assumptions.

## Plan for a close pair

For a paired binary metric, suppose a pilot suggests the models disagree on
10% of items, and a five percentage point improvement is worth detecting.
Request 80% power at the default 5% significance level:

```bash
openbench plan --discordance 0.1 --difference 0.05 --power 0.8 --format markdown
```

Observed output:

```text
# OpenBench AI sample-size plan

**337 paired items** are needed to detect a difference of 0.0500 with 0.80 power at alpha = 0.05.

| Field | Value |
|---|---|
| metric_kind | binary |
| test | mcnemar_exact |
| difference | 0.0500 |
| p01 | 0.0750 |
| p10 | 0.0250 |
| discordance | 0.1000 |
| assumption | discordance rate given |
| alpha | 0.0500 |
| target_power | 0.8000 |
| items_needed | 337 |
| power_at_items_needed | 0.8007 |
| method | exact |
| provider_calls | 0 |

Assumption: discordance rate given.

Plans are approximate: they assume the items you will use behave like the inputs given here.

[Reading guide](https://lakshaychhaba0001-sys.github.io/openbench-ai/reading_a_report/) · [Glossary](https://lakshaychhaba0001-sys.github.io/openbench-ai/glossary/)
```

The planner uses the discordance assumption to calculate the paired sample
size; it does not call a provider. Estimate discordance from a representative
pilot when possible, and choose the smallest practically important
difference before looking at results. See [sample-size planning](statistical_methodology.md#19-sample-size-planning-openbench-plan-v050)
and the glossary entry on [discordance](glossary.md#discordance).

## Switch to real models

To replace the offline mock provider with local or hosted models, follow
[Switch to real models](cookbook_finetune.md#switch-to-real-models) in the
fine-tuning cookbook. Use the same dataset and metric for all model runs.
