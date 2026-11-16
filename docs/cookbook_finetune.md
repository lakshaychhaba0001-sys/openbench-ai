# Compare a fine-tuned model with its base model

Use OpenBench AI 0.5.0 or newer and a fresh working directory. This
cookbook uses synthetic questions and the offline mock provider. The model
names label the two arms of the comparison. Both mocks answer from the
reference data, so their scores demonstrate the workflow.

## Plan the number of evaluation items

Choose the smallest gain worth detecting before seeing the scores. For
example, plan for a five percentage point gain in exact match and 80%
power. Suppose only one model answers correctly on 20% of pilot items:

```bash
openbench plan --discordance 0.2 --difference 0.05 --power 0.8 --alpha 0.05
```

Compare the estimated required item count with the size of your held-out
dataset. Refresh the discordance estimate using a representative pilot
under the same settings. The tiny dataset below verifies the commands
only. Repeating generation on the same items does not increase the number
of independent evaluation items.

## Run both arms on the same data

Save this synthetic holdout as `holdout.jsonl`:

```jsonl title="holdout.jsonl"
{"id":"math-1","category":"mathematics","prompt":"A trail is six kilometres long and you have already walked two kilometres. How many kilometres remain? Reply with a number.","expected_answer":"4"}
{"id":"math-2","category":"mathematics","prompt":"Split twelve marbles equally among three children. How many does each child receive? Reply with a number.","expected_answer":"4"}
{"id":"math-3","category":"mathematics","prompt":"A box holds five pencils. How many pencils fit in two such boxes? Reply with a number.","expected_answer":"10"}
{"id":"math-4","category":"mathematics","prompt":"Four buses each carry seven riders. What is the total number of riders? Reply with a number.","expected_answer":"28"}
{"id":"facts-1","category":"factual_knowledge","prompt":"What is the chemical symbol for oxygen? Reply with the symbol only.","expected_answer":"O"}
{"id":"facts-2","category":"factual_knowledge","prompt":"What planet has the well-known Great Red Spot? Reply with the name only.","expected_answer":"Jupiter"}
{"id":"facts-3","category":"factual_knowledge","prompt":"Which gas is most abundant in Earth's atmosphere? Reply with its name only.","expected_answer":"Nitrogen"}
{"id":"facts-4","category":"factual_knowledge","prompt":"What is the standard SI unit of electric current? Reply with its name only.","expected_answer":"ampere"}
```

In a Bash-compatible shell, run the base model once and the fine-tuned arm
three times. Keep the holdout, scoring, prompts, and comparable generation
settings fixed:

```bash
openbench run --dataset holdout.jsonl --provider mock --model base --metrics exact_match --no-cache > base-run.txt
openbench run --dataset holdout.jsonl --provider mock --model fine-tuned --metrics exact_match --no-cache --repetitions 3 --seed 100 > fine-tuned-run.txt
```

The fine-tuned arm records a repeat group and seeds 100 through 102 in its
lineage. The mock adapter does not forward generation seeds. When you swap
in the [provider](providers.md) that serves your models, set the model IDs
and versions to the actual checkpoints and configure its endpoint or key
where required. Check the seed status in the resulting run.

## Switch to real models

Before running an evaluation against a local or hosted server, verify
connectivity and model availability with `openbench providers check`:

```bash
openbench providers check ollama --endpoint http://localhost:11434 --model base-model
openbench providers check openai_compatible --endpoint http://localhost:8000/v1 --model fine-tuned-model
```

Once the provider responds, run the same `openbench run` commands with
`--provider ollama` or `--provider openai_compatible` and your model
identifiers:

```bash
# Local Ollama server
openbench run --dataset holdout.jsonl --provider ollama --endpoint http://localhost:11434 --model base-model --metrics exact_match --no-cache > base-run.txt
openbench run --dataset holdout.jsonl --provider ollama --endpoint http://localhost:11434 --model fine-tuned-model --metrics exact_match --no-cache --repetitions 3 --seed 100 > fine-tuned-run.txt

# OpenAI-compatible endpoint (vLLM, llama.cpp server, LM Studio, or hosted API)
openbench run --dataset holdout.jsonl --provider openai_compatible --endpoint http://localhost:8000/v1 --model base-model --metrics exact_match --no-cache > base-run.txt
openbench run --dataset holdout.jsonl --provider openai_compatible --endpoint http://localhost:8000/v1 --model fine-tuned-model --metrics exact_match --no-cache --repetitions 3 --seed 100 > fine-tuned-run.txt
```

See [Model Providers](providers.md) for authentication (`--api-key-env`),
rate-limiting (`--max-requests-per-minute`), and seed-forwarding details.

## Compare the arms and read the result

Extract the run ID and repeat-group ID printed by the two commands:

```bash
base_id=$(awk -F "'" '/^Run / { print $2; exit }' base-run.txt)
fine_tuned_group=$(awk -F "'" '/^Repeat group / { print $2; exit }' fine-tuned-run.txt)
openbench analyze --runs "$base_id" "$fine_tuned_group" --baseline "$base_id" --metrics exact_match --seed 41 --format markdown --output comparison.md
```

Open `comparison.md`. There are eight paired items. The challenger’s
item score averages its valid repetitions. Read the estimated difference
(fine-tuned minus base), its confidence interval, and the adjusted p-value
together. The two mock arms tie at 100% exact match with an adjusted
p-value of 1. A non-significant result leaves the
no-difference hypothesis unrejected at the chosen threshold. Effects within
the reported interval remain compatible with the data. An equivalence claim
requires a prespecified practical margin and an appropriate equivalence
test.

Find **By category (descriptive)** under `exact_match`. Both categories
have four items. On a real evaluation, look for opposing changes or small
groups before interpreting the overall result. These default breakdowns
are descriptive. See [Reading a comparison report](reading_a_report.md)
for the assumptions and warnings.

## Formal per-category tests

The default **By category (descriptive)** tables in `comparison.md` are
descriptive summaries only and do not run hypothesis tests per category.
To run formal, multiplicity-adjusted paired tests within each category,
pass `--stratum-tests` to `openbench analyze`:

```bash
openbench analyze --runs "$base_id" "$fine_tuned_group" --baseline "$base_id" --metrics exact_match --seed 41 --stratum-tests --format markdown --output comparison-strata.md
```

Testing many small categories reduces statistical power within each group
and increases the multiple-testing adjustment across strata, so interpret
per-category p-values alongside group sample sizes and confidence intervals.
See [Statistical methodology](statistical_methodology.md) for details on
stratified resampling and familywise error control.

## Audit the holdout against the fine-tuning corpus

Save this line as `fine-tuning.txt`. The overlap is deliberate so the
audit has a positive example:

```text title="fine-tuning.txt"
A trail is six kilometres long and you have already walked two kilometres. How many kilometres remain? Reply with a number.
```

```bash
openbench audit-contamination holdout.jsonl --corpus fine-tuning.txt --format markdown --output overlap.md
```

Read `overlap.md` and look for the flagged prompt on `math-1`. For a real
run, audit against the local fine-tuning corpus before scoring the holdout.
Word n-gram overlap is a review signal. Low overlap can miss paraphrases,
translations, short matches, or data outside the supplied corpus. See
[Benchmark tools](benchmark_tools.md#contamination-risk-openbench-audit-contamination)
for what the audit measures.
