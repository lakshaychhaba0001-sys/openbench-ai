# Benchmarks you can trust: import, contamination checks, cards and near-duplicates

Four read-mostly tools for getting a benchmark into OpenBench and checking it before you trust a
score. None of them calls a model or changes your dataset.

## Import a benchmark: `openbench import-dataset`

Converts a local file into an OpenBench JSONL dataset. Nothing is downloaded.

| `--format` | Input |
| --- | --- |
| `qa-jsonl`, `qa-csv` | Question/answer pairs. Field names default to `question` and `answer`; change them with `--question-field` and `--answer-field`. |
| `mcq-jsonl` | Multiple choice: `question`, `choices` (a list, or an object such as `{"A": ..., "B": ...}`), `answer` (a letter, a 0-based index, or the text of the right choice). |
| `mcq-csv` | Multiple choice in the common `question, A, B, C, D, answer` column layout (more lettered columns are fine). |

```bash
openbench import-dataset mmlu_math.csv --format mcq-csv --category math \
  --source "MMLU (local copy)" --license MIT --output datasets/mmlu_math.jsonl
```

- Multiple-choice prompts list the options as `A. ...` lines and ask for the letter only, and the
  reference answer is the letter, so `exact_match` scores them fairly.
- Each record keeps its original ID (`metadata.source_id`), position, file name and the
  `--source` and `--license` you give.
- Rows that cannot be converted are skipped and listed with the line number and reason. The
  output is validated with the normal dataset loader.
- **Check the benchmark's licence** before you share the converted file or results.

## Contamination risk: `openbench audit-contamination`

If test questions appeared in a model's training data, its score is inflated. OpenBench cannot
see training data, but it can check a dataset against text you supply: a public benchmark, a
forum dump, or your own fine-tuning data.

```bash
openbench audit-contamination datasets/reasoning/sample.jsonl --corpus my_finetune_data/ --format markdown
```

- Text is normalised (case, punctuation, whitespace) and split into words. The check uses word
  n-grams, 13 by default (`--ngram`), the length common in contamination studies.
- For each record it reports the share of the prompt's and reference answer's n-grams found in
  the corpus, and the longest shared run of words. Records at or above `--threshold` (default
  0.5) are flagged. Fields shorter than `--min-words` (default 8) are skipped and counted.
- The corpus can be one file or a directory of `.txt`, `.md`, `.json`, `.jsonl` and `.csv` files
  (for JSON files, every string value is read).

**A low overlap does not prove a dataset is clean.** Paraphrases, translations, copies shorter
than the n-gram length and anything outside your corpus are not detected. A high overlap is a
risk to review, not proof that a model saw the data.

## Dataset cards: `openbench dataset-card`

```bash
openbench dataset-card datasets/reasoning/sample.jsonl --output datasets/reasoning/CARD.md
```

The card fills in what OpenBench can compute (version, record count, checksum, category,
difficulty and language mix, prompt and answer lengths, quality-check results). The parts only
the author knows (purpose, source and licence, how answers were checked, known limitations) are
left as visible **TODO (author)** placeholders; OpenBench never invents them. The card has no
timestamps, so the same data always gives the same card. Use `--format json` for machine-readable
output or `--format html --output card.html` for a standalone page with highlighted author notes.

## Semantic near-duplicates

`openbench validate-dataset --quality-checks` finds near-duplicate prompts with a lexical
check (DQ003, character 5-gram similarity). It misses paraphrases. With the optional `semantic`
extra, an embedding pass finds them too:

```bash
pip install "openbench-ai[semantic]"
openbench validate-dataset datasets/ --quality-checks --near-duplicate-method semantic \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 --semantic-threshold 0.9
```

The lexical check still runs. Each DQ003 finding says which method produced it, and semantic
findings include the cosine similarity and the embedding model. The model must already be
available locally; nothing is downloaded unless your sentence-transformers setup does so. See
[quality_audits.md](quality_audits.md).
