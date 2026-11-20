# LLM-as-judge with bias controls

`openbench judge` asks a third model which of two responses is better, for every item two
runs share. It is built to be **measured, not trusted**: it reports how consistent and how
biased the judge is, and how well it agrees with people when you have human annotations.

```bash
openbench judge --runs run_<a> run_<b> \
  --judge-provider ollama --judge-model qwen2.5:7b --output judge.md

# Any OpenAI-compatible judge, with agreement against blinded human annotations
openbench judge --runs run_<a> run_<b> \
  --judge-provider openai_compatible --judge-endpoint http://localhost:8000/v1 --judge-model judge \
  --key prefs/tasks.key.json --annotations annotations.jsonl
```

Unless you pass `--format`, the judge writes JSON for `.json`, CSV for `.csv`, and Markdown for
other filenames or terminal output.

## What it does

1. **Both orders.** Each pair is judged twice: A shown first, then B shown first. A verdict
   counts only when both orders agree. If they disagree, the pair is **inconsistent** and
   scored as a tie.
2. **Verdicts.** The judge must answer `1`, `2` or `tie` on the first line. Anything else is
   counted as unparsed.
3. **Caching and limits.** Judge calls go through the same engine as `openbench run`, so they
   are cached (`--cache-dir`), retried, and can be rate-limited (`--max-requests-per-minute`).

## What it reports

| Measure | Meaning |
| --- | --- |
| Win rate of A | A = 1, tie = 0.5, B = 0, averaged over pairs, with a bootstrap CI. |
| Position consistency | Share of pairs with the same verdict in both orders. Low values mean the verdicts depend on order more than on content. |
| Position bias | Share of decisive single-order verdicts that pick the first-shown response. 0.5 is unbiased; flagged when the CI excludes 0.5. |
| Length bias | Among consistent decisive verdicts, the share won by the longer response. Flagged when the CI is above 0.5. |
| Self-preference | Flagged when the judge model has the same name as a judged model. |
| Agreement with people | With `--key` and `--annotations`: percent agreement and Cohen's kappa against the human majority per item. |

The judge's identity, prompt version (`pairwise-v1`) and prompt SHA-256 are recorded. Use
`--criteria FILE` to replace the judging criteria and `--show-reference` to show the reference
answer to the judge.

## Limits

Judge verdicts are another model's opinion. They can share the judged models' blind spots and
are not ground truth. Check agreement with people on a sample before relying on a judge. See
[statistical_methodology.md](statistical_methodology.md), section 21.

## Checking a metric against people

The same agreement statistics work for any metric. With a preference key and annotations:

```bash
openbench preferences analyze --key prefs/tasks.key.json --annotations annotations.jsonl \
  --compare-metric exact_match --experiments-dir experiments
```

This reports how often the metric (the response with the higher score) picks the same
winner as the human majority. See [human_preferences.md](human_preferences.md).
