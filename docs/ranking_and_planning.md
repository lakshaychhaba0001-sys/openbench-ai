# Ranking several models and planning sample sizes

Two commands answer the questions that come before and after a comparison:

- **`openbench rank`**: which of these models should we ship, and how sure are we of the order?
- **`openbench plan`**: how many questions do we need before a difference means anything?

Neither command calls a model. Both work on stored runs, [repeat groups](glossary.md#repeat-group) or evaluation files.

## `openbench rank`

```bash
openbench rank --runs run_<a> run_<b> grp_<c> run_<d> --metric exact_match --format markdown
openbench rank --results a.json b.json c.json --metric f1 --output ranking.json --format json
```

Output, one row per model:

| Column | Meaning |
| --- | --- |
| Rank | Rank of the observed mean (1 = best). Equal means share the average rank. |
| [Rank interval](glossary.md#rank-interval) | The ranks the data support. `1-1` means clearly first; `1-3` means it could be anywhere from first to third. All intervals hold together with about 95% confidence. |
| Mean, 95% CI | The model's score with a percentile bootstrap [confidence interval](glossary.md#confidence-interval) over items. |
| Tied with top | Not significantly different from the best observed model. |
| p(best), bootstrap | Share of resampled item sets in which the model scored highest. It describes how stable the top spot is, not the probability that the model is truly best. |

The report also lists every pairwise test with its [Holm-adjusted](glossary.md#holm-correction) p-value.

How it works: every pair of models is compared with the [paired test](glossary.md#paired-comparison) `openbench analyze`
would use, and all pairs are corrected together with Holm's method. A model's best possible
rank is 1 plus the number of models significantly better than it; its worst is the number of
models minus the number significantly worse. See [statistical_methodology.md](statistical_methodology.md), section 18.

Rules that also apply to `analyze`:
- Items are aligned by record ID; only items scored by every model are used.
- Inputs that fail hard compatibility checks (different record content, metric version or
  normalization) are refused. `--exploratory` ranks them anyway, labelled [descriptive only](glossary.md#descriptive-only).
- With few items, intervals are wide and several models are "tied with top". That means the data cannot tell them apart, not that they are equal.

## `openbench plan`

```bash
# How many items to detect a 5-point accuracy gain over a 70% baseline?
openbench plan --baseline-accuracy 0.7 --difference 0.05

# If the two models disagree on about 20% of items:
openbench plan --discordance 0.2 --difference 0.1 --power 0.9

# Estimate the inputs from two pilot runs you already have:
openbench plan --pilot-runs run_<a> run_<b> --metric exact_match --difference 0.05

# Reverse: what can 200 items detect?
openbench plan --items 200 --baseline-accuracy 0.7

# Continuous metrics: give (or estimate) the SD of per-item differences
openbench plan --continuous --sd-difference 0.3 --difference 0.05
```

What matters for binary metrics is not only the size of the difference but **how often the
two models disagree**. Two models at 70% and 75% that disagree on 10% of items need far fewer
questions than two that disagree on 40%. If you do not know the [discordance rate](glossary.md#discordance), the planner
assumes the models' errors are independent. That usually overstates disagreement, so the
answer is cautious. Pilot runs give a better estimate.

[Power](glossary.md#power-and-minimum-detectable-difference) for binary metrics is computed exactly for the [McNemar test](glossary.md#mcnemar-test) that `analyze` uses (normal
approximation above 3,000 items). See [statistical_methodology.md](statistical_methodology.md), section 19.
