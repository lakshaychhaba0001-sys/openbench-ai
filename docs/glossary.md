# Glossary

This page defines statistical and reporting terms used across OpenBench AI reports, CLI output, and documentation. Each entry provides a plain-language summary and links to the guide that covers the concept in detail.

## Paired comparison

A **paired comparison** evaluates two models on the exact same set of benchmark items aligned by stable record ID, rather than comparing independent aggregate scores from separate runs. Pairing removes item-to-item difficulty variance because every difference or win/loss is computed within the same prompt before aggregating across the dataset. See [Paired comparison](model_comparison.md) and [Statistical methodology](statistical_methodology.md) for alignment rules and test selection.

## Confidence interval

A **confidence interval** (CI) gives a range of plausible values for an estimated parameter—such as a single model's mean score or the paired mean difference between two models—at a chosen confidence level (typically 95%). OpenBench AI uses Wilson intervals for binary proportions and BCa (bias-corrected and accelerated) bootstrap intervals for continuous metrics and paired differences. See [Statistical methodology](statistical_methodology.md) and [Reading a comparison report](reading_a_report.md) for how intervals are constructed and interpreted.

## Holm correction

The **Holm correction** (Holm–Bonferroni step-down procedure) adjusts $p$-values when multiple hypothesis tests are run simultaneously—such as comparing several metrics or multiple model pairs—to control the family-wise error rate. It orders raw $p$-values from smallest to largest and compares each against a progressively less strict threshold, making it uniformly more powerful than a classic Bonferroni correction while remaining valid under arbitrary dependence. See [Statistical methodology](statistical_methodology.md) and [Paired comparison](model_comparison.md) for how adjusted $p$-values are reported.

## TOST

**TOST** (Two One-Sided Tests) is an equivalence testing procedure used to demonstrate that two models perform within a pre-specified practical equivalence margin $(-\Delta, +\Delta)$. A standard difference test returning "not significant" never means two models are "equivalent"—it can simply reflect a small sample size or high variance, whereas TOST requires the entire confidence interval of the paired difference to fall strictly inside the margin. See [Statistical methodology](statistical_methodology.md) and [Reading a comparison report](reading_a_report.md) for configuring equivalence margins.

## Repeat group

A **repeat group** is a collection of benchmark runs executed with identical model, dataset, and configuration settings across multiple seeds or trials to measure run-to-run variability. Grouping repeated executions allows OpenBench AI to separate prompt-level variance from generation sampling noise and report stability metrics across trials. See [Repeated runs](repeated_runs.md) for running and analyzing multi-seed evaluations.

## Discordance

**Discordance** measures the rate at which two compared models disagree on binary or pass/fail items—specifically the off-diagonal pairs $(n_{10}$ where Model A passes and Model B fails, and $n_{01}$ where Model A fails and Model B passes). High discordance indicates that the models solve different subsets of items even if their overall accuracy looks similar, and the total discordant count $n_{10} + n_{01}$ determines the effective sample size for McNemar's test. See [Paired comparison](model_comparison.md) and [Statistical methodology](statistical_methodology.md) for binary contingency table details.

## Rank interval

A **rank interval** reports the range of plausible leaderboard ranks (for example, rank 1–2 out of 5) for a model across bootstrap resamples of the evaluation items. Rather than treating a tiny point-estimate gap as a definitive ordering, rank intervals show when two or more models are statistically indistinguishable in rank. See [Ranking and planning](ranking_and_planning.md) and [Statistical methodology](statistical_methodology.md) for multi-model ranking methodology.

## Descriptive only

A **descriptive only** label indicates that a reported metric or comparison is summarized without formal hypothesis testing or inferential claims—for example, when sample sizes are too small, pairing requirements are not met, or a metric is purely diagnostic (such as token counts or cost). When a result is marked descriptive only, treat the numbers as observed summary statistics for that run rather than evidence of a statistically significant difference or equivalence. See [Reading a comparison report](reading_a_report.md) and [Statistical methodology](statistical_methodology.md) for report guardrails.
