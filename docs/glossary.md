# Glossary

## Confidence interval

A confidence interval shows the uncertainty around an estimate, such as the average score difference between two models. Over many equivalent studies, a well-calibrated 95% interval procedure would cover the true value about 95% of the time. OpenBench's paired bootstrap resamples whole items so the two scores for an item stay together.

See [Reading the difference and confidence interval](reading_a_report.md#3-read-the-difference-and-confidence-interval).

## Descriptive only

A descriptive-only result summarizes the observed scores without a formal test decision for that result. For example, a category breakdown can give a mean difference and a bootstrap interval while leaving out its adjusted p-value. An exploratory report can also be descriptive-only when the inputs fail compatibility checks.

See [What `descriptive_only` means on breakdowns](model_comparison.md#what-descriptive_only-means-on-breakdowns).

## Discordance

For paired binary scores, an item is discordant when one model succeeds and the other fails. The discordance rate is the share of items where that happens in either direction. Higher discordance generally means more items are needed to detect the same accuracy difference.

See [Sample-size planning](ranking_and_planning.md#openbench-plan).

## Equivalence margin

An equivalence margin is the largest absolute difference that is still considered practically the same, fixed before analysis. Equivalence is decided only by a formal two one-sided tests (TOST) procedure against that margin; a non-significant difference test does not establish equivalence.

See [Significance, practical importance and equivalence](statistical_methodology.md#7-significance-practical-importance-and-equivalence-are-separate).

## Estimand / unit of analysis

The unit of analysis is the dataset item. With repeated runs, an item's value is the mean of its metric over that model's valid repetitions, so the effective sample size is the number of items. The estimand for a paired comparison is the population mean of the within-item differences, treating the analysed items as a random sample from a larger population of similar items.

See [Unit of analysis and estimands](statistical_methodology.md#1-unit-of-analysis-and-estimands).

## Familywise error rate and Benjamini–Hochberg (FDR)

The familywise error rate is the chance of at least one false positive across a group of tests. OpenBench's default Holm correction controls this rate under arbitrary dependence. The Benjamini–Hochberg procedure instead controls the false discovery rate (expected proportion of false positives among rejected hypotheses) and is available as an option with a warning that it does not control the familywise error rate.

See [Multiple comparisons](statistical_methodology.md#6-multiple-comparisons).

## Holm correction

Holm correction adjusts a group of p-values to control the chance of at least one false positive across the group. It works through the ordered p-values, starting with the smallest. OpenBench adjusts difference tests and equivalence tests in separate families.

See [Multiple comparisons](statistical_methodology.md#6-multiple-comparisons).

## McNemar test

The exact McNemar test is used for binary metrics with one repetition per arm. It tests the null that, among discordant items, either direction is equally likely. The p-value is computed from the binomial distribution on the discordant counts in exact rational arithmetic.

See [Exact McNemar test](statistical_methodology.md#4-exact-mcnemar-test-binary-one-repetition-per-arm).

## Paired comparison

A paired comparison matches the two models' scores for each shared dataset item and analyzes the within-item differences. When an arm is a repeat group, OpenBench first averages its valid scores for that item. Each aligned item still contributes one difference.

See [Alignment and missing data](model_comparison.md#alignment-and-missing-data).

## Power and minimum detectable difference

Power is the probability that a planned study will reject the null when a true difference of a stated size exists. For binary metrics OpenBench computes it exactly from the McNemar test; the planning tools also report the minimum detectable difference for a chosen sample size and power.

See [Sample-size planning](statistical_methodology.md#19-sample-size-planning-openbench-plan-v050).

## Rank interval

A rank interval gives the best and worst ranks supported by the pairwise comparisons. After Holm correction, the best rank is one plus the number of models that scored significantly better, and the worst rank is the total number of models minus the number that scored significantly worse. A wide interval means the data leave several rankings plausible.

See [Ranking several models](statistical_methodology.md#18-ranking-several-models-openbench-rank-v050).

## Repeat group

A repeat group collects two or more runs from a repeated evaluation of one model and records their shared configuration and lineage. Repetitions show how much scores vary across runs. In a paired analysis, OpenBench averages them per item so they do not inflate the sample size.

See [Repeated runs](repeated_runs.md).

## Score kind (`score_kind`)

`score_kind` is a class attribute on each metric that selects the statistical procedure: `binary`, `bounded_continuous`, or `unknown`. It never changes the scores themselves. Binary metrics with one repetition use the McNemar test; otherwise a sign-flip test or interval-only analysis is used.

See [Metric kinds](statistical_methodology.md#2-metric-kinds-score_kind).

## Sign-flip permutation test

The paired sign-flip permutation test is used for bounded-continuous metrics and for per-item means of repeated runs. It tests the null of no difference by comparing the observed mean of the within-item differences against the distribution obtained by flipping the sign of each difference. Studentization does not change the p-value relative to the plain mean.

See [Paired sign-flip permutation test](statistical_methodology.md#5-paired-sign-flip-permutation-test-difference-test).

## TOST (two one-sided tests)

TOST checks whether a difference fits inside an equivalence margin chosen before the analysis. One test checks the lower bound and the other checks the upper bound, and both must pass with the applicable multiplicity adjustment. A non-significant difference test by itself leaves equivalence unresolved.

See [Significance, practical importance and equivalence](statistical_methodology.md#7-significance-practical-importance-and-equivalence-are-separate).

## Variance components

Variance components separate item variance from repetition variance in repeated runs. On complete cases the model is y_ir = mu + item_i + rep_ir; method-of-moments estimates give the share of variance that comes from re-running the model. Negative item-variance estimates are clipped at zero.

See [Variance components for repeated runs](statistical_methodology.md#20-variance-components-for-repeated-runs-v050).
