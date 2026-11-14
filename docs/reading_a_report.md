# Reading a model comparison report

This guide walks through a real OpenBench AI v0.4.0 comparison report generated with the offline mock provider. It explains how to read the result, understand the statistical test, and interpret warnings without overstating what the evidence shows.

## 1. Generate the example report

From the repository root, run:

```bash
python examples/v040_workflow.py demo
```

The workflow runs locally using mock models and writes the Markdown comparison report to `demo/comparison.md`.

The mock provider is deterministic for this example, but it does not represent real model performance. The numbers below are illustrative of report interpretation, not a real-world model evaluation.

## 2. Start with the comparison direction

The example report compares:

- Baseline: `example-model-b`, one run.
- Challenger: `example-model-a`, a repeat group of three runs.

The report defines:

**Difference = challenger - baseline**

A positive difference favours the challenger; a negative difference favours the baseline. This direction is important when reading every result.

The comparison uses 20 aligned dataset items. For the repeat group, each item's score is averaged across its valid repetitions. The sample size is 20 items, not 60 observations.

## 3. Read the difference and confidence interval

Here is an excerpt from the generated report:

```text
| Metric      | n  | Baseline mean | Challenger mean | Difference | 95% CI (not multiplicity-adjusted) |
|-------------|----|---------------|-----------------|------------|------------------------------------|
| exact_match | 20 | 1.0000        | 0.7500          | -0.2500    | [-0.3667, -0.1333]                 |
| f1          | 20 | 1.0000        | 0.7500          | -0.2500    | [-0.3667, -0.1333]                 |
```

Both metrics have a mean difference of -0.25.

Because the difference is challenger minus baseline, this means the challenger scored 0.25 points lower on average.

The 95% confidence interval is [-0.3667, -0.1333].

The interval is entirely below zero, which indicates that the observed difference favours the baseline.

A confidence interval describes uncertainty about the mean difference for a population of similar items, given these fixed models, prompts, settings and references.

Important:
- These intervals are not adjusted for multiple comparisons.
- They do not measure run-to-run generation noise.
- They do not establish equivalence.
- Very small samples can produce unreliable interval coverage.

## 4. Understand the statistical test and adjusted p-value

The report uses a paired sign-flip permutation test for both metrics.

Why?

- `f1` is a bounded continuous metric.
- `exact_match` is binary, but one arm has repeated runs, so the sign-flip procedure is selected.

The report shows:

```text
Test: sign_flip
p: 0.0009766
Adjusted p: 0.001953
Correction: holm
Alpha: 0.05
Permutation distribution: exact_enumeration
```

The raw p-value measures evidence against the test's null hypothesis.

The adjusted p-value accounts for the family of two metric comparisons using Holm correction.

Since 0.001953 is below the significance threshold of 0.05, both reported differences are statistically significant.

### What does the validity label mean?

The report states:

```text
sharp_exchangeability_null: exact
mean_difference_null: approximate
```

These labels describe different assumptions:

- `exact`: the sign-flip test is exact under the sharp exchangeability null, where the two arms are interchangeable on every item.
- `approximate`: inference about a zero mean difference is approximate, relying on assumptions about independent items and finite variance.

`exact_enumeration` describes how the permutation distribution was calculated. It does not mean every possible interpretation of the test is exact.

See [Statistical methodology](statistical_methodology.md) for the assumptions.

## 5. Understand wins, losses and ties

The report contains:

```text
Wins/losses/ties (challenger vs baseline): 0 / 11 / 9
```

This means:

- Wins: 0 items where the challenger scored higher.
- Losses: 11 items where the challenger scored lower.
- Ties: 9 items where both scores were effectively equal.

The counts help explain the direction of the overall difference.

They complement the mean difference; they do not replace the statistical test.

## 6. Read validity labels and warnings

The example report contains several warnings.

### `generation_configuration_differs`

The generation configuration differs because the seed settings are different.

This is a comparability caveat that should be considered when interpreting the results.

### `mock_provider`

Both arms use the mock provider.

The report is useful for learning the workflow, but its results do not represent real model performance.

### `repetition_counts_differ`

The baseline has one run, while the challenger has three repetitions.

Per-item means therefore use different numbers of repetitions across the arms.

### `low_power_wide_intervals`

The comparison has 20 aligned items, fewer than the warning threshold of 30.

This indicates low statistical power and potentially wide intervals. A statistically significant result should still be interpreted in light of the limited sample.

### Compatible for inference

The report says:

```text
compatible for inference
```

This means the inputs passed the hard compatibility checks required for inference.

It does not mean that there are no caveats. Warnings remain important and should be considered before drawing conclusions.

## 7. Understand unavailable classifications

The example also reports:

```text
Unavailable equivalence: no_equivalence_margin
Unavailable practical classification: no_min_practical_difference
```

These are not failed tests.

They mean:

- No equivalence margin was supplied before the analysis, so equivalence was not evaluated.
- No minimum practical difference was specified, so practical importance was not classified.

Statistical significance and practical importance are different questions.

A statistically significant result is not automatically practically important.

## 8. Read the limitations block

The report's limitations section highlights several important boundaries:

- Results apply to the analysed models, prompts, generation settings, references and items.
- A non-significant difference does not establish equivalence.
- Confidence intervals are not adjusted for multiple comparisons.
- Continuous-metric equivalence testing is disabled in v0.4.0.
- Category and difficulty breakdowns are descriptive unless stratum tests are requested.

Always read this section before using a comparison to make a model-selection decision.

## 9. Three common situations

### Situation A: A clear statistical win

Suppose a comparison has a positive difference, an adjusted p-value below the selected alpha, and a confidence interval entirely above zero.

Interpretation:

The challenger has a statistically significant higher score than the baseline under the assumptions of the analysis.

Still check practical importance, warnings and the evaluation dataset before making a decision.

### Situation B: No statistically significant difference

Suppose the adjusted p-value is greater than or equal to alpha.

Interpretation:

The analysis did not detect a statistically significant difference.

This does not mean the models are equivalent or perform identically. The confidence interval may still include meaningful positive and negative differences.

Equivalence requires a pre-specified margin and a valid formal equivalence test.

### Situation C: A result with warnings

Suppose a comparison is statistically significant but includes warnings about a mock provider, different generation settings or a small sample.

Interpretation:

The numerical result may be statistically significant under the test assumptions, but the warnings limit how confidently it can be generalized.

Investigate the warnings and, where appropriate, run a better-controlled comparison before making a real model-selection decision.

## 10. Recommended reading order

When opening a new comparison report, use this order:

1. **Arms:** identify the baseline and challenger.
2. **Difference and CI:** understand the size and direction of the result.
3. **Test and adjusted p-value:** assess statistical evidence.
4. **Validity labels:** understand the test's assumptions.
5. **Wins/losses/ties:** see how individual items contributed.
6. **Warnings:** identify limitations in the comparison.
7. **Limitations and provenance:** check what the report can and cannot establish.

For the full statistical definitions, see [Statistical Methodology](statistical_methodology.md) and [Paired Model Comparison](model_comparison.md).
