# OpenBench AI roadmap: v0.6.0

> **Status: planning.** v0.5.0 is released; see the [changelog](CHANGELOG.md). This page describes what v0.6.0 is meant to do. There are no dates: work lands when it meets its gate. Tell us what you would use in the [v0.6.0 ideas discussion](https://github.com/lakshaychhaba0001-sys/openbench-ai/discussions/153).

**Goal:** OpenBench AI should be as useful to one person choosing a model for their own work as it is to a small team running evaluations every week, and it should spend as few of their credits as possible to get there.

v0.5.0 answers the questions a team with a benchmark faces: which model to ship, how many questions are enough, and whether a judge or a dataset can be trusted. v0.6.0 adds the questions that come next:

- Which model should I use for *my* kind of work, and is the more expensive one worth it?
- What did this comparison cost in tokens, money and time?
- How do I evaluate work that has no answer key?
- Can I get the same answer while paying for fewer model calls?
- Which prompt or setting is better, not only which model?
- Did the model get worse since last week?

These principles stay fixed:
- The core install has no runtime dependencies.
- Every number carries its provenance.
- The tool never claims more than the statistics support.
- Prices are never built in. Cost is computed only from prices you supply.

Three are new for v0.6.0:
- **Spend the least that answers the question.** Anything that calls a model shows the expected cost first and reuses stored answers.
- **Honest with small numbers.** With three tasks, the report says what three tasks can and cannot show.
- **Private by default.** Your own tasks stay on your machine and go only to the provider you chose.

Everything new is opt-in, and v0.5.0 commands, runs, reports and APIs keep working.

## Themes

| # | Theme | What users get | Related issues |
| --- | --- | --- | --- |
| 1 | **Which model for my work?** | Bring a few of your own tasks and a list of candidate models, and get one report that recommends a model, with quality, tokens, cost, time and run-to-run noise side by side. When a few tasks cannot separate the models, the report says so and estimates how many more would | |
| 2 | **Count the cost** | Tokens, cost and time in every comparison and on the dashboard; budget caps, sampling, and a dry run that shows the estimate before anything is spent | |
| 3 | **Tasks without an answer key** | Open-ended tasks with no reference answer; a rubric judge; a local page for reviewing two answers blind; records that accept any of several answers | [#69](https://github.com/lakshaychhaba0001-sys/openbench-ai/issues/69), [#67](https://github.com/lakshaychhaba0001-sys/openbench-ai/issues/67), [#79](https://github.com/lakshaychhaba0001-sys/openbench-ai/issues/79), [#92](https://github.com/lakshaychhaba0001-sys/openbench-ai/issues/92) |
| 4 | **Spend less to learn the same** | Small pilots that feed the sample-size planner; stopping early once the answer is clear; stored answers reused everywhere | |
| 5 | **Compare more than models** | Prompts, settings and reasoning effort compared with the same statistics as models; multi-turn tasks | [#70](https://github.com/lakshaychhaba0001-sys/openbench-ai/issues/70) |
| 6 | **Evaluation as a routine** | One suite file; a baseline check that fails CI when a model gets worse; trends over time; native Anthropic and Gemini adapters; import from the Hugging Face Hub; clustered intervals, Bradley–Terry ratings and ranking against a baseline; HTML for every report; a leaderboard page | [#71](https://github.com/lakshaychhaba0001-sys/openbench-ai/issues/71), [#72](https://github.com/lakshaychhaba0001-sys/openbench-ai/issues/72), [#73](https://github.com/lakshaychhaba0001-sys/openbench-ai/issues/73), [#89](https://github.com/lakshaychhaba0001-sys/openbench-ai/issues/89), [#90](https://github.com/lakshaychhaba0001-sys/openbench-ai/issues/90), [#91](https://github.com/lakshaychhaba0001-sys/openbench-ai/issues/91), [#93](https://github.com/lakshaychhaba0001-sys/openbench-ai/issues/93) |
| 7 | **First five minutes** | The built-in datasets and examples shipped inside the package; setup and self-check commands | [#84](https://github.com/lakshaychhaba0001-sys/openbench-ai/issues/84), [#85](https://github.com/lakshaychhaba0001-sys/openbench-ai/issues/85), [#86](https://github.com/lakshaychhaba0001-sys/openbench-ai/issues/86), [#87](https://github.com/lakshaychhaba0001-sys/openbench-ai/issues/87), [#94](https://github.com/lakshaychhaba0001-sys/openbench-ai/issues/94) |

Themes 1–3 are the core of the release. Two items are conditional:

- Stopping early ships only if its method passes a calibration study, like every statistical method in v0.5.0.
- Agent and tool-use tasks follow the multi-turn design and may move to a later release.

**Out of scope:**
- a hosted service;
- training or fine-tuning;
- model serving;
- a built-in price list;
- any required dependency in the core install.

## Order of work

The stages run in order, and each one ends with something usable. There are no fixed dates.

| Stage | Main work | Ends when |
| --- | --- | --- |
| **1. Cost foundations** | Pricing input, reasoning tokens, sampling, dry run, budget cap, cost columns in comparisons and rankings | A two-model comparison shows quality and cost with intervals |
| **2. Open-ended tasks** | Tasks without a reference answer, rubric judge, blind review page, any-of references | A task set with no answer key can be scored three ways |
| **3. Which model for my work?** | The guided workflow and its recommendation report | A person with three tasks and three models gets an honest recommendation |
| **4. Smarter spending and variants** | Pilot-then-plan, early stopping, prompts and settings as named variants | A comparison stops early with a valid result |
| **5. Routine for a lab** | Suite file, baseline check, trends, new adapters, Hub import, multi-turn tasks | A lab can run one command in CI |
| **6. Harden and release** | Independent review of the new statistics, documentation, release candidate | Every release gate is green on GitHub Actions |

## Release gates

v0.6.0 ships only when all of these pass on GitHub Actions:

- **No breaking changes:** v0.5.0 commands, public Python names, run directories, datasets and scores keep working.
- **Dependencies:** the core install stays dependency-free; new libraries live in optional extras.
- **Checked statistics:** every new statistical method has reference tests, a committed calibration or coverage study, and a methodology section.
- **Determinism:** analysis output is identical across Python 3.10–3.13 and across Linux, Windows and macOS.
- **Security:** secrets are scrubbed everywhere, HTML output is escaped, local pages listen on localhost only, and plugins are documented as trusted code.
- **Real models:** the weekly real-model workflow has passed three times in a row.
- **Cost honesty:** every cost figure names its price source and says whether tokens were reported by the provider or estimated.
- **Spending safety:** when prices are known, nothing calls a paid provider without showing the estimate first, and a budget cap is never exceeded.

## How to help

The core of the release (themes 1–4) is maintainer-led, because the design and the statistics need one owner. There is plenty around it:

- [`good first issue`](https://github.com/lakshaychhaba0001-sys/openbench-ai/labels/good%20first%20issue): small, well-scoped tasks with pointers to the code. Comment on the issue to claim it before you start.
- [`help wanted`](https://github.com/lakshaychhaba0001-sys/openbench-ai/labels/help%20wanted): larger features. Please comment with your plan before starting, so we can agree on the design.
- The [v0.6.0 ideas discussion](https://github.com/lakshaychhaba0001-sys/openbench-ai/discussions/153): say which themes you would use and what is missing.

Other questions and ideas are welcome in [Discussions](https://github.com/lakshaychhaba0001-sys/openbench-ai/discussions). This roadmap is updated as stages close.

## Earlier releases

v0.5.0 delivered all six of its themes: any model through one OpenAI-compatible adapter, ranking and sample-size planning, a measurable LLM judge, benchmark import and audits, HTML reports with a local dashboard, and plugins. The [changelog](CHANGELOG.md) has the details.
