# OpenBench AI roadmap: v0.5.0

**Goal:** a team can go from "we have three candidate models" to a shareable, statistically defensible decision report in one afternoon, on their own hardware or any hosted API.

v0.4.0 answers one question well: *is the gap between two models real?* v0.5.0 widens that to the questions teams face every week:

- Which of these ten models should we ship, ranked with uncertainty?
- How many questions do we need before the answer means anything?
- Can we trust an LLM judge for open-ended answers, and how much?
- Is this benchmark sound: contaminated, duplicated or ambiguous?
- Can someone read the result without running Python?

Three principles stay fixed:
- The core install has no runtime dependencies.
- Every number carries its provenance.
- The tool never claims more than the statistics support.

Everything new is opt-in, and v0.4.0 runs, reports and APIs keep working.

## Themes

| # | Theme | What users get | Tracking |
| --- | --- | --- | --- |
| 1 | **Any model** | One adapter for every OpenAI-compatible server (vLLM, LM Studio, llama.cpp, hosted APIs); concurrent execution with rate limits; token and cost accounting | [#8](https://github.com/lakshaychhaba0001-sys/openbench-ai/issues/8) |
| 2 | **Statistics 2.0** | `openbench rank`: many models ranked with uncertainty. `openbench plan`: sample-size and power planning. Calibrated equivalence for continuous scores. Variance components for repeated runs | [#20](https://github.com/lakshaychhaba0001-sys/openbench-ai/issues/20) |
| 3 | **Judges you can measure** | LLM-as-judge with order swapping, length and self-preference checks, and agreement against human ratings before its scores are used | [#19](https://github.com/lakshaychhaba0001-sys/openbench-ai/issues/19) |
| 4 | **Benchmarks you can trust** | Import standard benchmarks, semantic near-duplicate checks, contamination risk checks, generated dataset cards | [#21](https://github.com/lakshaychhaba0001-sys/openbench-ai/issues/21), [#9](https://github.com/lakshaychhaba0001-sys/openbench-ai/issues/9) |
| 5 | **Results anyone can read** | Self-contained HTML reports and a read-only local dashboard (`openbench serve`) | [#18](https://github.com/lakshaychhaba0001-sys/openbench-ai/issues/18), [#4](https://github.com/lakshaychhaba0001-sys/openbench-ai/issues/4) |
| 6 | **A platform others build on** | Plugins for metrics, adapters and datasets; Windows and macOS support; weekly real-model CI; cross-process cache locking; a documentation site | [#7](https://github.com/lakshaychhaba0001-sys/openbench-ai/issues/7), [#16](https://github.com/lakshaychhaba0001-sys/openbench-ai/issues/16), [#17](https://github.com/lakshaychhaba0001-sys/openbench-ai/issues/17), [#15](https://github.com/lakshaychhaba0001-sys/openbench-ai/issues/15) |

Themes 1–3 are the core of the release. Themes 4–6 can move to v0.6.0 without blocking it.

**Out of scope:**
- a hosted service;
- training or fine-tuning;
- model serving;
- any required dependency in the core install.

Agent and multi-turn evaluation is planned for after v0.5.0.

## Milestones

Dates are targets, not promises. Each milestone ends in a gate and an installable pre-release on PyPI, so you can try features early.

| Milestone | Target | Main work | Gate |
| --- | --- | --- | --- |
| **1. Foundations** | Dec 2026 – Jan 2027 | Plugin interface, OpenAI-compatible adapter, concurrency and costs, Windows/macOS CI, cache locking, real-model CI | **Alpha 1**: any supported model runs the full workflow end to end |
| **2. Statistics and judges** | Feb – Mar 2027 | Multi-model ranking, sample-size planner, LLM judge with bias checks, judge–human agreement, variance components, continuous equivalence | **Alpha 2**: every new method passes its reference tests and calibration |
| **3. Benchmarks and reports** | Apr 2027 | Benchmark import, semantic duplicates, contamination check, dataset cards, HTML reports, local dashboard | **Beta**: feature complete |
| **4. Harden and release** | May 2027 | Independent review of the new statistics, documentation site, release candidate | **v0.5.0**: every release gate green on GitHub Actions |

## Release gates

v0.5.0 ships only when all of these pass on GitHub Actions:

- **No breaking changes:** v0.4.0 commands, public Python names, run directories, datasets and scores keep working.
- **Dependencies:** the core install stays dependency-free; new libraries live in optional extras.
- **Checked statistics:** every new statistical method has reference tests, a committed calibration or coverage study, and a methodology section.
- **Determinism:** analysis output is identical across Python 3.10–3.13 and across Linux, Windows and macOS.
- **Security:** secrets are scrubbed everywhere, HTML output is escaped, the dashboard listens on localhost only, and plugins are documented as trusted code.
- **Real models:** the weekly real-model workflow has passed three times in a row.

## How to help

Every theme has an entry point. Look for these labels:

- [`good first issue`](https://github.com/lakshaychhaba0001-sys/openbench-ai/labels/good%20first%20issue): small, well-scoped tasks with pointers to the code.
- [`help wanted`](https://github.com/lakshaychhaba0001-sys/openbench-ai/labels/help%20wanted): larger features. Please comment with your plan before starting, so we can agree on the design.

Questions and ideas are welcome in [Discussions](https://github.com/lakshaychhaba0001-sys/openbench-ai/discussions). This roadmap is updated as milestones close; the [v0.5.0 roadmap issue](https://github.com/lakshaychhaba0001-sys/openbench-ai/issues?q=is%3Aissue+in%3Atitle+%22v0.5.0+roadmap%22) tracks progress.
