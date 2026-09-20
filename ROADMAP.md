# OpenJev Roadmap

Last updated: September 20, 2026.

## First model preview

The first experimental OpenJev model preview is coming soon.

首个模型预览版即将发布。

The preview will report its training scope and limitations. Larger training runs and shared-attention optimization follow the milestones below.

### Preview deliverables

- [ ] Loadable model weights, or adapter weights together with the scoring head, new token embeddings, and an exact base-model revision.
- [ ] Tokenizer, input compiler configuration, candidate mode, and all artifacts required to reproduce inference.
- [ ] A minimal inference example with a documented environment and verified output.
- [ ] A completed [model card](docs/MODEL_CARD.md), including actual training scope, supported primitives, and limitations.
- [ ] Preliminary evaluations with dataset provenance, split policy, hardware, and reproducible commands.
- [ ] A versioned release with artifact checksums, license information, and download locations.

### Three-stage plan

| Stage | Planned work | Completion evidence |
| --- | --- | --- |
| 1. Prepare data | Extract chat inputs; API question/option construction; JSON and semantic validation; local reasoning teacher probabilities | About 100 reviewed decisions, then 1K–3K; complete distributions and 2–255 candidate coverage |
| 2. Direct PiSSA distillation | Configurable backbone; joint low-rank, scoring-head and structural-token training on soft labels only | Reloadable checkpoint and development evaluation; no hard-label warmup |
| 3. Validate | Evaluate quality, fit independent calibration when needed, select policy, test frozen system | Quality/cost report, supported scope and release artifacts |

The first preview follows **1 → 2 → 3**; RLCD is outside the current plan. Standard PiSSA freezes the residual base while adapting effective backbone weights through low-rank parameters. See the [training plan](docs/openjev_training_pipeline.md) for parameter scope and implementation gaps.

The [API contract](docs/openjev_api_contract.md) aligns request and response fields with Jev, including Choice criteria and named answers. OpenJev defines its own confidence statistic and does not claim numerical equivalence. A local reference HTTP service and CLI now implement all three typed outputs and usage. Quality of the trained model remains unvalidated.

LoRA comparisons, A/B ablations, larger datasets, tree sharing, and quantization follow evidence from the initial validation. They are not additional mandatory preview stages.

## Active data plan

The [conversation data plan](docs/openjev_training_pipeline.md) is the current execution plan. Keep existing user messages and necessary history; generate the decision questions and candidate sets. Use the configured LLM API for question construction and review, and a local reasoning teacher for complete candidate distributions. Start with Choice and Noul; add Score only when a meaningful rubric and labels are available.

- Current source data: [five frozen splits](docs/data_audit/README.md), including 6,095 train prefixes. Protected source assignments remain unchanged.
- Archived: the earlier 101-question hard-label trial, 70-question retained set, remote preflight, and their scripts/configs/reports. They are not current training inputs.
- Implemented: the public request compiler, typed responses, reference scorer, PiSSA model/checkpoint components, and the [255-code tokenizer audit](docs/data_audit/teacher_codebook.md).
- Implemented: unlabeled API synthesis, strict JSON and semantic checks, deterministic K quotas, resume, offline replay, sample export and local reasoning/logit collection. See the [canary report](docs/data_audit/pipeline_canary.md) for measured coverage and remaining defects.
- Next: improve question quality and reviewed large-K catalogues, approve about 100 complete soft-labeled decisions, then expand and implement direct PiSSA soft-target training. No active training entrypoint or approved soft-label dataset exists yet.

Superseded protocols, source intermediates, old datasets and one-off scripts are [archived](docs/ARCHIVE.md). Current commands are in the [execution guide](docs/execution.md).

Teacher credentials and raw conversations stay outside versioned files. The current user instruction is to perform no automatic staging, commit, push, or data publication; any later Git operation requires an explicit instruction covering its scope.

## How we decide what to do next

1. Establish decision quality using ordinary causal forwards.
2. Start with full candidate context; compare independent candidates later where justified.
3. Check whether probabilities support useful automation coverage at the chosen risk level.
4. Optimize shared computation after the quality pilot passes.
5. Expand data and training only when experiments justify the cost.

Evaluation reports will include baselines, failed hypotheses, and regressions alongside gains.

## Current contribution opportunities

Useful early work includes schema/compiler reviews, verified data generators, real workflow evaluation tasks, reference model implementation, and documentation translations. Propose work through the repository's Issues tab using the feature or research template. See [CONTRIBUTING.md](CONTRIBUTING.md).

Direct teacher-distribution distillation is the planned training objective, not an implementation of Jev RLCD. Preserve public request/response fields, version question banks and soft targets separately, and require independent evaluation. Incomplete scores or poor question quality block data freezing; there is no automatic fallback to hard-label training.
