# OpenJev Roadmap

Last updated: September 19, 2026.

## First model preview: within the next few days

The first experimental OpenJev model preview will be released within the next few days.

首个模型预览版将在几天内发布。

The preview will report its training scope and limitations. Larger training runs and shared-attention optimization follow the milestones below.

### Preview deliverables

- [ ] Loadable model weights, or adapter weights together with the scoring head, new token embeddings, and an exact base-model revision.
- [ ] Tokenizer, input compiler configuration, candidate mode, and all artifacts required to reproduce inference.
- [ ] A minimal inference example with a documented environment and verified output.
- [ ] A completed [model card](docs/MODEL_CARD.md), including actual training scope, supported primitives, and limitations.
- [ ] Preliminary evaluations with dataset provenance, split policy, hardware, and reproducible commands.
- [ ] A versioned release with artifact checksums, license information, and download locations.

### Delivery sequence

| Step | Planned work | Completion evidence |
| --- | --- | --- |
| 1 | Research design and public-facing repository materials | Design, architecture, training, and contributor documentation |
| 2 | Environment setup, reference scorer, verified small dataset | Reproducible data records and a working training/inference path |
| 3 | Initial adaptation and preliminary evaluation | A trained checkpoint and recorded evaluation results |
| 4 | Package and release the experimental preview | Downloadable artifacts, inference instructions, and completed model card |

## Research milestones

| Milestone | Scope | Status |
| --- | --- | --- |
| Research documentation | Overall approach, architecture, data recipes, and training plan | Available in this repository |
| M0 | Task contracts, grouped splits, baseline evaluation, and real workflow definitions | Planned |
| M1 | Ordinary causal reference, shared scalar head, and 20K–50K decision adaptation | Planned |
| M2 | About 100K decision pilot; A/B, initialization, and supervision comparisons | Planned |
| M3 | Tree sharing, path positions, gradient equivalence, and performance profiling | Planned |
| M4 | 0.5M–2M decision training, hard-case refinement, and independent calibration | Planned |
| M5 | Deployment optimization and independent workflow acceptance evaluation | Planned |

The [training pipeline](docs/openjev_training_pipeline.md) defines the experiments and exit criteria for each milestone.

## How we decide what to do next

1. Establish decision quality using ordinary causal forwards.
2. Compare independent candidates with full candidate context on tasks that need it.
3. Check whether probabilities support useful automation coverage at the chosen risk level.
4. Optimize shared computation after the quality pilot passes.
5. Expand data and training only when experiments justify the cost.

Evaluation reports will include baselines, failed hypotheses, and regressions alongside gains.

## Current contribution opportunities

Useful early work includes schema/compiler reviews, verified data generators, real workflow evaluation tasks, reference model implementation, and documentation translations. Propose work through the repository's Issues tab using the feature or research template. See [CONTRIBUTING.md](CONTRIBUTING.md).
