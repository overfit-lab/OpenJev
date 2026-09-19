# OpenJev — Model Card

**Status: unreleased.** Training details and evaluation results will be added with the first preview.

The first model preview will be released within the next few days. See the [roadmap](../ROADMAP.md) for release scope. No model download URL is available yet.

## Model identity

| Field | Current information |
| --- | --- |
| Project | OpenJev |
| Artifact version | Not released |
| Planned base | Qwen/Qwen3-1.7B-Base |
| Exact base revision | To be recorded at release |
| Artifact format | To be recorded: full weights, or complete adapter/head/embedding bundle |
| Choice architecture | A or B, to be selected and disclosed |
| Supported primitives | To be measured and declared for the released checkpoint |
| Supported languages and lengths | To be evaluated and declared |
| Weight / adapter license | To be recorded with upstream terms at release |
| Download location and checksums | Not available |

The model design uses candidate scoring in place of free-text answer decoding. See the [architecture specification](openjev_model_architecture.md); this card will record the released checkpoint's supported features.

## Intended use

The first preview is intended for research into text-based routing, policy judgments, candidate matching, and descriptive scoring. Use it to reproduce experiments and investigate probability quality in bounded workflows.

It is not yet validated for production automation or high-impact decisions. Type-valid outputs do not establish semantic correctness, calibration, robustness to distribution shift, or application safety.

## Training

Record the actual training stages completed, trainable modules, unique source/state/decision counts, candidate counts, token budgets, languages, task families, and data provenance. Distinguish verified labels, observed outcomes, known distributions, human distributions, and teacher labels.

Include the data split policy, generator and verifier versions, base revision, tokenizer changes, input compiler version, optimizer settings, random seeds, training hardware, runtime, and code commit. Do not substitute the planned 0.5M–2M decision budget for the actual preview's training size.

## Evaluation

Results are pending. Each reported metric will include the dataset, configuration, and reproduction procedure.

| Evaluation | Required context | Current result |
| --- | --- | --- |
| Decision quality | Dataset, held-out split, primitive, language, and candidate count | Not evaluated |
| NLL and Brier | Label semantics and evaluation distribution | Not evaluated |
| Calibration | Calibrator, calibration data, reliability analysis, and groups | Not evaluated |
| Workflow risk and coverage | Fixed workflow, fixed thresholds, sample counts, and intervals | Not evaluated |
| Latency and throughput | Hardware, dtype, lengths, candidate counts, batch/concurrency, and baseline | Not evaluated |
| Memory | Backend and full measurement conditions | Not evaluated |
| Perturbations and OOD | Input transformations, task shift, and failure categories | Not evaluated |

If the preview has not completed independent calibration, label its probabilities as unvalidated for calibration. If only a reference backend is included, do not claim tree-sharing speedups.

## Evaluation scope

- Supported and unsupported candidate-set semantics, including dynamic `other` and overlapping labels.
- Behavior for missing evidence, ambiguous instructions, and distribution shift.
- Task, language, length, and candidate-count coverage gaps.
- High-confidence errors and conditional workflow failures.
- Whether quantization, adapters, or different inference backends have been evaluated.
- Any use of teacher-generated labels and the limits of their validation.

## Loading and reproduction

Working installation and loading commands will be added when the artifacts exist. They must be verified in a clean environment and identify the exact model and code revisions. See [RELEASING.md](RELEASING.md).

## Attribution

OpenJev is independent of TypeSafe AI and Qwen. It uses public research descriptions as inspiration and plans to build on the Qwen backbone. See [third-party notices](../THIRD_PARTY_NOTICES.md) for provenance, and [CITATION.cff](../CITATION.cff) for the project citation.
