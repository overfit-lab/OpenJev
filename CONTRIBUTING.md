# Contributing to OpenJev

Issues, pull requests, and documentation contributions are welcome in English or Chinese.

Current work covers the reference model, training data, and evaluation. See the [roadmap](ROADMAP.md) for progress.

## Where to start

- Review the [overall design](docs/openjev_design.md), [architecture](docs/openjev_model_architecture.md), and [training pipeline](docs/openjev_training_pipeline.md).
- Open an issue for a bug, a focused feature proposal, or a research question.
- Small documentation fixes can go directly into a pull request. Discuss changes to model semantics, data splits, loss functions, or release scope before substantial implementation.
- Follow our [Code of Conduct](CODE_OF_CONDUCT.md). Sensitive reports belong in the process described in [SECURITY.md](SECURITY.md).

## Local workflow

Fork the repository, create a branch for your change, and run:

```bash
python3 scripts/check_docs.py
python3 -m unittest scripts.test_chat_records scripts.test_prepare_chat_pool scripts.test_contract scripts.test_teacher_codes scripts.test_questions scripts.test_teacher_collection
```

The check requires Python 3.10+ and validates local documentation links, code fences, JSON examples, and SVG assets. It uses only the standard library.

The [conversation data plan](docs/openjev_training_pipeline.md) covers corpus preparation and teacher labeling. Historical synthetic experiments are listed in the [archive record](docs/ARCHIVE.md). The [execution guide](docs/execution.md) includes the scoring-model environment, data synthesis commands, and numerical tests.

## Documentation and examples

Keep the English and Chinese READMEs aligned when changing project status, supported capabilities, or release dates. Changes to technical specifications should update the relevant detailed document and its summary links.

Label examples and proposals clearly. Benchmark results should include the model version and reproduction details; installation and download instructions should be verified before merging.

Use relative links inside the repository. Keep diagrams as editable text or SVG where practical. Provide alternative text for images, and avoid requiring external fonts or scripts to understand a diagram.

Keep machine-specific absolute paths out of source files, documentation, configuration, and shareable reports. Supply local dataset directories through runtime arguments such as --root and record provenance relative to that directory.

## Code contributions

Keep the input compiler, reference scorer, optimized scorer, loss computation, calibration, and workflow evaluation responsibilities separate. Add checks appropriate to the change and describe how you validated it.

For changes to shared computation, include reference comparisons for logits, loss, and gradients. For performance claims, report hardware, dtype, input sizes, candidate counts, batch/concurrency, warmup, and the baseline configuration. A lower theoretical operation count alone is not a measured speedup.

## Data and model contributions

- Share only material you are entitled to contribute, with source and license metadata. Avoid including credentials, personal records, or proprietary business data in examples.
- Preserve source groups and transformation families across splits. Do not turn validation or test failures into training examples while continuing to call the old test independent.
- Record whether labels are verified facts, observed outcomes, known distributions, human distributions, or teacher outputs.
- Describe the generator and verifier, their versions, and failure cases. A teacher's confident response is not a verified label.
- Keep large datasets and weights out of Git. Use versioned artifact storage and include checksums and loading instructions in a release.

Model contributions must include a completed model card. See [RELEASING.md](docs/RELEASING.md) for the required artifact metadata.

## Pull requests

Use a descriptive title. Explain the problem, the resulting behavior, the validation performed, and any limitations. Link the relevant issue when one exists. Keep unrelated edits in separate PRs.

For research changes, state the hypothesis and provide enough experiment detail for another contributor to reproduce the conclusion. Negative results are useful contributions.

By contributing original material, you agree to make it available under the project's [Apache-2.0 license](LICENSE), unless a separately documented arrangement applies. Do not relicense third-party material merely by adding it to this repository; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
