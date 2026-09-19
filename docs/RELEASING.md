# Releasing an OpenJev Model

Release checklist for model artifacts, evaluation, and documentation. The first preview's plan is in [ROADMAP.md](../ROADMAP.md).

## 1. Freeze the actual release scope

Record which primitives, languages, task families, candidate counts, and input lengths the checkpoint supports. State the selected Choice mode and whether inference uses the ordinary reference or a shared-computation backend.

For an experimental preview, document its supported tasks and known limitations in the model card.

## 2. Package a complete artifact

- [ ] Full model weights, or adapter weights plus the scoring head and new token embeddings.
- [ ] Exact base-model revision and instructions to obtain it when not bundled.
- [ ] Tokenizer files, structural token IDs, input compiler version, and model configuration.
- [ ] Backend, dtype, and environment requirements.
- [ ] Calibration parameters when provided, bound to the exact checkpoint and precision; otherwise an explicit uncalibrated status.
- [ ] Any demonstrated workflow policy and its version, separate from the model.
- [ ] Artifact checksums and versioned download locations.
- [ ] Applicable licenses and upstream notices, including training-data provenance disclosures appropriate to the release.

Do not put large weights or generated datasets directly into the source repository. Store them as versioned release artifacts or in a model registry, then link their actual locations.

## 3. Verify loading and inference

- [ ] Reproduce installation and model loading in a clean environment.
- [ ] Run a minimal example for each advertised primitive.
- [ ] Check that candidate IDs, probabilities, score statistics, and invalid-input behavior match the published contract.
- [ ] Confirm that every dependency needed for the head, embeddings, adapter, or tokenizer is present.
- [ ] Run `python3 scripts/check_docs.py` from the repository root.

Label measured examples with their model version and input; keep illustrative values identified as examples.

## 4. Publish evaluation evidence

Complete [MODEL_CARD.md](MODEL_CARD.md) using actual training and evaluation information. Include dataset definitions, source-group separation, actual data counts, seeds, configuration, commands, and hardware.

For probability or workflow claims, disclose which data fitted temperatures and thresholds and which independent data measured final risk. If the preview lacks these evaluations, say so instead of describing the model as calibrated or deployment-ready.

For speed claims, include a meaningful baseline, output information parity, input lengths, candidate counts, concurrency, precision, warmup, P50/P95, throughput, and memory. An optimized attention design is not itself performance evidence.

## 5. Update project metadata

- [ ] Mark only completed work in ROADMAP.md and update the release progress.
- [ ] Add the actual release version, date, and changes to CHANGELOG.md.
- [ ] Update both READMEs with working artifact links and inference instructions.
- [ ] Replace the unreleased model-card fields with actual values or explicit not-evaluated entries.
- [ ] Update CITATION.cff with the release version, release date, and actual repository/model URLs when known.
- [ ] State the support scope in SECURITY.md for released software.

## 6. Publish and verify

Create a versioned release with the artifact manifest, model card, inference example, evaluation report, and license files. Verify that a reader can access the artifact links and reproduce loading without private paths or credentials.

Mark the release as published once the artifacts are accessible and the loading instructions have been verified.
