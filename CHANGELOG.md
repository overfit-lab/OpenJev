# Changelog

Changes to the active implementation are recorded here. Planned work lives in [ROADMAP.md](ROADMAP.md); superseded experiments are described in the [archive record](docs/ARCHIVE.md).

## Unreleased

### Added

- Jev-compatible input compilation and typed Choice, Noul and Score responses through CLI and HTTP.
- A configurable causal scorer with a shared scalar head, PiSSA components and complete checkpoint reload support.
- API construction of unlabeled questions from grouped conversation prefixes, strict JSON validation, semantic review, candidate-count quotas, resume and offline replay.
- Local thinking teacher collection with single-token answer codes, complete candidate logits, temperature variants and replay validation.
- Regression tests for public contracts, data boundaries, resume, soft-target collection, gradients and checkpoint reloads.

### Changed

- Training follows data preparation → direct PiSSA soft distillation → validation. Hard-label warmup and RLCD are outside the current route.
- Consolidated training and data plans; runtime commands and record formats have separate guides.
- Archived old hard-label scripts, configurations and generated data. Preserved frozen source splits and exclusion anchors.
- Replaced full-text research copies with attributed source links; removed duplicate plans and outdated project status.

### Validation and limitations

- 43 local tests pass. The [pipeline canary](docs/data_audit/pipeline_canary.md) records actual generation quality and eight local teacher traces.
- No approved soft-label training dataset or trained OpenJev release exists. The full student trainer and independent evaluation remain pending.
