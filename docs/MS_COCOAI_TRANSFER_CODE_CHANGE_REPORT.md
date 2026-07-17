# MS COCOAI Transfer Code Change Report

## Baseline and compatibility

- Base branch: `exp-ddfsd-dual-domain-margin-v1`
- Audited base commit: `bcd725acc49c366d07cf46da8fa65f853428bf9e`
- Development branch: `exp-ddfsd-ms-cocoai-transfer-allsource-v1`
- Default training behavior remains leave-one-out.
- All-source remains a three-way episode (`real + random two fake`).
- FSD was not changed.

## Implemented source areas

- Training scope resolution, training logs, validation split behavior, checkpoint metadata, and strict resume protection.
- Provenance-bearing GenImage frequency statistics, SHA sidecar preparation, and checkpoint SHA/metadata binding.
- Streaming Parquet row-group extraction of original bytes with SHA verification and provenance.
- Stable caption/label occurrence grouping with anomaly output.
- Fixed test 5-seed × 10-shot and validation smoke manifests with path/group/SHA leakage checks and a machine-readable lock.
- External manifest Dataset and no-gradient DDFSD transfer evaluator with formal-artifact validation and overwrite protection.
- Per-image scores, metrics/config/provenance JSON, and independent ACC/AP/AUC summarization.
- AutoDL training/smoke/formal/acceptance scripts.
- Pure-standard-library grouping and split tests.

## Validation status vocabulary

The final task report must distinguish:

1. **Locally verified:** diff whitespace, shell syntax, Python syntax when available, pure standard-library tests, and manual code review.
2. **Written but not run locally:** imports and execution that require torch/torchvision/timm/Pillow/PyArrow/scikit-learn or real artifacts.
3. **Must be verified on AutoDL:** dependency imports, pytest in the project environment, real Parquet extraction, 1,500/7,500 groups, fixed manifests, real checkpoint load, smoke inference, all-source training, and formal test.

No dependency installation, model/data download, training, inference, or Parquet extraction belongs in local validation.
