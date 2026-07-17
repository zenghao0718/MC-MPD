#!/usr/bin/env bash
set -euo pipefail

MODE="code-only"
if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [--code-only|--data-smoke]" >&2
    exit 2
fi
if [[ $# -eq 1 ]]; then
    case "$1" in
        --code-only) MODE="code-only" ;;
        --data-smoke) MODE="data-smoke" ;;
        *) echo "Unknown mode: $1" >&2; exit 2 ;;
    esac
fi

echo "branch=$(git branch --show-current)"
echo "commit=$(git rev-parse HEAD)"
python - <<'PY'
import importlib
import sys

print("python", sys.version)
for name in ("torch", "torchvision", "timm", "pyarrow", "pandas", "sklearn"):
    module = importlib.import_module(name)
    print(name, getattr(module, "__version__", "unknown"))
PY

PYTHON_FILES=(
    datasets/ddfsd_datasets.py
    datasets/transfer_manifest_dataset.py
    util/ddfsd_frequency.py
    train_ddfsd.py
    test_ddfsd_transfer.py
    tools/extract_ms_cocoai_parquet.py
    tools/build_ms_cocoai_groups.py
    tools/build_ms_cocoai_fewshot_manifests.py
    tools/summarize_ms_cocoai_ddfsd.py
    tests/test_ms_cocoai_logic.py
)
python -m py_compile "${PYTHON_FILES[@]}"
SHELL_FILES=(
    scripts/train_ddfsd_allsource.sh
    scripts/run_ms_cocoai_ddfsd_smoke.sh
    scripts/run_ms_cocoai_ddfsd_formal.sh
    scripts/validate_ms_cocoai_transfer_code.sh
)
for script in "${SHELL_FILES[@]}"; do
    bash -n "${script}"
done
python -m pytest -q tests/test_ms_cocoai_logic.py

if [[ "${MODE}" == "data-smoke" ]]; then
    GENIMAGE_ROOT=${GENIMAGE_ROOT:-"/root/autodl-tmp/data_fsd_full/GenImage"}
    MS_COCOAI_ROOT=${MS_COCOAI_ROOT:-"/root/autodl-tmp/MS_COCOAI"}
    for class_name in real ADM BigGAN glide Midjourney SD VQDM; do
        [[ -d "${GENIMAGE_ROOT}/${class_name}/train" ]] || {
            echo "Missing GenImage train directory: ${GENIMAGE_ROOT}/${class_name}/train" >&2
            exit 1
        }
    done
    shopt -s nullglob
    validation_parquet=("${MS_COCOAI_ROOT}"/validation/validation-*.parquet)
    test_parquet=("${MS_COCOAI_ROOT}"/test/test-*.parquet)
    [[ ${#validation_parquet[@]} -eq 2 ]] || { echo "Expected 2 validation Parquet shards" >&2; exit 1; }
    [[ ${#test_parquet[@]} -eq 8 ]] || { echo "Expected 8 test Parquet shards" >&2; exit 1; }
    SMOKE_ROOT=${SMOKE_ROOT:-"/root/autodl-tmp/runs/transfer_ms_cocoai/acceptance/data_smoke"}
    SMOKE_MAX_ROWS=${SMOKE_MAX_ROWS:-3000}
    python tools/extract_ms_cocoai_parquet.py \
        --input_root "${MS_COCOAI_ROOT}" \
        --output_root "${SMOKE_ROOT}" \
        --split validation \
        --max_rows "${SMOKE_MAX_ROWS}" \
        --verify_existing
    python tools/build_ms_cocoai_groups.py \
        --base_manifest "${SMOKE_ROOT}/manifests/validation/base_rows.csv" \
        --output_dir "${SMOKE_ROOT}/manifests/validation" \
        --allow_anomalies
    python tools/build_ms_cocoai_fewshot_manifests.py \
        --grouped_manifest "${SMOKE_ROOT}/manifests/validation/all_rows_grouped.csv" \
        --output_dir "${SMOKE_ROOT}/manifests/validation/fewshot" \
        --split validation
fi

echo "Validation mode ${MODE} completed. No training or formal inference was started."
