#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

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
    tools/prepare_ddfsd_allsource_freq_stats.py
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
    python tools/extract_ms_cocoai_parquet.py \
        --input_root "${MS_COCOAI_ROOT}" \
        --output_root "${MS_COCOAI_ROOT}" \
        --split validation \
        --verify_existing
    python tools/build_ms_cocoai_groups.py \
        --base_manifest "${MS_COCOAI_ROOT}/manifests/validation/base_rows.csv" \
        --output_dir "${MS_COCOAI_ROOT}/manifests/validation"
    python - "${MS_COCOAI_ROOT}/manifests/validation" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
def count_rows(name):
    with (root / name).open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))

assert count_rows("base_rows.csv") == 9000, "Validation extraction must contain 9000 rows"
assert count_rows("groups.csv") == 1500, "Validation must contain 1500 valid groups"
assert count_rows("group_anomalies.csv") == 0, "Validation grouping must have zero anomalies"
PY
    python tools/build_ms_cocoai_fewshot_manifests.py \
        --grouped_manifest "${MS_COCOAI_ROOT}/manifests/validation/all_rows_grouped.csv" \
        --output_dir "${MS_COCOAI_ROOT}/manifests/validation/fewshot" \
        --split validation
    python - "${MS_COCOAI_ROOT}/manifests/validation/fewshot/dalle3/seed_42" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
def labels(name):
    with (root / name).open(newline="", encoding="utf-8") as handle:
        return [int(row["label"]) for row in csv.DictReader(handle)]

support = labels("support.csv")
query = labels("query.csv")
assert support.count(0) == support.count(1) == 10, "Smoke support must be 10 real + 10 fake"
assert query.count(0) == query.count(1) == 100, "Smoke query must be 100 real + 100 fake"
PY
    python tools/build_ms_cocoai_fewshot_manifests.py \
        --verify_lock "${MS_COCOAI_ROOT}/manifests/validation/fewshot/manifest_lock.json"
fi

echo "Validation mode ${MODE} completed. No training or formal inference was started."
