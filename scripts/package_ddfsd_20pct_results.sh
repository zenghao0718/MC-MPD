#!/usr/bin/env bash
# Package the DDFSD 20% GenImage analysis artifacts without modifying the
# experiment outputs. Result files are copied into a temporary staging tree,
# metadata is generated there, and only that tree is archived.
#
# Optional environment overrides:
#   RUN_ROOT                 Parent run directory.
#   RUN_CONFIG               Experiment directory name.
#   RESULTS_ROOT             Complete experiment result directory. This takes
#                            precedence over RUN_ROOT/RUN_CONFIG.
#   PACKAGE_OUTPUT_DIR       Directory that receives the final tar.gz.
#   PACKAGE_TIMESTAMP        Timestamp used in the default archive name.
#   PACKAGE_NAME             Final archive basename (must end in .tar.gz).
#   PACKAGE_PATH             Complete final archive path. This takes precedence
#                            over PACKAGE_OUTPUT_DIR/PACKAGE_NAME.
#   EXCLUDE_CLASSES          Comma- or whitespace-separated class list.
#   CKPT_STEPS               Comma- or whitespace-separated checkpoint steps.
#   ALPHA_GRID_STEPS         Fixed-alpha checkpoint steps.
#   STAGING_PARENT           Parent directory used by mktemp.
#   CONFIG_SNAPSHOT_FILES    Comma- or whitespace-separated repository-relative
#                            configuration files to copy into the package.
set -euo pipefail

DEFAULT_RUN_ROOT="/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1"
DEFAULT_RUN_CONFIG="ddfsd_20pct_steps30000"
DEFAULT_CLASSES="ADM,BigGAN,glide,Midjourney,SD,VQDM"
DEFAULT_CKPT_STEPS="5000,10000,15000,20000,25000,30000"
DEFAULT_ALPHA_GRID_STEPS="15000,25000,30000"

RUN_ROOT=${RUN_ROOT:-"${DEFAULT_RUN_ROOT}"}
RUN_CONFIG=${RUN_CONFIG:-"${DEFAULT_RUN_CONFIG}"}

fatal() {
    echo "ERROR: $*" >&2
    exit 1
}

[[ "${RUN_CONFIG}" =~ ^[A-Za-z0-9._-]+$ ]] || fatal "RUN_CONFIG must be a safe basename using only A-Z, a-z, 0-9, dot, underscore, or hyphen: ${RUN_CONFIG}"
[[ "${RUN_CONFIG}" != "." && "${RUN_CONFIG}" != ".." ]] || fatal "RUN_CONFIG must not be . or .."

RAW_RESULTS_ROOT=${RESULTS_ROOT:-${EXPERIMENT_ROOT:-"${RUN_ROOT}/${RUN_CONFIG}"}}
[[ -n "${RAW_RESULTS_ROOT//[[:space:]]/}" ]] || fatal "RESULTS_ROOT must not be empty."
if [[ -d "${RAW_RESULTS_ROOT}" ]]; then
    RESULTS_ROOT=$(cd -- "${RAW_RESULTS_ROOT}" && pwd -P)
elif [[ -e "${RAW_RESULTS_ROOT}" || -L "${RAW_RESULTS_ROOT}" ]]; then
    fatal "RESULTS_ROOT exists but is not a directory: ${RAW_RESULTS_ROOT}"
elif command -v realpath >/dev/null 2>&1; then
    RESULTS_ROOT=$(realpath -m -- "${RAW_RESULTS_ROOT}")
else
    # Fallback for minimal systems without realpath: safely resolve an existing
    # physical parent, or stop rather than guessing through missing parents.
    RESULTS_PARENT=$(dirname -- "${RAW_RESULTS_ROOT}")
    RESULTS_BASENAME=$(basename -- "${RAW_RESULTS_ROOT}")
    [[ -d "${RESULTS_PARENT}" ]] || fatal "Cannot safely normalize missing RESULTS_ROOT without realpath: ${RAW_RESULTS_ROOT}"
    RESULTS_PARENT=$(cd -- "${RESULTS_PARENT}" && pwd -P)
    RESULTS_ROOT="${RESULTS_PARENT%/}/${RESULTS_BASENAME}"
fi
RESULTS_ROOT=${RESULTS_ROOT%/}
[[ -n "${RESULTS_ROOT}" && "${RESULTS_ROOT}" != "/" ]] || fatal "Normalized RESULTS_ROOT must not be empty or /."

PACKAGE_OUTPUT_DIR=${PACKAGE_OUTPUT_DIR:-"${RESULTS_ROOT}"}
PACKAGE_TIMESTAMP=${PACKAGE_TIMESTAMP:-${TIMESTAMP:-"$(date +%Y%m%d_%H%M%S)"}}
if [[ -n "${PACKAGE_PATH+x}" ]]; then
    [[ -n "${PACKAGE_PATH}" ]] || fatal "PACKAGE_PATH must not be empty when explicitly set."
    PACKAGE_OUTPUT_DIR=$(dirname -- "${PACKAGE_PATH}")
    PACKAGE_NAME=$(basename -- "${PACKAGE_PATH}")
else
    PACKAGE_NAME=${PACKAGE_NAME:-"${RUN_CONFIG}_analysis_package_${PACKAGE_TIMESTAMP}.tar.gz"}
    [[ "${PACKAGE_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] || fatal "PACKAGE_NAME must be a safe basename: ${PACKAGE_NAME}"
    [[ "${PACKAGE_NAME}" != "." && "${PACKAGE_NAME}" != ".." ]] || fatal "PACKAGE_NAME must not be . or .."
    PACKAGE_PATH="${PACKAGE_OUTPUT_DIR%/}/${PACKAGE_NAME}"
fi

EXCLUDE_CLASSES=${EXCLUDE_CLASSES:-${CLASSES:-"${DEFAULT_CLASSES}"}}
CKPT_STEPS=${CKPT_STEPS:-"${DEFAULT_CKPT_STEPS}"}
ALPHA_GRID_STEPS=${ALPHA_GRID_STEPS:-"${DEFAULT_ALPHA_GRID_STEPS}"}
STAGING_PARENT=${STAGING_PARENT:-${TMPDIR:-/tmp}}

DEFAULT_CONFIG_SNAPSHOT_FILES="scripts/train_ddfsd_20pct.sh scripts/run_ddfsd_one_class_pipeline_20pct.sh scripts/run_ddfsd_all_classes_pipeline_20pct.sh scripts/eval_ddfsd_formal_all_steps.sh scripts/eval_ddfsd_branch_modes.sh scripts/eval_ddfsd_alpha_grid.sh tools/summarize_ddfsd_20pct.py tools/parse_ddfsd_train_alpha_loss.py scripts/package_ddfsd_20pct_results.sh"
CONFIG_SNAPSHOT_FILES=${CONFIG_SNAPSHOT_FILES:-"${DEFAULT_CONFIG_SNAPSHOT_FILES}"}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[[ -n "${PACKAGE_OUTPUT_DIR}" ]] || fatal "PACKAGE_OUTPUT_DIR must not be empty."
[[ "${PACKAGE_NAME}" == *.tar.gz ]] || fatal "PACKAGE_NAME must end in .tar.gz: ${PACKAGE_NAME}"

CLASS_LIST_RAW=${EXCLUDE_CLASSES//,/ }
STEP_LIST_RAW=${CKPT_STEPS//,/ }
ALPHA_STEP_LIST_RAW=${ALPHA_GRID_STEPS//,/ }
CONFIG_LIST_RAW=${CONFIG_SNAPSHOT_FILES//,/ }
CLASS_LIST=()
STEP_LIST=()
ALPHA_STEP_LIST=()
CONFIG_LIST=()
read -r -a CLASS_LIST <<< "${CLASS_LIST_RAW}" || true
read -r -a STEP_LIST <<< "${STEP_LIST_RAW}" || true
read -r -a ALPHA_STEP_LIST <<< "${ALPHA_STEP_LIST_RAW}" || true
read -r -a CONFIG_LIST <<< "${CONFIG_LIST_RAW}" || true

[[ ${#CLASS_LIST[@]} -gt 0 ]] || fatal "EXCLUDE_CLASSES resolved to an empty list."
[[ ${#STEP_LIST[@]} -gt 0 ]] || fatal "CKPT_STEPS resolved to an empty list."
[[ ${#ALPHA_STEP_LIST[@]} -gt 0 ]] || fatal "ALPHA_GRID_STEPS resolved to an empty list."
for CLASS_NAME in "${CLASS_LIST[@]}"; do
    case "${CLASS_NAME}" in
        ADM|BigGAN|glide|Midjourney|SD|VQDM) ;;
        *) fatal "Invalid exclude class: ${CLASS_NAME}" ;;
    esac
done
for STEP in "${STEP_LIST[@]}" "${ALPHA_STEP_LIST[@]}"; do
    [[ "${STEP}" =~ ^[0-9]+$ ]] || fatal "Invalid checkpoint step: ${STEP}"
done

mkdir -p -- "${STAGING_PARENT}"
STAGING_PARENT=$(cd "${STAGING_PARENT}" && pwd -P)
STAGING_PREFIX="${STAGING_PARENT%/}/ddfsd_20pct_package."
STAGING_DIR=$(mktemp -d "${STAGING_PREFIX}XXXXXX")
STAGING_DIR=$(cd "${STAGING_DIR}" && pwd -P)
STAGING_MARKER="${STAGING_DIR}/.ddfsd_package_staging"
touch -- "${STAGING_MARKER}"

cleanup_staging() {
    # The marker and mktemp-only path guard ensure this can never target the
    # source experiment directory, even when STAGING_PARENT is overridden.
    if [[ -n "${STAGING_DIR:-}" && -f "${STAGING_MARKER:-}" && "${STAGING_DIR}" == "${STAGING_PREFIX}"* ]]; then
        rm -rf -- "${STAGING_DIR}"
    fi
}
trap cleanup_staging EXIT

PAYLOAD_BASENAME="${RUN_CONFIG}_analysis"
PAYLOAD_ROOT_LEXICAL="${STAGING_DIR}/${PAYLOAD_BASENAME}"
case "${PAYLOAD_ROOT_LEXICAL}" in
    "${STAGING_DIR}/"*) ;;
    *) fatal "Payload path escapes staging lexically: ${PAYLOAD_ROOT_LEXICAL}" ;;
esac
mkdir -- "${PAYLOAD_ROOT_LEXICAL}"
PAYLOAD_ROOT=$(cd "${PAYLOAD_ROOT_LEXICAL}" && pwd -P)
case "${PAYLOAD_ROOT}" in
    "${STAGING_DIR}/"*) ;;
    *) fatal "Payload path escapes staging physically: ${PAYLOAD_ROOT}" ;;
esac
STAGED_RESULTS="${PAYLOAD_ROOT}/results"
STAGED_CONFIG="${PAYLOAD_ROOT}/configuration"
mkdir -p -- "${STAGED_RESULTS}" "${STAGED_CONFIG}"

MISSING_TMP="${STAGING_DIR}/missing_files.unsorted"
COPIED_TMP="${STAGING_DIR}/copied_results.unsorted"
: > "${MISSING_TMP}"
: > "${COPIED_TMP}"

record_missing() {
    printf '%s\n' "$1" >> "${MISSING_TMP}"
}

require_result_file() {
    local RELATIVE_PATH=$1
    if [[ ! -f "${RESULTS_ROOT}/${RELATIVE_PATH}" ]]; then
        record_missing "${RELATIVE_PATH}"
    fi
}

csv_quote() {
    local VALUE=${1//\"/\"\"}
    printf '"%s"' "${VALUE}"
}

CHECKPOINT_MANIFEST="${PAYLOAD_ROOT}/checkpoint_manifest.csv"
printf 'exclude_class,step,file_name,relative_path,size_bytes,exists\n' > "${CHECKPOINT_MANIFEST}"
for CLASS_NAME in "${CLASS_LIST[@]}"; do
    for STEP in "${STEP_LIST[@]}"; do
        FILE_NAME="ddfsd_step[${STEP}].pth"
        RELATIVE_PATH="exclude_${CLASS_NAME}/ckpt/${FILE_NAME}"
        CHECKPOINT_PATH="${RESULTS_ROOT}/${RELATIVE_PATH}"
        SIZE_BYTES="NA"
        EXISTS="false"
        if [[ -f "${CHECKPOINT_PATH}" ]]; then
            EXISTS="true"
            SIZE_BYTES=$(stat -c '%s' -- "${CHECKPOINT_PATH}" 2>/dev/null || true)
            SIZE_BYTES=${SIZE_BYTES:-NA}
        else
            record_missing "${RELATIVE_PATH}"
        fi
        {
            csv_quote "${CLASS_NAME}"
            printf ','
            csv_quote "${STEP}"
            printf ','
            csv_quote "${FILE_NAME}"
            printf ','
            csv_quote "${RELATIVE_PATH}"
            printf ','
            csv_quote "${SIZE_BYTES}"
            printf ','
            csv_quote "${EXISTS}"
            printf '\n'
        } >> "${CHECKPOINT_MANIFEST}"
    done
done

# Check the required result categories. Missing artifacts are reported but are
# deliberately not fatal, so a partial experiment remains packageable.
require_result_file "PIPELINE_STATUS_ALL.txt"
require_result_file "PIPELINE_CONFIG_ALL.txt"
require_result_file "summary/ddfsd_20pct_all_classes_summary.md"
require_result_file "summary/ddfsd_20pct_all_classes_summary.csv"
require_result_file "summary/ddfsd_20pct_all_classes_completeness.csv"
require_result_file "summary/ddfsd_20pct_missing_files_and_failures.csv"

for CLASS_NAME in "${CLASS_LIST[@]}"; do
    CLASS_ROOT="exclude_${CLASS_NAME}"
    require_result_file "${CLASS_ROOT}/PIPELINE_STATUS.txt"
    require_result_file "${CLASS_ROOT}/PIPELINE_CONFIG.txt"
    require_result_file "${CLASS_ROOT}/logs/train_${CLASS_NAME}_20pct.log"
    require_result_file "${CLASS_ROOT}/logs/parse_${CLASS_NAME}_train_alpha_loss.log"

    require_result_file "${CLASS_ROOT}/formal_eval/ddfsd_eval_per_seed_all_steps.csv"
    require_result_file "${CLASS_ROOT}/formal_eval/ddfsd_eval_summary_all_steps.csv"
    for STEP in "${STEP_LIST[@]}"; do
        require_result_file "${CLASS_ROOT}/formal_eval/step_${STEP}/ddfsd_eval_per_seed.csv"
        require_result_file "${CLASS_ROOT}/formal_eval/step_${STEP}/ddfsd_eval_summary.csv"
        require_result_file "${CLASS_ROOT}/formal_eval/logs/eval_${CLASS_NAME}_step_${STEP}.log"
    done

    require_result_file "${CLASS_ROOT}/branch_modes/ddfsd_${CLASS_NAME}_branch_modes_per_seed.csv"
    require_result_file "${CLASS_ROOT}/branch_modes/ddfsd_${CLASS_NAME}_branch_modes_summary.csv"
    for STEP in "${STEP_LIST[@]}"; do
        require_result_file "${CLASS_ROOT}/branch_modes/step_${STEP}/ddfsd_${CLASS_NAME}_branch_modes_per_seed.csv"
        require_result_file "${CLASS_ROOT}/branch_modes/step_${STEP}/ddfsd_${CLASS_NAME}_branch_modes_summary.csv"
        require_result_file "${CLASS_ROOT}/branch_modes/logs/eval_${CLASS_NAME}_step_${STEP}.log"
    done

    require_result_file "${CLASS_ROOT}/alpha_grid/ddfsd_${CLASS_NAME}_alpha_grid_per_seed.csv"
    require_result_file "${CLASS_ROOT}/alpha_grid/ddfsd_${CLASS_NAME}_alpha_grid_summary.csv"
    for STEP in "${ALPHA_STEP_LIST[@]}"; do
        require_result_file "${CLASS_ROOT}/alpha_grid/step_${STEP}/ddfsd_${CLASS_NAME}_alpha_grid_per_seed.csv"
        require_result_file "${CLASS_ROOT}/alpha_grid/step_${STEP}/ddfsd_${CLASS_NAME}_alpha_grid_summary.csv"
        require_result_file "${CLASS_ROOT}/alpha_grid/logs/eval_${CLASS_NAME}_step_${STEP}.log"
    done

    require_result_file "${CLASS_ROOT}/csv/ddfsd_${CLASS_NAME}_train_alpha_loss_by_ckpt.csv"
    require_result_file "${CLASS_ROOT}/summary/ddfsd_${CLASS_NAME}_20pct_summary.md"
    require_result_file "${CLASS_ROOT}/summary/ddfsd_${CLASS_NAME}_20pct_summary.csv"
    require_result_file "${CLASS_ROOT}/summary/ddfsd_${CLASS_NAME}_20pct_completeness.csv"
done

is_excluded_artifact() {
    local LOWER_NAME=${1,,}
    case "${LOWER_NAME}" in
        *.pth|*.tar|*.tar.gz|*.tgz|*.zip|*.gz|*.bz2|*.xz|*.zst|*.7z|*.rar)
            return 0
            ;;
    esac
    return 1
}

is_analysis_artifact() {
    local LOWER_NAME=${1,,}
    case "${LOWER_NAME}" in
        *.csv|*.tsv|*.md|*.log|*.txt|*.json|*.yaml|*.yml|*.toml|*.ini|*.cfg|*.conf|*.args|*.out|*.env)
            return 0
            ;;
        pipeline_status*|*config*|*parameter*|*params*|*arguments*|*command*|*metadata*|*integrity*|*completeness*|*summary*)
            return 0
            ;;
    esac
    return 1
}

COPIED_RESULT_COUNT=0
if [[ -d "${RESULTS_ROOT}" ]]; then
    RESULT_FILE_LIST="${STAGING_DIR}/result_files.nul"
    if ! find "${RESULTS_ROOT}" \
        \( -type d -path "${STAGING_DIR}" -prune \) -o \
        -type f -print0 > "${RESULT_FILE_LIST}"; then
        record_missing "result_scan_incomplete:${RESULTS_ROOT}"
    fi
    while IFS= read -r -d '' SOURCE_FILE; do
        RELATIVE_PATH=${SOURCE_FILE#"${RESULTS_ROOT}/"}
        FILE_NAME=${SOURCE_FILE##*/}
        if is_excluded_artifact "${FILE_NAME}"; then
            continue
        fi
        if ! is_analysis_artifact "${FILE_NAME}"; then
            continue
        fi

        DESTINATION="${STAGED_RESULTS}/${RELATIVE_PATH}"
        mkdir -p -- "$(dirname -- "${DESTINATION}")"
        cp -p -- "${SOURCE_FILE}" "${DESTINATION}"
        printf '%s\n' "${RELATIVE_PATH}" >> "${COPIED_TMP}"
        COPIED_RESULT_COUNT=$((COPIED_RESULT_COUNT + 1))
    done < "${RESULT_FILE_LIST}"
fi

COPIED_CONFIG_COUNT=0
for RELATIVE_PATH in "${CONFIG_LIST[@]}"; do
    if [[ "${RELATIVE_PATH}" == /* || "${RELATIVE_PATH}" == ".." || "${RELATIVE_PATH}" == ../* || "${RELATIVE_PATH}" == */../* || "${RELATIVE_PATH}" == */.. ]]; then
        record_missing "repository_config:${RELATIVE_PATH} (invalid repository-relative path)"
        continue
    fi
    SOURCE_FILE="${REPO_ROOT}/${RELATIVE_PATH}"
    if [[ ! -f "${SOURCE_FILE}" ]]; then
        record_missing "repository_config:${RELATIVE_PATH}"
        continue
    fi
    DESTINATION="${STAGED_CONFIG}/${RELATIVE_PATH}"
    mkdir -p -- "$(dirname -- "${DESTINATION}")"
    cp -p -- "${SOURCE_FILE}" "${DESTINATION}"
    COPIED_CONFIG_COUNT=$((COPIED_CONFIG_COUNT + 1))
done

GIT_INFO="${PAYLOAD_ROOT}/GIT_INFO.txt"
{
    echo "generated_at=$(date -Iseconds)"
    echo "repository_root=${REPO_ROOT}"
    if git -C "${REPO_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "branch=$(git -C "${REPO_ROOT}" branch --show-current 2>/dev/null || true)"
        echo "commit=$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || true)"
        echo "commit_summary=$(git -C "${REPO_ROOT}" log -1 --oneline 2>/dev/null || true)"
        echo
        echo "[git status --short --branch]"
        git -C "${REPO_ROOT}" status --short --branch 2>/dev/null || true
        echo
        echo "[git remote -v]"
        git -C "${REPO_ROOT}" remote -v 2>/dev/null || true
    else
        echo "git_status=repository_not_available"
    fi
} > "${GIT_INFO}"

sort -u -- "${MISSING_TMP}" > "${STAGING_DIR}/missing_files.sorted"
MISSING_COUNT=$(wc -l < "${STAGING_DIR}/missing_files.sorted")
MISSING_COUNT=${MISSING_COUNT//[[:space:]]/}
MISSING_FILES="${PAYLOAD_ROOT}/missing_files.txt"
{
    echo "# DDFSD 20% expected artifacts missing at package time"
    echo "generated_at=$(date -Iseconds)"
    echo "results_root=${RESULTS_ROOT}"
    echo "missing_count=${MISSING_COUNT}"
    echo
    if [[ "${MISSING_COUNT}" -eq 0 ]]; then
        echo "NONE"
    else
        cat "${STAGING_DIR}/missing_files.sorted"
    fi
} > "${MISSING_FILES}"

PACKAGE_CONFIGURATION="${PAYLOAD_ROOT}/PACKAGE_CONFIGURATION.txt"
{
    echo "generated_at=$(date -Iseconds)"
    echo "run_root=${RUN_ROOT}"
    echo "run_config=${RUN_CONFIG}"
    echo "results_root=${RESULTS_ROOT}"
    echo "results_root_exists=$([[ -d "${RESULTS_ROOT}" ]] && echo true || echo false)"
    echo "package_path=${PACKAGE_PATH}"
    echo "exclude_classes=${EXCLUDE_CLASSES}"
    echo "checkpoint_steps=${CKPT_STEPS}"
    echo "alpha_grid_steps=${ALPHA_GRID_STEPS}"
    echo "copied_result_files=${COPIED_RESULT_COUNT}"
    echo "copied_repository_config_files=${COPIED_CONFIG_COUNT}"
    echo "missing_expected_artifacts=${MISSING_COUNT}"
    echo "copy_policy=analysis text/csv/log/config artifacts only"
    echo "excluded_artifacts=.pth checkpoints and existing compressed archives"
} > "${PACKAGE_CONFIGURATION}"

# Build a deterministic inventory of the staged payload itself. This file is
# generated after all other package metadata and includes its own path.
(
    cd "${PAYLOAD_ROOT}"
    find . -type f -print
    echo "./packaged_files.txt"
) | LC_ALL=C sort -u > "${STAGING_DIR}/packaged_files.sorted"
cp -- "${STAGING_DIR}/packaged_files.sorted" "${PAYLOAD_ROOT}/packaged_files.txt"

# Defense in depth: the staging tree and the completed archive are both checked
# before the archive is copied to its final location.
FORBIDDEN_STAGED=$(find "${PAYLOAD_ROOT}" -type f \
    \( -iname '*.pth' -o -iname '*.tar' -o -iname '*.tar.gz' -o -iname '*.tgz' \
       -o -iname '*.zip' -o -iname '*.gz' -o -iname '*.bz2' -o -iname '*.xz' \
       -o -iname '*.zst' -o -iname '*.7z' -o -iname '*.rar' \) \
    -print -quit)
[[ -z "${FORBIDDEN_STAGED}" ]] || fatal "Forbidden artifact entered staging: ${FORBIDDEN_STAGED}"

TEMP_ARCHIVE="${STAGING_DIR}/${PACKAGE_NAME}"
tar -C "${STAGING_DIR}" -czf "${TEMP_ARCHIVE}" -- "${PAYLOAD_BASENAME}"
tar -tzf "${TEMP_ARCHIVE}" > "${STAGING_DIR}/archive_contents.txt"
if grep -Eiq '\.(pth|tar|tar\.gz|tgz|zip|gz|bz2|xz|zst|7z|rar)/?$' "${STAGING_DIR}/archive_contents.txt"; then
    fatal "Archive validation found a forbidden checkpoint or compressed archive."
fi

mkdir -p -- "${PACKAGE_OUTPUT_DIR}"
[[ ! -e "${PACKAGE_PATH}" ]] || fatal "Refusing to overwrite existing package: ${PACKAGE_PATH}"
(
    set -o noclobber
    cat -- "${TEMP_ARCHIVE}" > "${PACKAGE_PATH}"
) || fatal "Could not copy package without overwriting: ${PACKAGE_PATH}"

PACKAGE_SIZE=$(stat -c '%s' -- "${PACKAGE_PATH}" 2>/dev/null || true)
echo "Package created: ${PACKAGE_PATH}"
echo "Package size (bytes): ${PACKAGE_SIZE:-unknown}"
echo "Copied result files: ${COPIED_RESULT_COUNT}"
echo "Missing expected artifacts: ${MISSING_COUNT} (see missing_files.txt in the package)"
echo "Checkpoint models and existing compressed archives were excluded."
