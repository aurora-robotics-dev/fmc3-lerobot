#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

single_train_script="scripts/train/train_pi05_agi_arm_tissue_20260512.sh"

DRY_RUN=false
EXTRA_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --dry-run)
      DRY_RUN=true
      EXTRA_ARGS+=("$arg")
      ;;
    *)
      EXTRA_ARGS+=("$arg")
      ;;
  esac
done

run_variant() {
  local task_variant="$1"
  local run_name="$2"
  local task_text="$3"

  echo "=========================================="
  echo "Training PI0.5 LoRA: ${task_variant}"
  echo "Output run name: ${run_name}"
  echo "=========================================="

  TASK_VARIANT="${task_variant}" \
  RUN_NAME="${run_name}" \
  JOB_NAME="${run_name}" \
  TASK_TEXT="${task_text}" \
    "./${single_train_script}" "${EXTRA_ARGS[@]}"
}

run_variant \
  "black_to_yellow" \
  "pi05_o10_right_tissue_black_to_yellow_lora" \
  "use the right arm to move the tissue from the black paper to the yellow paper"
run_variant \
  "yellow_to_black" \
  "pi05_o10_right_tissue_yellow_to_black_lora" \
  "use the right arm to move the tissue from the yellow paper to the black paper"

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "Dry run finished. No training was started."
fi
