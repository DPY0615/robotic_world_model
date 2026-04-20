#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/data0/user/xiejiayi/miniconda3/envs/rwm/bin/python}"
GPUS=(0 1)
MAX_JOBS=${#GPUS[@]}
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/offline/launch_logs"
mkdir -p "$LOG_DIR"

TASKS=(
  "lite3_flat_wm_safe"
  "lite3_flat_ftbest_ref"
  "lite3_flat_ftbest_ref_u03"
  "lite3_flat_ftbest_track"
  "lite3_flat_ftbest_stable"
  "lite3_flat_ftbest_recover"
  "lite3_flat_ftbest_track_aggr"
  "lite3_flat_ftbest_track_aggr_u05"
  "lite3_flat_ftbest_track_aggr_smooth"
  "lite3_flat_ftbest_track_aggr_gait"
)

RUN_NUM_BASE=9500

launch_one() {
  local idx="$1"
  local gpu="$2"
  local task="${TASKS[$idx]}"
  local run_num=$((RUN_NUM_BASE + idx + 1))
  local log_file="$LOG_DIR/${task}_${TIMESTAMP}.log"

  echo "[LAUNCH] task=$task gpu=$gpu run_num=$run_num log=$log_file"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" scripts/reinforcement_learning/model_based/train.py \
    --task "$task" \
    --run_num "$run_num" \
    --max_iterations_override 500 \
    --sim_ref_steps_override 1200 \
    --sim_ref_num_envs_override 32 \
    > "$log_file" 2>&1 &
}

for i in "${!TASKS[@]}"; do
  while [[ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]]; do
    wait -n
  done
  gpu="${GPUS[$(( i % MAX_JOBS ))]}"
  launch_one "$i" "$gpu"
done

wait

echo "[DONE] all ${#TASKS[@]} offline runs finished (500 iterations each)."
