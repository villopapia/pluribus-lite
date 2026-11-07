#!/usr/bin/env bash
# SLURM stub for blueprint training (M3): 96 h jobs chained by checkpoint.
#
# Usage:   sbatch scripts/slurm_train.sh <config.cfg> <run_dir> [total_hours]
#
# How the chain works:
#   * SLURM sends SIGUSR1 to this batch script 15 min before the 96 h limit
#     (--signal=B:USR1@900). The trap creates <run_dir>/STOP.
#   * train_blueprint polls for STOP, writes a final checkpoint and exits 0.
#   * If <run_dir>/DONE does not exist and the cumulative training time is below
#     total_hours, this script resubmits itself with a dependency on the current job.
#     The new job resumes from the newest checkpoint in <run_dir> (train_blueprint
#     always resumes when a checkpoint is present).
#
# STUB: train_blueprint is the M3 trainer (engine/nlhe/src/bin/train_blueprint.rs).
# Adjust partition/account/modules for the cluster. Nothing here is Windows-specific.

#SBATCH --job-name=pluribus-lite-blueprint
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --signal=B:USR1@900
#SBATCH --output=slurm-%j.out
#SBATCH --requeue

set -euo pipefail

CONFIG=${1:?config path}
RUN_DIR=${2:?run directory}
TOTAL_HOURS=${3:-192}
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
BIN="$REPO/engine/target/release/train_blueprint"

mkdir -p "$RUN_DIR"
rm -f "$RUN_DIR/STOP"

if [ ! -x "$BIN" ]; then
  (cd "$REPO/engine" && cargo build --release --bin train_blueprint)
fi

on_usr1() {
  echo "$(date -Is) USR1: time limit approaching, requesting checkpoint + stop"
  touch "$RUN_DIR/STOP"
}
trap on_usr1 USR1

THREADS=${SLURM_CPUS_PER_TASK:-$(nproc)}
START=$(date +%s)

# Run in the background and `wait`, so the USR1 trap fires while the trainer runs.
"$BIN" --config "$CONFIG" --run-dir "$RUN_DIR" --threads "$THREADS" \
       --stop-file "$RUN_DIR/STOP" --checkpoint-minutes 60 &
PID=$!
wait $PID || wait $PID   # a trapped signal interrupts the first wait
STATUS=$?

ELAPSED_H=$(( ($(date +%s) - START) / 3600 ))
PREV_H=$(cat "$RUN_DIR/hours_done" 2>/dev/null || echo 0)
echo $(( PREV_H + ELAPSED_H )) > "$RUN_DIR/hours_done"

if [ $STATUS -ne 0 ]; then
  echo "trainer exited with status $STATUS; not resubmitting" >&2
  exit $STATUS
fi
if [ -f "$RUN_DIR/DONE" ] || [ "$(cat "$RUN_DIR/hours_done")" -ge "$TOTAL_HOURS" ]; then
  echo "training complete ($(cat "$RUN_DIR/hours_done") h)"
  exit 0
fi
echo "resubmitting to continue from checkpoint"
sbatch --dependency=afterany:"$SLURM_JOB_ID" "$0" "$CONFIG" "$RUN_DIR" "$TOTAL_HOURS"
