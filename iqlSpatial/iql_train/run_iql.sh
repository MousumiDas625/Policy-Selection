#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  run_iql.sh — IQL pipeline for pi0 vs pi0.5 policy selection.
#
#  Held-out tasks (5, 9) are NEVER used here — reserved for online eval.
#  Train pool = 8 tasks (0,1,2,3,4,6,7,8) by default.
#
#  Usage:
#    bash run_iql.sh expt_pipi_test                                   # natural, no balancing
#    bash run_iql.sh expt_pipi_balanced "" "" --balance               # balanced (25/25 per VLA-task)
#    bash run_iql.sh my_expt "0,1,2,3,4,6,7,8" 0.2 --balance          # explicit
# ═══════════════════════════════════════════════════════════════════
set -e

EXPT_NAME="${1:-expt_pipi_test}"
TRAIN_TASKS="${2:-0,1,2,3,4,6,7,8}"
VAL_SPLIT="${3:-0.2}"
BALANCE_FLAG="${4:-}"      # pass "--balance" to enable; empty otherwise
EXTRA_ARGS="${@:5}"        # forwarded to train_iql.py (e.g. --freeze-policy-emb)

# Force defaults if the caller passed empty strings explicitly
[ -z "$TRAIN_TASKS" ] && TRAIN_TASKS="0,1,2,3,4,6,7,8"
[ -z "$VAL_SPLIT"   ] && VAL_SPLIT="0.2"

ROOT="/project2/jessetho_1732/mousumid/PolicySel/iqlSpatial"
IQL_ROOT="$ROOT/iql_train"
EXPT="$IQL_ROOT/$EXPT_NAME"
HF_CACHE="/project2/jessetho_1732/mousumid/PolicySel/hf_cache"
GPU=0
POLICIES="pi0,pi05"
HELDOUT="5,9"

echo "═══════════════════════════════════════════════════════"
echo "  IQL Experiment: $EXPT_NAME"
echo "  Train tasks:    $TRAIN_TASKS"
echo "  Val split:      $VAL_SPLIT"
echo "  Held-out:       $HELDOUT  (online eval only)"
echo "  Policies:       $POLICIES"
echo "  Balance flag:   ${BALANCE_FLAG:-OFF}"
echo "  Extra args:     $EXTRA_ARGS"
echo "═══════════════════════════════════════════════════════"

# Env
source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate qwen-eval
export PYTHONNOUSERSITE=1
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"

# Stage scripts into the experiment folder
mkdir -p "$EXPT"
for f in config.py build_chunks.py embed_qwen.py train_iql.py eval_online.py; do
    cp "$IQL_ROOT/$f" "$EXPT/$f" 2>/dev/null || cp "$(dirname $0)/$f" "$EXPT/$f"
done

cd "$EXPT"

# ── Step 1 ───────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo "  [1/3] build_chunks.py"
echo "════════════════════════════════════════════════════"
python build_chunks.py \
    --data-root "$ROOT" \
    --policies "$POLICIES" \
    --train-tasks "$TRAIN_TASKS" \
    --heldout-tasks "$HELDOUT" \
    --val-split "$VAL_SPLIT" \
    --balance-target 25 \
    --seed 42 \
    --expt-dir "$EXPT_NAME" \
    $BALANCE_FLAG

DATA="$EXPT/data"
if [ ! -f "$DATA/train_chunks.pkl" ]; then
    echo "ERROR: $DATA/train_chunks.pkl not found, aborting."
    exit 1
fi

# ── Step 2 ───────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo "  [2/3] embed_qwen.py  (slowest step)"
echo "════════════════════════════════════════════════════"
CUDA_VISIBLE_DEVICES=$GPU python embed_qwen.py \
    --expt-dir "$EXPT" \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --batch-size 4 \
    --cache-dir "$HF_CACHE"

# ── Step 3 ───────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo "  [3/3] train_iql.py"
echo "════════════════════════════════════════════════════"
CUDA_VISIBLE_DEVICES=$GPU python train_iql.py \
    --expt-dir "$EXPT" \
    --policies "$POLICIES" \
    --epochs 200 \
    --lr 3e-4 \
    --gamma 0.99 \
    --tau 0.7 \
    --batch-size 256 \
    --eval-every 10 \
    $EXTRA_ARGS

echo ""
echo "═══════════════════════════════════════════════════════"
echo "Done."
echo "  expt dir   : $EXPT"
echo "  audit      : $EXPT/audit.json"
echo "  best ckpt  : $EXPT/runs/*/best.pt"
echo "  log        : $EXPT/runs/*/log.csv"
echo ""
echo "Next: online eval on task 5 — see TERMINAL_COMMANDS.md."
echo "═══════════════════════════════════════════════════════"
