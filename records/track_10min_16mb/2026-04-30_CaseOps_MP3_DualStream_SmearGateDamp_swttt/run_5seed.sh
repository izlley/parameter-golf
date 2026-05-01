#!/bin/bash
# 5-seed runner for CaseOps + MP3 + SmearGate(alias-prev dampening 0.5x) + swttt.
# Usage:
#   bash run_5seed.sh
# Env overrides:
#   SEEDS="42 0 1234 314 999"   # default
#   DATA_PATH=...                # default ./data/datasets/fineweb10B_sp8192_caseops_marker_pair_v3
#   TOKENIZER_PATH=...           # default ./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
set -o pipefail

SEEDS="${SEEDS:-42 0 1234 314 999}"
DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp8192_caseops_marker_pair_v3}"
TOKENIZER_PATH="${TOKENIZER_PATH:-./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model}"

if [ ! -d "$DATA_PATH" ]; then
  echo "ERROR: dataset dir not found: $DATA_PATH"
  echo "Run prepare_caseops_data.py then prepare_marker_pair_v3.py first."
  exit 1
fi
if [ ! -f "$TOKENIZER_PATH" ]; then
  echo "ERROR: tokenizer model not found: $TOKENIZER_PATH"
  exit 1
fi

ts() { date '+%Y-%m-%d %H:%M:%S'; }
echo "[$(ts)] === 5-seed run START ==="
echo "  SEEDS:          $SEEDS"
echo "  DATA_PATH:      $DATA_PATH"
echo "  TOKENIZER_PATH: $TOKENIZER_PATH"

for SEED in $SEEDS; do
  LOG="train_seed${SEED}.log"
  echo ""
  echo "[$(ts)] >>> seed=$SEED -> $LOG >>>"
  env SEED="$SEED" SKIP_TRAINING=0 \
    MUON_WD=0.085 WARMUP_STEPS=100 \
    MATRIX_CLIP_SIGMAS=13.0 MATRIX_LR=0.026 \
    GPTQ_CALIBRATION_BATCHES=96 \
    TTT_ENABLED=1 TTT_LR=0.0001 \
    TTT_LORA_ALPHA=96 TTT_OUT_LORA_ALPHA=512 \
    TTT_MAX_B_NORM=5.0 TTT_MAX_B_NORM_OUT=8.0 TTT_MAX_B_NORM_EMB=2.0 \
    TTT_LORA_RANK=64 TTT_EMB_LORA_RANK=16 TTT_OUT_LORA_RANK=32 \
    TTT_TRAIN_EVERY_WINDOWS=480 TTT_EPOCHS=1 TTT_BATCH_SEQS=20 \
    TTT_LORA_LAYERS="0,1,2,3,4,5,6,7,8,9,10" \
    TTT_LORA_MODULES="c_qkv,proj,mlp.proj,mlp.fc,final_mlp.fc,final_mlp.proj" \
    TTT_WARM_START_A=0 \
    EVAL_STRIDE=64 \
    MIN_LR=0.10 \
    LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 \
    LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64 \
    ATTN_OUT_GATE=0 \
    SMEAR_GATE=1 SMEAR_GATE_WIDTH=12 \
    SMEAR_TARGET=fast SMEAR_LANE_INIT=clean SMEAR_LR_MULT=2.0 \
    CASEOPS_ENABLED=1 \
    ALIAS_COMPOSITE=0 \
    MARKER_PAIR_MODE=1 \
    MARKER_PAIR_W_SPACE=0.4 \
    MARKER_PAIR_W_TITLE=0.6 \
    MARKER_PAIR_NORM_TARGET=title \
    ALIAS_PREV_SMEAR_SCALE=0.5 \
    ALIAS_CURR_SMEAR_SCALE=1.0 \
    DATA_PATH="$DATA_PATH" \
    TOKENIZER_PATH="$TOKENIZER_PATH" \
    torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "$LOG"
  rc="${PIPESTATUS[0]}"
  echo "[$(ts)] <<< seed=$SEED done (rc=$rc) <<<"
done

echo ""
echo "[$(ts)] === 5-seed run DONE ==="
echo ""
echo "=== Summary ==="
printf "%-8s %-14s %-14s %-12s %s\n" "seed" "swttt_bpb" "val_loss" "size" "step_avg"
echo "------------------------------------------------------------------------"
for SEED in $SEEDS; do
  LOG="train_seed${SEED}.log"
  sw=$(grep -E "final_int6_swttt_exact" "$LOG" 2>/dev/null | grep -oP "val_bpb:\K[0-9.]+" | tail -1)
  vl=$(grep -E "final_int6_swttt_exact" "$LOG" 2>/dev/null | grep -oP "val_loss:\K[0-9.]+" | tail -1)
  sz=$(grep "Total submission size (minified):" "$LOG" 2>/dev/null | grep -oE "[0-9]+" | tail -1)
  st=$(grep -oP "step_avg:[0-9.]+" "$LOG" 2>/dev/null | tail -1)
  printf "%-8s %-14s %-14s %-12s %s\n" "$SEED" "${sw:-?}" "${vl:-?}" "${sz:-?}" "${st:-?}"
done
