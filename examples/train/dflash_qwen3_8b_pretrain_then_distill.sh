#!/bin/bash
# DFlash Pretraining, then Distillation -- two-stage recipe
#
# Stage 1 pretrains the draft on raw text with no verifier forward pass: it
# predicts the corpus's own next tokens from the frozen input embedding, so it
# needs no vLLM server, no hidden-state extraction, and no prepared dataset.
# That makes tokens cheap, which is the point -- the draft can see far more of
# them than distillation could afford.
#
# Stage 2 is ordinary DFlash distillation, warm-started from stage 1. No
# conversion step sits between them: pretraining trains only the `fc` columns
# fed by verifier layer 0 (the embedding) and leaves the rest at zero, so the
# checkpoint is a plain DFlash checkpoint that stage 2 loads with
# --from-pretrained and resumes from exactly.
#
# Between them sits `speculators expand-aux-layers`, which widens the pretrained
# `fc` projection to the auxiliary layers distillation will use and leaves the
# new slots at zero. Pretraining itself commits to nothing: it learns from the
# embedding alone, so the same run can be widened several ways to feed a
# layer-selection sweep.
#
# Usage: Copy this script, modify the configuration variables below, then run:
#   bash examples/train/dflash_qwen3_8b_pretrain_then_distill.sh
#
# For a walkthrough see
# https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/pretraining/

set -euo pipefail

# ============ Configuration ============
MODEL="Qwen/Qwen3-8B"
OUTPUT_DIR="./output/dflash_qwen3_8b_pretrain_then_distill"
SEQ_LENGTH=8192

# Stage 1: pretraining. The budget is counted across all ranks and fixes the
# run length, so the epoch ends on its own -- no --max-steps needed. Start
# small (a few billion) to sanity-check throughput before committing to a
# long run.
PRETRAIN_DATASET="HuggingFaceFW/fineweb"
PRETRAIN_DATASET_CONFIG="sample-10BT"
PRETRAIN_TOKEN_BUDGET=1000000000
PRETRAIN_LR=3e-4

# Stage 2: distillation. Same data path as any other DFlash run.
DATASET="hf:inference-optimization/speculators-ci-datasets:tutorial_regen"
MAX_SAMPLES=5000
EPOCHS=5
LR=3e-4
VLLM_PORT=8000

# Shared draft geometry. Both stages must agree on all of it.
BLOCK_SIZE=16
MAX_ANCHORS=3072
NUM_LAYERS=5
TARGET_LAYER_IDS="0 18 33"  # chosen at the widening step; must match vLLM's eagle_aux_hidden_state_layer_ids

VLLM_GPUS="0,1"
TRAIN_GPUS="2,3"
NUM_TRAIN_GPUS=2

PRETRAIN_CKPT="$OUTPUT_DIR/pretrain/checkpoints"
WIDENED_CKPT="$OUTPUT_DIR/pretrain/widened"
DISTILL_CKPT="$OUTPUT_DIR/distill/checkpoints"
# =======================================

# Step 1: Pretrain on raw text. Note what is absent: no prepare-data, no vLLM.
# The verifier contributes weights only -- its embedding and LM head are read
# off the checkpoint on disk and never run.
echo "=== Step 1: Pretraining on $PRETRAIN_DATASET (no vLLM) ==="
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    -m speculators.train \
    --training-mode pretrain \
    --speculator-type dflash \
    --verifier-name-or-path "$MODEL" \
    --pretrain-dataset "$PRETRAIN_DATASET" \
    --pretrain-dataset-config "$PRETRAIN_DATASET_CONFIG" \
    --pretrain-token-budget "$PRETRAIN_TOKEN_BUDGET" \
    --save-path "$PRETRAIN_CKPT" \
    --total-seq-len "$SEQ_LENGTH" \
    --lr "$PRETRAIN_LR" \
    --block-size "$BLOCK_SIZE" \
    --max-anchors "$MAX_ANCHORS" \
    --num-layers "$NUM_LAYERS" \
    --target-layer-ids 0

# Step 2: Widen the projection to the layers distillation will consume. The
# pretrained columns move to layer 0's slot; the rest start at zero, so
# distillation resumes exactly where pretraining stopped.
echo "=== Step 2: Widening the pretrained projection ==="
speculators expand-aux-layers "$PRETRAIN_CKPT/checkpoint_best" $TARGET_LAYER_IDS \
    --output "$WIDENED_CKPT"

# Step 3: Prepare the distillation data.
echo "=== Step 3: Preparing distillation data ==="
speculators prepare-data \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$OUTPUT_DIR/data" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH"

# Step 4: Launch vLLM for online hidden-state extraction.
echo "=== Step 4: Launching vLLM server ==="
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --provenance-dir "$DISTILL_CKPT" \
    -- --data-parallel-size 2 --port "$VLLM_PORT" &
VLLM_PID=$!

cleanup() {
    echo "Stopping vLLM server..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server to be ready..."
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 2
done
echo "vLLM server ready."

# Step 5: Distill, warm-started from the widened checkpoint. Everything here is
# a normal DFlash run except --from-pretrained, which also supplies the
# auxiliary layer ids -- --target-layer-ids would be ignored here.
echo "=== Step 5: Distilling from the widened checkpoint ==="
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    -m speculators.train \
    --speculator-type dflash \
    --verifier-name-or-path "$MODEL" \
    --from-pretrained "$WIDENED_CKPT" \
    --data-path "$OUTPUT_DIR/data" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$DISTILL_CKPT" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --block-size "$BLOCK_SIZE" \
    --max-anchors "$MAX_ANCHORS" \
    --num-layers "$NUM_LAYERS" \
    --on-missing generate \
    --on-generate delete

echo "Done. Pretrained checkpoint:  $PRETRAIN_CKPT"
echo "      Distilled checkpoint:   $DISTILL_CKPT"
echo "Compare against a cold-start run (same flags, no --from-pretrained) to"
echo "measure what the pretraining bought you."
