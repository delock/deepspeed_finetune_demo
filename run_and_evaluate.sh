#!/bin/bash
# CodeAlpaca + Muon: baseline eval -> finetune -> convert -> eval
# Usage: bash finetune_codealpaca_and_evaluate.sh [model_name] [ds_config] [eval_steps] [wandb_name]
# Example: bash finetune_codealpaca_and_evaluate.sh moonshotai/Moonlight-16B-A3B z2_moonlight_autoep_muon.json 100 my_run
#          bash finetune_codealpaca_and_evaluate.sh Qwen/Qwen2.5-0.5B z2_config.json 0
set -euo pipefail

MODEL=${1:-moonshotai/Moonlight-16B-A3B}
DS_CONFIG=${2:-z2_moonlight_autoep_muon.json}
EVAL_STEPS=${3:-100}
WANDB_NAME=${4:-}
# Derive a safe directory name from model (replace / with _)
MODEL_SLUG=$(echo "$MODEL" | tr '/' '_')
CONFIG_SLUG=$(basename "$DS_CONFIG" .json)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=${PYTHON:-$(which python3)}
WORKDIR=$(cd "$(dirname "$0")" && pwd)
LOGDIR=$WORKDIR/experiment_logs
OUTPUT_DIR=$WORKDIR/output_codealpaca_${MODEL_SLUG}_${CONFIG_SLUG}
HF_DIR=$WORKDIR/hf_model_codealpaca_${MODEL_SLUG}_${CONFIG_SLUG}
EVAL_DIR=$WORKDIR/evalplus_results/codealpaca_${MODEL_SLUG}_${CONFIG_SLUG}
BASELINE_DIR=$WORKDIR/evalplus_results/baseline_${MODEL_SLUG}

mkdir -p $LOGDIR $BASELINE_DIR $EVAL_DIR

cd $WORKDIR

echo "===== STEP 0: BASELINE EVALUATION (pre-finetune) ====="
echo "Model:      ${MODEL}"
echo "Config:     ${DS_CONFIG}"
echo "Eval steps: ${EVAL_STEPS}"
echo "W&B name:   ${WANDB_NAME:-<disabled>}"
echo "Start: $(date)"
$PYTHON evaluate/humaneval/gen_humaneval.py \
  --model $MODEL \
  --output $BASELINE_DIR \
  --instruction \
  2>&1 | tee $LOGDIR/baseline_${MODEL_SLUG}_gen.log
$PYTHON -m evalplus.evaluate \
  --dataset humaneval \
  --samples $BASELINE_DIR/samples.jsonl \
  2>&1 | tee $LOGDIR/baseline_${MODEL_SLUG}_eval.log
echo "Baseline eval done: $(date)"

echo "===== STEP 1: TRAINING ====="
echo "Start: $(date)"
deepspeed --num_gpus=8 finetune_llama.py \
  --model_name $MODEL \
  --output_dir $OUTPUT_DIR \
  --batch_size 16 --max_length 512 \
  --deepspeed_config $DS_CONFIG \
  --dataset_name sahil2801/CodeAlpaca-20k \
  --num_train_epochs 1 \
  --eval_steps $EVAL_STEPS \
  ${WANDB_NAME:+--wandb_name "$WANDB_NAME"} \
  2>&1 | tee $LOGDIR/codealpaca_${MODEL_SLUG}_${CONFIG_SLUG}_train.log
echo "Training done: $(date)"

echo "===== STEP 2: CONVERT ====="
$PYTHON convert_ds_to_hf.py \
  --ds_checkpoint $OUTPUT_DIR \
  --original_model $MODEL \
  --output_dir $HF_DIR \
  --ep_size 8 \
  2>&1 | tee $LOGDIR/codealpaca_${MODEL_SLUG}_${CONFIG_SLUG}_convert.log
echo "Convert done: $(date)"

# Verify .py files were copied
if [ ! -f "$HF_DIR/modeling_deepseek.py" ]; then
  echo "ERROR: modeling_deepseek.py not found in $HF_DIR"
  exit 1
fi
echo "Verified: custom code files present in HF model dir"

# Delete DS checkpoint to save disk (HF model is all we need)
echo "Removing DS checkpoint to save disk..."
rm -rf $OUTPUT_DIR
echo "DS checkpoint removed"

echo "===== STEP 3: GENERATE ====="
$PYTHON evaluate/humaneval/gen_humaneval.py \
  --model $HF_DIR \
  --output $EVAL_DIR \
  --instruction \
  2>&1 | tee $LOGDIR/codealpaca_${MODEL_SLUG}_${CONFIG_SLUG}_gen.log
echo "Generate done: $(date)"

echo "===== STEP 4: EVALUATE ====="
$PYTHON -m evalplus.evaluate \
  --dataset humaneval \
  --samples $EVAL_DIR/samples.jsonl \
  2>&1 | tee $LOGDIR/codealpaca_${MODEL_SLUG}_${CONFIG_SLUG}_eval.log
echo "Evaluate done: $(date)"

echo "===== ALL DONE ====="
echo ""
echo "========== RESULTS SUMMARY =========="
echo "--- Baseline (pre-finetune) ---"
grep -E "pass@1|Base|Plus" $LOGDIR/baseline_${MODEL_SLUG}_eval.log || true
echo ""
echo "--- Finetuned (post-finetune) ---"
grep -E "pass@1|Base|Plus" $LOGDIR/codealpaca_${MODEL_SLUG}_${CONFIG_SLUG}_eval.log || true
echo "====================================="
