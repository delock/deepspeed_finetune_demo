#!/bin/bash
# eval_humaneval.sh - Evaluate a model on HumanEval and HumanEval+
# Usage: bash evaluate/humaneval/eval_humaneval.sh <model_path> <output_name> [tp]
# Run from repo root.
#
# Example:
#   bash evaluate/humaneval/eval_humaneval.sh moonshotai/Moonlight-16B-A3B baseline 8
#   bash evaluate/humaneval/eval_humaneval.sh ./output_muon/hf_model muon_finetuned 8

set -e

# Always run from repo root so paths are consistent
cd "$(git rev-parse --show-toplevel)"

MODEL_PATH=${1:?Usage: bash eval_humaneval.sh <model_path> <output_name> [tp]}
OUTPUT_NAME=${2:?Usage: bash eval_humaneval.sh <model_path> <output_name> [tp]}
TP=${3:-8}
ROOT="evalplus_results/${OUTPUT_NAME}"

echo "=== Evaluating ${MODEL_PATH} as '${OUTPUT_NAME}' (tp=${TP}) ==="

# Generate completions for HumanEval
echo "[1/4] Generating HumanEval completions..."
evalplus.codegen \
    --model "${MODEL_PATH}" \
    --dataset humaneval \
    --backend vllm \
    --tp "${TP}" \
    --root "${ROOT}" \
    --trust_remote_code \
    --attn_implementation flash_attention_2 \
    --dtype bfloat16 \
    --temperature 0.0 \
    --n_samples 1 \
    --force_base_prompt

# Evaluate HumanEval
echo "[2/4] Evaluating HumanEval..."
evalplus.evaluate \
    --dataset humaneval \
    --samples "${ROOT}/"*humaneval* \
    2>&1 | tee "${ROOT}/humaneval_results.txt"

# Generate completions for HumanEval+
echo "[3/4] Generating HumanEval+ completions..."
# HumanEval+ uses the same generations, just stricter tests
# evalplus.evaluate with --dataset humaneval automatically runs both

echo "[4/4] Results summary:"
echo "--- HumanEval & HumanEval+ ---"
grep -E "pass@1|Base|Plus" "${ROOT}/humaneval_results.txt" || true

echo "=== Done: ${OUTPUT_NAME} ==="
