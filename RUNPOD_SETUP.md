# RunPod Environment Setup for MMLU/GSM8K/MBPP Experiments

## Prerequisites

- Allocate a RunPod instance with H200 GPUs (4 or 8 GPUs)
- Select a CUDA 12.x base image
- Ensure `/workspace` has enough disk space (~500GB)

## Step 0: Verify GPU Health

RunPod machines can have faulty GPUs. Always test before investing time in setup.

```bash
/workspace/miniforge/envs/ds/bin/python -c "
import torch
for i in range(torch.cuda.device_count()):
    try:
        torch.cuda.set_device(i)
        x = torch.zeros(1, device=i)
        print(f'GPU {i}: OK, mem={torch.cuda.mem_get_info(i)[0]/1e9:.1f}GB free')
    except Exception as e:
        print(f'GPU {i}: FAIL - {e}')
"
```

If any GPU fails, request a new node from RunPod. Do NOT proceed with faulty GPUs.

## Environment Setup

### 1. Create conda environment

```bash
/workspace/miniforge/bin/conda create -n ds python=3.11 -y
/workspace/miniforge/bin/conda install -n ds pip -y
```

Note: conda env will be at `/workspace/miniforge/envs/ds/`. Use `-n ds` (not `-p`) because
`-n` and `-p` conflict. Then install pip separately since conda doesn't include it by default.

If miniforge is not installed:

```bash
curl -L -o /workspace/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /workspace/miniforge.sh -b -p /workspace/miniforge
```

All pip commands below use `/workspace/miniforge/envs/ds/bin/pip`.

### 2. Install PyTorch + vLLM + transformers

```bash
export HF_HOME=/workspace/.hf_cache
export PIP_CACHE=/workspace/.pip_cache
mkdir -p $HF_HOME $PIP_CACHE

/workspace/ds/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 --cache-dir $PIP_CACHE
/workspace/ds/bin/pip install vllm==0.7.2 transformers datasets --cache-dir $PIP_CACHE
```

### 3. Install local DeepSpeed

```bash
/workspace/ds/bin/pip install -e /workspace/DeepSpeed --cache-dir $PIP_CACHE
```

### 4. Install missing dependencies

```bash
/workspace/ds/bin/pip install accelerate wandb --cache-dir $PIP_CACHE
/workspace/ds/bin/pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl" --cache-dir $PIP_CACHE
```

### 5. Verify environment

```bash
/workspace/ds/bin/python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}, Device: {torch.cuda.get_device_name(0)}')"
/workspace/ds/bin/python -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')"
/workspace/ds/bin/python -c "import flash_attn; print(f'flash_attn: {flash_attn.__version__}')"
```

### 6. Clone repos (if not already on /workspace)

```bash
cd /workspace
git clone -b add_mmlu_gsm8k https://github.com/comaniac/deepspeed_finetune_demo.git
git clone -b gma/autoep-muon-fixes https://github.com/deepspeedai/DeepSpeed.git
```

## GPU Count Configuration

The number of GPUs (`NUM_GPUS`) affects several config values:

| GPUs | autoep_size (Adam) | autoep_size (Muon) | train_batch_size | grad_acc | micro_batch_per_gpu |
|------|-------------------|--------------------|-----------------|----------|--------------------|
| 4    | 4                 | 4                  | 16              | 2        | 2                  |
| 8    | 4                 | 8                  | 16              | 2        | 1                  |

**Rules:**
- `autoep_size` must be divisible by `NUM_GPUS`
- `train_batch_size = micro_batch_per_gpu * gradient_accumulation_steps * NUM_GPUS`
- `eval` uses vLLM with `--tp 2` (16 heads divisible by 2, NOT by 4) on 2 GPUs

### Generating per-GPU-count configs

```bash
cd /workspace/deepspeed_finetune_demo
NUM_GPUS=4  # or 8

python3 -c "
import json
for base, out_name in [('z2_moonlight_autoep_adam.json', f'z2_adam_{NUM_GPUS}gpu.json'),
                        ('z2_moonlight_autoep_muon.json', f'z2_muon_{NUM_GPUS}gpu.json')]:
    with open(base) as f:
        c = json.load(f)
    c['expert_parallel']['autoep_size'] = ${NUM_GPUS}
    with open(out_name, 'w') as f:
        json.dump(c, f, indent=4)
    print('wrote', out_name)
"
```

Note: For Muon with 4 GPUs, autoep_size must be changed from 8 to 4.
For Muon with 8 GPUs, autoep_size stays at 8.

## Running Experiments

### Set common env vars

```bash
export HF_HOME=/workspace/.hf_cache
cd /workspace/deepspeed_finetune_demo
mkdir -p experiment_logs eval_results evalplus_results
NUM_GPUS=4  # or 8
PYTHON=/workspace/miniforge/envs/ds/bin/python
DS=/workspace/miniforge/envs/ds/bin/deepspeed
```

---

## MMLU Experiments

Training dataset: `cais/mmlu` (auxiliary_train split, ~95k examples).
Eval uses `evaluate/mmlu/gen_mmlu.py` (vLLM generation) + `evaluate/mmlu/eval_mmlu.py` (accuracy scoring).

### MMLU Baseline Eval (no training)

```bash
CUDA_VISIBLE_DEVICES=0,1 $PYTHON evaluate/mmlu/gen_mmlu.py \
  --model moonshotai/Moonlight-16B-A3B \
  --output eval_results/mmlu_baseline --tp 2

$PYTHON evaluate/mmlu/eval_mmlu.py --samples eval_results/mmlu_baseline/samples.jsonl
```

### MMLU AdamW Finetune

```bash
$DS --num_gpus=$NUM_GPUS finetune_llama.py \
  --model_name moonshotai/Moonlight-16B-A3B \
  --output_dir output_mmlu_adam \
  --batch_size 16 --max_length 512 \
  --deepspeed_config z2_adam_${NUM_GPUS}gpu.json \
  --dataset_name cais/mmlu \
  --num_train_epochs 1 \
  2>&1 | tee experiment_logs/mmlu_adam_train.log &
```

### MMLU Muon Finetune

```bash
$DS --num_gpus=$NUM_GPUS finetune_llama.py \
  --model_name moonshotai/Moonlight-16B-A3B \
  --output_dir output_mmlu_muon \
  --batch_size 16 --max_length 512 \
  --deepspeed_config z2_muon_${NUM_GPUS}gpu.json \
  --dataset_name cais/mmlu \
  --num_train_epochs 1 \
  2>&1 | tee experiment_logs/mmlu_muon_train.log &
```

### MMLU Convert & Eval Finetuned Models

After training completes, convert DeepSpeed checkpoints to HF format, then evaluate:

```bash
# Convert AdamW checkpoint
$PYTHON convert_ds_to_hf.py \
  --ds_checkpoint output_mmlu_adam \
  --original_model moonshotai/Moonlight-16B-A3B \
  --output_dir hf_model_mmlu_adam \
  --ep_size $NUM_GPUS

# Convert Muon checkpoint
$PYTHON convert_ds_to_hf.py \
  --ds_checkpoint output_mmlu_muon \
  --original_model moonshotai/Moonlight-16B-A3B \
  --output_dir hf_model_mmlu_muon \
  --ep_size $NUM_GPUS

# Eval AdamW
CUDA_VISIBLE_DEVICES=0,1 $PYTHON evaluate/mmlu/gen_mmlu.py \
  --model hf_model_mmlu_adam --output eval_results/mmlu_adam --tp 2
$PYTHON evaluate/mmlu/eval_mmlu.py --samples eval_results/mmlu_adam/samples.jsonl

# Eval Muon
CUDA_VISIBLE_DEVICES=0,1 $PYTHON evaluate/mmlu/gen_mmlu.py \
  --model hf_model_mmlu_muon --output eval_results/mmlu_muon --tp 2
$PYTHON evaluate/mmlu/eval_mmlu.py --samples eval_results/mmlu_muon/samples.jsonl
```

### MMLU Previous Results (4x H200, autoep_size=4)

| Optimizer | Learning Rate | adam_lr | MMLU Accuracy |
|-----------|--------------|---------|---------------|
| baseline  | —            | —       | 0.401 (40.05%) |
| AdamW     | 2e-6         | —       | 0.660 (65.96%) |
| Muon      | 2e-4         | 2e-6    | 0.677 (67.66%) |

---

## MBPP Experiments

Training dataset: `sahil2801/CodeAlpaca-20k` (code instruction tuning).
Eval uses `evaluate/humaneval/gen_vllm.py --dataset mbpp` (vLLM generation) + `evalplus.evaluate` (pass@1 scoring via EvalPlus).

### Install EvalPlus (MBPP eval dependency)

```bash
/workspace/miniforge/envs/ds/bin/pip install evalplus --cache-dir $PIP_CACHE
```

### MBPP Baseline Eval (no training)

```bash
CUDA_VISIBLE_DEVICES=0,1 $PYTHON evaluate/humaneval/gen_vllm.py \
  --model moonshotai/Moonlight-16B-A3B \
  --output evalplus_results/mbpp_baseline \
  --dataset mbpp --tp 2

$PYTHON -m evalplus.evaluate --dataset mbpp \
  --samples "evalplus_results/mbpp_baseline/*mbpp*" 2>&1 | tee experiment_logs/mbpp_baseline_eval.log
```

### MBPP AdamW Finetune

```bash
$DS --num_gpus=$NUM_GPUS finetune_llama.py \
  --model_name moonshotai/Moonlight-16B-A3B \
  --output_dir output_mbpp_adam \
  --batch_size 16 --max_length 512 \
  --deepspeed_config z2_adam_${NUM_GPUS}gpu.json \
  --dataset_name sahil2801/CodeAlpaca-20k \
  --num_train_epochs 1 \
  2>&1 | tee experiment_logs/mbpp_adam_train.log &
```

### MBPP Muon Finetune

```bash
$DS --num_gpus=$NUM_GPUS finetune_llama.py \
  --model_name moonshotai/Moonlight-16B-A3B \
  --output_dir output_mbpp_muon \
  --batch_size 16 --max_length 512 \
  --deepspeed_config z2_muon_${NUM_GPUS}gpu.json \
  --dataset_name sahil2801/CodeAlpaca-20k \
  --num_train_epochs 1 \
  2>&1 | tee experiment_logs/mbpp_muon_train.log &
```

### MBPP Convert & Eval Finetuned Models

```bash
# Convert AdamW checkpoint
$PYTHON convert_ds_to_hf.py \
  --ds_checkpoint output_mbpp_adam \
  --original_model moonshotai/Moonlight-16B-A3B \
  --output_dir hf_model_mbpp_adam \
  --ep_size $NUM_GPUS

# Convert Muon checkpoint
$PYTHON convert_ds_to_hf.py \
  --ds_checkpoint output_mbpp_muon \
  --original_model moonshotai/Moonlight-16B-A3B \
  --output_dir hf_model_mbpp_muon \
  --ep_size $NUM_GPUS

# Eval AdamW
CUDA_VISIBLE_DEVICES=0,1 $PYTHON evaluate/humaneval/gen_vllm.py \
  --model hf_model_mbpp_adam \
  --output evalplus_results/mbpp_adam \
  --dataset mbpp --tp 2
$PYTHON -m evalplus.evaluate --dataset mbpp \
  --samples "evalplus_results/mbpp_adam/*mbpp*" 2>&1 | tee experiment_logs/mbpp_adam_eval.log

# Eval Muon
CUDA_VISIBLE_DEVICES=0,1 $PYTHON evaluate/humaneval/gen_vllm.py \
  --model hf_model_mbpp_muon \
  --output evalplus_results/mbpp_muon \
  --dataset mbpp --tp 2
$PYTHON -m evalplus.evaluate --dataset mbpp \
  --samples "evalplus_results/mbpp_muon/*mbpp*" 2>&1 | tee experiment_logs/mbpp_muon_eval.log
```

### MBPP Previous Results (4x H200, autoep_size=4)

| Optimizer | Learning Rate | adam_lr | MBPP | MBPP+ |
|-----------|--------------|---------|------|-------|
| baseline  | —            | —       | 0.495| 0.431 |
| AdamW     | 2e-6         | —       | 0.611| 0.505 |
| Muon      | 2e-4         | 2e-6    | 0.661| 0.553 |

---

## GSM8K Experiments

Training dataset: `meta-math/MetaMathQA` (~375k math reasoning examples).
Eval uses `evaluate/gsm8k/gen_gsm8k.py` (vLLM generation) + `evaluate/gsm8k/eval_gsm8k.py` (accuracy scoring).

### GSM8K Baseline Eval (no training)

```bash
CUDA_VISIBLE_DEVICES=0,1 $PYTHON evaluate/gsm8k/gen_gsm8k.py \
  --model moonshotai/Moonlight-16B-A3B \
  --output eval_results/gsm8k_baseline --tp 2

$PYTHON evaluate/gsm8k/eval_gsm8k.py --samples eval_results/gsm8k_baseline/samples.jsonl
```

### GSM8K AdamW Finetune

```bash
$DS --num_gpus=$NUM_GPUS finetune_llama.py \
  --model_name moonshotai/Moonlight-16B-A3B \
  --output_dir output_gsm8k_adam \
  --batch_size 16 --max_length 512 \
  --deepspeed_config z2_adam_${NUM_GPUS}gpu.json \
  --dataset_name meta-math/MetaMathQA \
  --num_train_epochs 1 \
  2>&1 | tee experiment_logs/gsm8k_adam_train.log &
```

### GSM8K Muon Finetune

```bash
$DS --num_gpus=$NUM_GPUS finetune_llama.py \
  --model_name moonshotai/Moonlight-16B-A3B \
  --output_dir output_gsm8k_muon \
  --batch_size 16 --max_length 512 \
  --deepspeed_config z2_muon_${NUM_GPUS}gpu.json \
  --dataset_name meta-math/MetaMathQA \
  --num_train_epochs 1 \
  2>&1 | tee experiment_logs/gsm8k_muon_train.log &
```

### GSM8K Convert & Eval Finetuned Models

```bash
# Convert AdamW checkpoint (use step subdirectory)
$PYTHON convert_ds_to_hf.py \
  --ds_checkpoint output_gsm8k_adam/step_XXXXX \
  --original_model moonshotai/Moonlight-16B-A3B \
  --output_dir hf_model_gsm8k_adam \
  --ep_size $NUM_GPUS

# Convert Muon checkpoint
$PYTHON convert_ds_to_hf.py \
  --ds_checkpoint output_gsm8k_muon/step_XXXXX \
  --original_model moonshotai/Moonlight-16B-A3B \
  --output_dir hf_model_gsm8k_muon \
  --ep_size $NUM_GPUS

# Eval AdamW
CUDA_VISIBLE_DEVICES=0,1 $PYTHON evaluate/gsm8k/gen_gsm8k.py \
  --model hf_model_gsm8k_adam --output eval_results/gsm8k_adam --tp 2
$PYTHON evaluate/gsm8k/eval_gsm8k.py --samples eval_results/gsm8k_adam/samples.jsonl

# Eval Muon
CUDA_VISIBLE_DEVICES=0,1 $PYTHON evaluate/gsm8k/gen_gsm8k.py \
  --model hf_model_gsm8k_muon --output eval_results/gsm8k_muon --tp 2
$PYTHON evaluate/gsm8k/eval_gsm8k.py --samples eval_results/gsm8k_muon/samples.jsonl
```

### GSM8K Results (4x H200, autoep_size=4)

| Optimizer | Learning Rate | adam_lr | GSM8K |
|-----------|--------------|---------|-------|
| baseline  | —            | —       | 52.62% |
| AdamW     | 2e-6         | —       | 81.96% |
| Muon      | 2e-4         | 2e-6    | 79.91% |

---

## Important Notes

- Do NOT add `--eval_steps` during training — eval with full model causes OOM
- vLLM eval uses `--tp 2` on 2 GPUs only (16 attention heads must divide evenly)
- `CUDA_VISIBLE_DEVICES=0,1` limits eval to first 2 GPUs
- HF cache must be on `/workspace` (root partition only 20GB)
- pip cache must also be on `/workspace` to avoid filling root partition
- After training, run `convert_ds_to_hf.py` before eval (DeepSpeed checkpoints are not directly loadable by vLLM)
