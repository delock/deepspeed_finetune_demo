# RunPod Environment Setup for MMLU/GSM8K/MBPP Experiments

## Prerequisites

- Allocate a RunPod instance with H200 GPUs (4 or 8 GPUs)
- Select a CUDA 12.x base image
- Ensure `/workspace` has enough disk space (~500GB)

## Environment Setup

### 1. Create conda environment

```bash
/workspace/miniforge/bin/conda create -n ds python=3.11 -y -p /workspace/ds
```

If miniforge is not installed:

```bash
curl -L -o /workspace/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /workspace/miniforge.sh -b -p /workspace/miniforge
```

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
/workspace/ds/bin/pip install accelerate --cache-dir $PIP_CACHE
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
mkdir -p experiment_logs eval_results
NUM_GPUS=4  # or 8
```

### MMLU Baseline Eval (no training)

```bash
CUDA_VISIBLE_DEVICES=0,1 /workspace/ds/bin/python evaluate/mmlu/gen_mmlu.py \
  --model_name moonshotai/Moonlight-16B-A3B \
  --output_dir eval_results/mmlu_baseline --tp 2

/workspace/ds/bin/python evaluate/mmlu/eval_mmlu.py --samples eval_results/mmlu_baseline/samples.jsonl
```

### AdamW MMLU Finetune

```bash
/workspace/ds/bin/deepspeed --num_gpus=$NUM_GPUS finetune_llama.py \
  --model_name moonshotai/Moonlight-16B-A3B \
  --output_dir output_mmlu_adam \
  --batch_size 16 --max_length 512 \
  --deepspeed_config z2_adam_${NUM_GPUS}gpu.json \
  --dataset_name cais/mmlu \
  --num_train_epochs 1 \
  2>&1 | tee experiment_logs/mmlu_adam_train.log | tail -1 &
```

### Muon MMLU Finetune

```bash
/workspace/ds/bin/deepspeed --num_gpus=$NUM_GPUS finetune_llama.py \
  --model_name moonshotai/Moonlight-16B-A3B \
  --output_dir output_mmlu_muon \
  --batch_size 16 --max_length 512 \
  --deepspeed_config z2_muon_${NUM_GPUS}gpu.json \
  --dataset_name cais/mmlu \
  --num_train_epochs 1 \
  2>&1 | tee experiment_logs/mmlu_muon_train.log | tail -1 &
```

### Eval Finetuned Models

```bash
# AdamW
CUDA_VISIBLE_DEVICES=0,1 /workspace/ds/bin/python evaluate/mmlu/gen_mmlu.py \
  --model_path output_mmlu_adam --output_dir eval_results/mmlu_adam --tp 2
/workspace/ds/bin/python evaluate/mmlu/eval_mmlu.py --samples eval_results/mmlu_adam/samples.jsonl

# Muon
CUDA_VISIBLE_DEVICES=0,1 /workspace/ds/bin/python evaluate/mmlu/gen_mmlu.py \
  --model_path output_mmlu_muon --output_dir eval_results/mmlu_muon --tp 2
/workspace/ds/bin/python evaluate/mmlu/eval_mmlu.py --samples eval_results/mmlu_muon/samples.jsonl
```

## Important Notes

- Do NOT add `--eval_steps` during training — eval with full model causes OOM
- vLLM eval uses `--tp 2` on 2 GPUs only (16 attention heads must divide evenly)
- `CUDA_VISIBLE_DEVICES=0,1` limits eval to first 2 GPUs
- HF cache must be on `/workspace` (root partition only 20GB)
- pip cache must also be on `/workspace` to avoid filling root partition
- Previous results: MMLU baseline = 40.05% (5624/14042 correct)
