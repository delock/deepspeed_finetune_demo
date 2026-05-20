# Lab Environment Setup

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

### 3. Clone and install DeepSpeed (autoep+Muon branch)

```bash
cd /workspace
git clone -b gma/autoep-muon-fixes https://github.com/deepspeedai/DeepSpeed.git
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

### 6. Clone finetune demo repo (if not already on /workspace)

```bash
cd /workspace
git clone https://github.com/comaniac/deepspeed_finetune_demo.git
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
for base, out_name in [('configs/z2_moonlight_autoep_adam.json', f'configs/z2_adam_{NUM_GPUS}gpu.json'),
                        ('configs/z2_moonlight_autoep_muon.json', f'configs/z2_muon_{NUM_GPUS}gpu.json')]:
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

## Common Env Vars

```bash
export HF_HOME=/workspace/.hf_cache
cd /workspace/deepspeed_finetune_demo
mkdir -p experiment_logs eval_results evalplus_results
NUM_GPUS=4  # or 8
PYTHON=/workspace/miniforge/envs/ds/bin/python
DS=/workspace/miniforge/envs/ds/bin/deepspeed
```

## Important Notes

- Do NOT add `--eval_steps` during training — eval with full model causes OOM
- vLLM eval uses `--tp 2` on 2 GPUs only (16 attention heads must divide evenly)
- `CUDA_VISIBLE_DEVICES=0,1` limits eval to first 2 GPUs
- HF cache must be on `/workspace` (root partition only 20GB)
- pip cache must also be on `/workspace` to avoid filling root partition
- After training, run `convert_ds_to_hf.py` before eval (DeepSpeed checkpoints are not directly loadable by vLLM)
