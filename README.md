# DeepSpeed finetune examples
This finetune example is extracted and modified from [ZenFlow Llama-2 Fine-Tuning Example](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/training/DeepSpeed-ZenFlow/finetuning) in [DeepSpeedExamples](https://github.com/deepspeedai/DeepSpeedExamples).  The purpose is to demostrate how to use different DeepSpeed training features and compare their performance in a single place.

Currently in DeepSpeedExamples, each technology has a dedicated directory to show how to use it.  However, DeepSpeed's philosophy is to allow users to use different features with different configuration file with no code change needed.  This project put this claim to the test.

# How to use

To run the example, simply run:
```
./finetune.sh <NUM_GPUS> <MODEL_NAME> <DS_CONFIG>
```

For example, if we want to run Qwen2.5-3B model with ZeRO offload on 2 GPUs, we can run:
```
./finetune.sh 2 Qwen2.5-3B zo_config.json
```

## Key arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--batch_size` | Training batch size per GPU | required |
| `--eval_batch_size` | Eval batch size per rank | 1 |
| `--eval_steps` | Run evaluation every N steps (0 disables) | 0 |
| `--max_steps` | Stop after N steps (-1 = full epoch) | -1 |
| `--wandb_name` | Wandb run name (optional) | None |
| `--num_train_epochs` | Number of training epochs | 1 |
| `--weight_decay` | Weight decay | 0.01 |
| `--warmup` | Warmup steps | 0 |

Note: Learning rate is controlled entirely by the DeepSpeed config JSON, not by command-line arguments.

## Batch size
In DeepSpeed, batch size is decided by configuration file.  However, to avoid modify the config file, this python script takes `--batch_size` parameter and use it to decide train batch size.  Keep this in mind if you need to try different batch size.

## Wandb support
An optional `--wandb_name` can be supplied to finetune_llama.py to generate wandb graph.  But you need to modify `finetune.sh` manually to supply this argument.

# Benchmarking

To run benchmark, run:
```
./benchmark.sh <NUM_GPUS> <MODEL_NAME> <DS_CONFIG>
```

# Profiling

To run profiling, run:
```
./profile.sh <NUM_GPUS> <MODEL_NAME> <DS_CONFIG>
```

# Config files

For quick start, some config files are added, you may also modify the config to fit your need.

| Config File | Description |
|-------------|-------------|
| z2_config.json | ZeRO Stage 2 with AdamW |
| z3_config.json | ZeRO Stage 3 with AdamW |
| zo_config.json | ZeRO Offload, stage 2 |
| z3o_config.json | ZeRO Offload, stage 3 |
| zf_config.json | ZeRO Offload with ZenFlow |
| so_config.json | ZeRO Offload with SuperOffload |
| z2_muon.json | ZeRO 2 with Muon optimizer |
| z3_muon.json | ZeRO 3 with Muon optimizer |
| tp_config.json | ZeRO 2 with AutoTP |

## Muon optimizer config

Muon is a hybrid optimizer: it applies Muon updates to 2D hidden weights and Adam to everything else.  The config supports separate learning rates:

```json
{
    "optimizer": {
        "type": "Muon",
        "params": {
            "muon_lr": 1e-3,
            "adam_lr": 2e-5,
            "momentum": 0.95,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    }
}
```

| Parameter | Description |
|-----------|-------------|
| `muon_lr` | Learning rate for Muon (2D hidden weights) |
| `adam_lr` | Learning rate for Adam (embeddings, layer norms, lm_head, etc.) |
| `momentum` | Muon momentum factor |
| `betas` | Adam betas (for non-Muon parameters) |
| `eps` | Adam epsilon |
| `weight_decay` | Weight decay for both Muon and Adam parameters |
