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
| z2_config.json | ZeRO Stage 2 |
| z3_config.json | ZeRO Stage 3 |
| zo_config.json | ZeRO Offload, stage 2 |
| z3o_config.json | ZeRO Offload, stage 3 |
| zf_config.json | ZeRO Offload with ZenFlow |
| so_config.json | ZeRO Offload with SuperOffload |
| z2_muon.json | ZeRO 2 with Muon optimizer |
| tp_config.json | ZeRO 2 with AutoTP |
