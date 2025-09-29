# DeepSpeed finetune examples
This finetune example is extracted and modified from [ZenFlow Llama-2 Fine-Tuning Example](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/training/DeepSpeed-ZenFlow/finetuning) in [DeepSpeedExamples](https://github.com/deepspeedai/DeepSpeedExamples).  The purpose is to demostrate how to use different DeepSpeed training features and compare their performance in a single place.

Currently in DeepSpeedExamples, each technology has a dedicated directory to show how to use it.  However, DeepSpeed's philosophy is to allow users to use different features with different configuration file with no code change needed.  This project put this claim to the test.

# How to use

To run the example, simply run:
```
./run.sh <NUM_GPUS> <MODEL_NAME> <DS_CONFIG>
```

For example, if we want to run Qwen2.5-3B model with ZeRO offload, we can run:
```
./run.sh 2 Qwen2.5-3B zo_config.json
```
