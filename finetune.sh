NUM="${1:-2}"
MODEL="${2:-Qwen/Qwen2.5-0.5B}"
CONFIG="${3:-z2_config.json}"
deepspeed --num_gpus=$NUM --bind_cores_to_rank finetune_llama.py --model_name $MODEL --output_dir output --lr 2e-5 --batch_size 8 --deepspeed_config $CONFIG --num_train_epochs 3
