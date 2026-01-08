NUM="${1:-2}"
MODEL="${2:-Qwen/Qwen2.5-0.5B}"
CONFIG="${3:-z2_config.json}"
BATCH="${4:-8}"
TRAINSAMPLE="${5:-2048}"
TRAIN_DATASET_SIZE="${6:-}"
#deepspeed --num_gpus=$NUM --bind_cores_to_rank finetune_moe.py --model_name $MODEL --output_dir output --lr 2e-5 --batch_size $BATCH --deepspeed_config $CONFIG --num_train_epochs 30 --eval_steps 100 --wandb_name $CONFIG-$BATCH
#deepspeed --bind_cores_to_rank finetune_moe.py --model_name $MODEL --output_dir output --lr 2e-5 --batch_size $BATCH --deepspeed_config $CONFIG --num_train_samples $TRAINSAMPLE --eval_steps 100 #--calc_accuracy

if [ -n "$TRAIN_DATASET_SIZE" ] && [ "$TRAIN_DATASET_SIZE" != "" ]; then
    deepspeed --num_gpus=$NUM --master_port=$MASTER_PORT --bind_cores_to_rank finetune_moe.py --model_name $MODEL --output_dir output --lr 2e-5 --batch_size $BATCH --deepspeed_config $CONFIG --num_train_samples $TRAINSAMPLE --train_dataset_size $TRAIN_DATASET_SIZE --eval_steps 100 --calc_accuracy
else
    deepspeed --num_gpus=$NUM --master_port=$MASTER_PORT --bind_cores_to_rank finetune_moe.py --model_name $MODEL --output_dir output --lr 2e-5 --batch_size $BATCH --deepspeed_config $CONFIG --num_train_samples $TRAINSAMPLE --eval_steps 100 --calc_accuracy
fi
