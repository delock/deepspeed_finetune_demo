NUM="${1:-2}"
TEACHER_MODEL="${2:-Qwen/Qwen2.5-3B}"
STUDENT_MODEL="${3:-Qwen/Qwen2.5-0.5B}"
CONFIG="${4:-z2_config.json}"
BATCH="${5:-8}"
TRAINSAMPLE="${6:-2048}"
TRAIN_DATASET_SIZE="${7:-}"
PROMPT_FILE="${8:-}"
ANALYZE="${9:-}"

# If a prompt file is provided, generate dataset first using the teacher model
if [ -n "$PROMPT_FILE" ] && [ "$PROMPT_FILE" != "" ]; then
    echo "Generating dataset using teacher model $TEACHER_MODEL and prompt file $PROMPT_FILE"

    # Calculate number of samples to generate (train samples + 30% for eval/test)
    TOTAL_SAMPLES=$(echo "$TRAINSAMPLE * 1.3" | bc | cut -d. -f1)

    # For faster testing, limit the number of samples generated
    # In production, remove this limitation
    if [ $TOTAL_SAMPLES -gt 50 ]; then
        TOTAL_SAMPLES=50
        echo "Limiting dataset generation to 50 samples for faster testing (would normally generate $TOTAL_SAMPLES)"
    fi

    # Generate dataset using the teacher model
    python generate_dataset.py --model_name $TEACHER_MODEL --prompt_file $PROMPT_FILE --num_samples $TOTAL_SAMPLES --output_file train_dataset.json

    # Use the generated dataset
    DATASET_FILE="train_dataset.json"
else
    DATASET_FILE=""
fi

if [ "$ANALYZE" = "--analyze_gates" ]; then
    if [ -n "$DATASET_FILE" ] && [ "$DATASET_FILE" != "" ]; then
        deepspeed --bind_cores_to_rank transfer_moe.py --model_name $STUDENT_MODEL --output_dir output --lr 2e-5 --batch_size $BATCH --deepspeed_config $CONFIG --num_train_samples $TRAINSAMPLE --train_dataset_size $TRAIN_DATASET_SIZE --dataset $DATASET_FILE --eval_steps 100 --no_mask_instruction_input
    else
        deepspeed --bind_cores_to_rank transfer_moe.py --model_name $STUDENT_MODEL --output_dir output --lr 2e-5 --batch_size $BATCH --deepspeed_config $CONFIG --num_train_samples $TRAINSAMPLE --train_dataset_size $TRAIN_DATASET_SIZE --eval_steps 100 --no_mask_instruction_input
    fi
else
    if [ -n "$TRAIN_DATASET_SIZE" ] && [ "$TRAIN_DATASET_SIZE" != "" ]; then
        if [ -n "$DATASET_FILE" ] && [ "$DATASET_FILE" != "" ]; then
            deepspeed --num_gpus=$NUM --master_port=$MASTER_PORT --bind_cores_to_rank transfer_moe.py --model_name $STUDENT_MODEL --output_dir output --lr 2e-5 --batch_size $BATCH --deepspeed_config $CONFIG --num_train_samples $TRAINSAMPLE --train_dataset_size $TRAIN_DATASET_SIZE --dataset $DATASET_FILE --eval_steps 100 --no_mask_instruction_input
        else
            deepspeed --num_gpus=$NUM --master_port=$MASTER_PORT --bind_cores_to_rank transfer_moe.py --model_name $STUDENT_MODEL --output_dir output --lr 2e-5 --batch_size $BATCH --deepspeed_config $CONFIG --num_train_samples $TRAINSAMPLE --train_dataset_size $TRAIN_DATASET_SIZE --eval_steps 100 --no_mask_instruction_input
        fi
    else
        if [ -n "$DATASET_FILE" ] && [ "$DATASET_FILE" != "" ]; then
            deepspeed --num_gpus=$NUM --master_port=$MASTER_PORT --bind_cores_to_rank transfer_moe.py --model_name $STUDENT_MODEL --output_dir output --lr 2e-5 --batch_size $BATCH --deepspeed_config $CONFIG --num_train_samples $TRAINSAMPLE --dataset $DATASET_FILE --eval_steps 100 --no_mask_instruction_input
        else
            deepspeed --num_gpus=$NUM --master_port=$MASTER_PORT --bind_cores_to_rank transfer_moe.py --model_name $STUDENT_MODEL --output_dir output --lr 2e-5 --batch_size $BATCH --deepspeed_config $CONFIG --num_train_samples $TRAINSAMPLE --eval_steps 100 --no_mask_instruction_input
        fi
    fi
fi
