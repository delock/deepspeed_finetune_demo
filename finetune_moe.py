import torch
import time
import deepspeed
import argparse
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    default_data_collator
)
from transformers.integrations.deepspeed import (
    HfDeepSpeedConfig
)
import json
import random
import numpy as np
from deepspeed import comm as dist
import logging

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import wandb


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def preprocess_alpaca(example, tokenizer, max_length=512):
    prompt = f"### Instruction:\n{example['instruction']}\n\n"
    if example.get("input", ""):
        prompt += f"### Input:\n{example['input']}\n\n"
    prompt += f"### Response:\n{example['output']}"
    tokenized = tokenizer(prompt, truncation=True, max_length=max_length, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def calculate_accuracy(model_engine, eval_dataloader):
    """
    Calculate accuracy by comparing model predictions with true labels.
    Accuracy is computed as the percentage of correctly predicted tokens.
    """
    import torch
    from tqdm import tqdm
    from deepspeed import comm as dist

    model_engine.eval()
    total_correct = 0
    total_tokens = 0
    processed_batches = 0

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    with torch.no_grad():
        if rank == 0:
            enum = enumerate(tqdm(eval_dataloader, desc=f"Accuracy Calc [rank {rank}]", leave=False))
        else:
            enum = enumerate(eval_dataloader)

        for batch_idx, batch in enum:
            if batch_idx % world_size != rank:
                continue
            # in zero stage 3, you need other rank to participate to continue inference
            if batch_idx >= len(eval_dataloader) - (len(eval_dataloader) % world_size):
                continue

            batch = {k: v.to(model_engine.device) for k, v in batch.items()}

            # Get model predictions
            outputs = model_engine(**batch)
            logits = outputs.logits

            # Get predictions (highest probability tokens)
            predictions = torch.argmax(logits, dim=-1)

            # Get labels (ignore padding tokens - typically -100)
            labels = batch["labels"]

            # Create mask for non-padding tokens
            mask = labels != -100  # -100 is typically used for ignored tokens in labels

            # Count correct predictions only for non-padding tokens
            correct_predictions = (predictions == labels) & mask
            total_correct += correct_predictions.sum().item()
            total_tokens += mask.sum().item()

            processed_batches += 1

    model_engine.train()

    # Gather totals from all ranks
    device = model_engine.device
    local_correct = torch.tensor([total_correct], dtype=torch.float32, device=device)
    local_total = torch.tensor([total_tokens], dtype=torch.float32, device=device)

    if dist.is_initialized():
        dist.all_reduce(local_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_total, op=dist.ReduceOp.SUM)

    if local_total.item() == 0:
        return 0.0  # Return 0 accuracy if no valid tokens

    accuracy = (local_correct / local_total).item()
    return accuracy


def evaluate(model_engine, eval_dataloader, calc_accuracy=False):
    import torch
    from tqdm import tqdm
    from deepspeed import comm as dist

    model_engine.eval()
    losses = []
    total_correct = 0
    total_tokens = 0
    total_batches = len(eval_dataloader)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    # Split dataloader among ranks: each rank processes a subset of batches
    # Iterate only batches where (batch_idx % world_size) == rank
    with torch.no_grad():
        if rank == 0:
            enum = enumerate(tqdm(eval_dataloader, desc=f"Evaluating [rank {rank}]", leave=False))
        else:
            enum = enumerate(eval_dataloader)
        for batch_idx, batch in enum:
            if batch_idx % world_size != rank:
                continue
            # in zero stage 3, you need other rank to participate to continue inference
            if batch_idx >= len(eval_dataloader) - (len(eval_dataloader) % world_size):
                continue
            batch = {k: v.to(model_engine.device) for k, v in batch.items()}
            outputs = model_engine(**batch)
            loss = outputs.loss
            losses.append(loss.item())

            # Calculate accuracy if requested
            if calc_accuracy:
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                labels = batch["labels"]

                # Create mask for non-padding tokens
                mask = labels != -100  # -100 is typically used for ignored tokens in labels

                # Count correct predictions only for non-padding tokens
                correct_predictions = (predictions == labels) & mask
                total_correct += correct_predictions.sum().item()
                total_tokens += mask.sum().item()

    model_engine.train()

    # Gather total loss and total count from all ranks
    device = model_engine.device
    local_sum = torch.tensor([sum(losses)], dtype=torch.float32, device=device)
    local_count = torch.tensor([len(losses)], dtype=torch.float32, device=device)

    if dist.is_initialized():
        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_count, op=dist.ReduceOp.SUM)

    if local_count.item() == 0:
        return None, None if calc_accuracy else None

    avg_loss = (local_sum / local_count).item()

    if calc_accuracy:
        # Gather accuracy counts from all ranks
        local_correct = torch.tensor([total_correct], dtype=torch.float32, device=device)
        local_total = torch.tensor([total_tokens], dtype=torch.float32, device=device)

        if dist.is_initialized():
            dist.all_reduce(local_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_total, op=dist.ReduceOp.SUM)

        if local_total.item() == 0:
            accuracy = 0.0
        else:
            accuracy = (local_correct / local_total).item()

        return avg_loss, accuracy
    else:
        return avg_loss, None

def main(args):
    logging.basicConfig(level=logging.INFO, filename='pytorch_log.txt')
    set_seed(args.seed)

    # override batch size in ds_config
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)
    ds_config["train_batch_size"] = args.batch_size
    delattr(args, "deepspeed_config")

    dschf=HfDeepSpeedConfig(ds_config)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained( args.model_name, torch_dtype=torch.bfloat16)

    #config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    #with deepspeed.zero.Init():
        #model = AutoModelForCausalLM.from_pretrained( args.model_name, torch_dtype=torch.bfloat16)
        #model = AutoModelForCausalLM.from_config(config)

    model.gradient_checkpointing_enable()

    # Freeze all parameters except gate parameters BEFORE DeepSpeed initialization
    # This needs to be done before passing to DeepSpeed
    for name, param in model.named_parameters():
        if 'gate' in name.lower() and not 'gate_proj' in name.lower():
            param.requires_grad = True
            print(f"Unfrozen parameter: {name}")
        else:
            param.requires_grad = False

    # Enable input gradient requirements to ensure gradient flow
    # This is needed when using gradient checkpointing with partially frozen models
    model.enable_input_require_grads()

    # Load Alpaca 52K dataset and split into train/eval
    #dataset = load_dataset("tatsu-lab/alpaca")
    # Load the codealpaca dataset
    dataset = load_dataset("theblackcat102/evol-codealpaca-v1")
    split_dataset = dataset["train"].train_test_split(test_size=0.01, seed=args.seed)
    #train_dataset = split_dataset["train"]
    split_dataset2 = split_dataset["train"].train_test_split(test_size=0.01, seed=args.seed)
    #split_dataset3 = split_dataset2["train"].train_test_split(test_size=0.9, seed=args.seed)
    eval_dataset = split_dataset["test"]
    train_dataset = split_dataset2["train"]
    test_dataset = split_dataset2["test"]

    tokenized_train_dataset = train_dataset.map(lambda x: preprocess_alpaca(x, tokenizer), batched=False)
    tokenized_eval_dataset = eval_dataset.map(lambda x: preprocess_alpaca(x, tokenizer), batched=False)
    tokenized_test_dataset = test_dataset.map(lambda x: preprocess_alpaca(x, tokenizer), batched=False)

    # Create DataLoader - let DeepSpeed handle the actual batching
    train_dataloader = DataLoader(
        tokenized_train_dataset,
        batch_size=1,  # This will be overridden by DeepSpeed config
        collate_fn=default_data_collator,
        shuffle=True
    )
    eval_dataloader = DataLoader(
        tokenized_eval_dataset,
        batch_size=1,  # small eval batch for stability
        collate_fn=default_data_collator,
        shuffle=False
    )
    test_dataloader = DataLoader(
        tokenized_test_dataset,
        batch_size=1,  # small eval batch for stability
        collate_fn=default_data_collator,
        shuffle=False
    )

    # DeepSpeed will automatically parse the config file passed via --deepspeed argument
    model_engine, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_data=tokenized_train_dataset,
        collate_fn=default_data_collator,
        config=ds_config
    )

    # Load checkpoint if provided (after DeepSpeed initialization)
    print (model)

    # Calculate baseline accuracy before training
    #if args.calc_accuracy and dist.get_rank() == 0:
        #if dist.get_rank() == 0:
            #print("Calculating baseline accuracy before training...")
        #baseline_accuracy = calculate_accuracy(model_engine, test_dataloader)  # Limit batches for speed
        #if dist.get_rank() == 0:
            #print(f"Baseline accuracy before training: {baseline_accuracy:.4f}")

    model_engine.train()
    global_step = 0
    total_time = 0
    total_count = 0

    if args.profile_start >=0:
        prof = torch.profiler.profile(
                  activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                  record_shapes=True,
                  profile_memory=True,
              )
    else:
        prof = None

    # setup logging
    if args.wandb_name != None and dist.get_rank() == 0:
        wandb.init(project="deepspeed_finetune_demo", name=args.wandb_name)

    global_samples = 0
    # check accuracy before training
    #eval_loss, eval_accuracy = evaluate(model_engine, eval_dataloader, calc_accuracy=args.calc_accuracy)
    #if args.wandb_name != None:
        #wandb.log({"global_samples": 0, "loss": eval_loss})
    #if eval_accuracy is not None:
        #if args.wandb_name != None:
            #wandb.log({"global_samples": 0, "accuracy": eval_accuracy})
        #if dist.get_rank() == 0:
            #print(f"[Eval @ step before train] Average eval loss: {eval_loss:.4f}, Accuracy: {eval_accuracy:.4f}")
    #else:
        #if dist.get_rank() == 0:
            #print(f"[Eval @ step before train] Average eval loss: {eval_loss:.4f}")
    #for epoch in range(args.num_train_epochs):
    epoch = 1
    while True:
        if dist.get_rank() == 0:
            print(f"Starting epoch {epoch}")
        epoch += 1

        for step, batch in enumerate(train_dataloader):
            if prof != None and global_step == args.profile_start:
                prof.start()
            if prof != None and global_step - args.profile_start == args.profile_steps:
                prof.stop()
                # print profile
                if dist.get_rank() == 0:
                    prof.export_chrome_trace("trace.json")
                    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
            step_start_time = time.time()
            batch = {k: v.to(model_engine.device) for k, v in batch.items()}
            outputs = model_engine(**batch)
            loss = outputs.loss

            model_engine.backward(loss)
            model_engine.step()
            global_samples += model_engine.train_batch_size()

            step_time = time.time() - step_start_time
            if args.bench_start >= 0 and args.bench_steps > 0:
                if global_step >= args.bench_start:
                    total_time += step_time
                    total_count += 1
                if global_step >= args.bench_start + args.bench_steps - 1:
                    break

            if dist.get_rank() == 0 and global_step%10==0:  # Print every 10 steps
                print(f"Step {global_step}, Loss: {loss.item():.4f}, Time: {step_time*1000:.0f}ms")

            # Evaluation after every eval_steps
            #if args.eval_steps > 0 and global_step % args.eval_steps == 0 and global_step !=0:
                #eval_loss, eval_accuracy = evaluate(model_engine, eval_dataloader, calc_accuracy=args.calc_accuracy)
                #if dist.get_rank() == 0:
                    #if eval_loss is not None:
                        #if args.wandb_name != None:
                            #wandb.log({"global_samples": global_samples, "loss": eval_loss})
                        #if eval_accuracy is not None:
                            #if args.wandb_name != None:
                                #wandb.log({"global_samples": global_samples, "accuracy": eval_accuracy})
                            #print(f"[Eval @ step {global_step}] Average eval loss: {eval_loss:.4f}, Accuracy: {eval_accuracy:.4f}")
                        #else:
                            #print(f"[Eval @ step {global_step}] Average eval loss: {eval_loss:.4f}")
            global_step += 1
            if prof != None:
                prof.step()
            if global_samples >= args.num_train_samples:
                break

        if args.bench_start >= 0 and args.bench_steps > 0:
            if global_step >= args.bench_start + args.bench_steps - 1:
                break
        if global_samples >= args.num_train_samples:
                break

    if args.bench_start >= 0 and args.bench_steps > 0:
        if dist.get_rank() == 0:
            print (f"Average iteration time = {total_time/total_count}")

    # Save model using DeepSpeed's save_checkpoint method
    model_engine.save_checkpoint(f"{args.output_dir}{dist.get_rank()}")
    tokenizer.save_pretrained(f"{args.output_dir}{dist.get_rank()}")
    print("Model saved.")

    # Calculate final accuracy after training
    if args.calc_accuracy:
        if dist.get_rank() == 0:
            print("Calculating final accuracy after training...")
        final_accuracy = calculate_accuracy(model_engine, test_dataloader)  # Limit batches for speed
        if dist.get_rank() == 0:
            print(f"Final accuracy after training: {final_accuracy:.4f}")
#
        #if 'baseline_accuracy' in locals():
            #print(f"Accuracy improvement: {final_accuracy - baseline_accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument('--local_rank',
                    type=int,
                    default=-1,
                    help='local rank passed from distributed launcher')
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--profile_start", type=int, default=-1)
    parser.add_argument("--profile_steps", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=float, default=0.01)
    #parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--num_train_samples", type=int, default=4000)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bench_start", type=int, default=-1)
    parser.add_argument("--bench_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=0, help="Run evaluation every N steps (0 disables)")
    parser.add_argument("--calc_accuracy", action="store_true", help="Calculate accuracy during evaluation (slower)")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    main(args)
