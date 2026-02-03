import torch
import time
import deepspeed
import argparse
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from transformers.integrations.deepspeed import HfDeepSpeedConfig
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


def preprocess_alpaca(example, tokenizer, max_length=2048):
    # Build instruction part (will be masked from loss)
    instruction = f"### Instruction:\n{example['instruction']}\n\n"
    if example.get("input", ""):
        instruction += f"### Input:\n{example['input']}\n\n"
    instruction += "### Response:\n"
    response = example["output"]

    full_prompt = instruction + response
    tokenized = tokenizer(
        full_prompt, truncation=True, max_length=max_length, padding="max_length"
    )

    # Find instruction length to mask it from loss
    # Use full_prompt tokenization to get accurate instruction boundary after truncation
    instruction_ids = tokenizer(instruction, add_special_tokens=False)["input_ids"]
    instruction_len = len(instruction_ids)

    # Ensure at least one token is unmasked to avoid NaN loss
    # If instruction is longer than max_length, only mask padding tokens
    seq_len = sum(1 for t in tokenized["input_ids"] if t != tokenizer.pad_token_id)
    if instruction_len >= seq_len:
        instruction_len = max(0, seq_len - 1)  # Keep at least the last non-pad token

    # Mask instruction and padding tokens in labels (set to -100, ignored by CrossEntropyLoss)
    labels = tokenized["input_ids"].copy()
    for i in range(len(labels)):
        if i < instruction_len or labels[i] == tokenizer.pad_token_id:
            labels[i] = -100
    tokenized["labels"] = labels
    return tokenized


def evaluate(model_engine, eval_dataloader):
    import torch
    from tqdm import tqdm
    from deepspeed import comm as dist

    model_engine.eval()
    losses = []
    total_batches = len(eval_dataloader)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    # Split dataloader among ranks: each rank processes a subset of batches
    # Iterate only batches where (batch_idx % world_size) == rank
    with torch.no_grad():
        if rank == 0:
            enum = enumerate(
                tqdm(eval_dataloader, desc=f"Evaluating [rank {rank}]", leave=False)
            )
        else:
            enum = enumerate(eval_dataloader)
        for batch_idx, batch in enum:
            if batch_idx % world_size != rank:
                continue
            batch = {k: v.to(model_engine.device) for k, v in batch.items()}
            outputs = model_engine(**batch)
            loss = outputs.loss
            losses.append(loss.item())
    model_engine.train()

    # Gather total loss and total count from all ranks
    device = model_engine.device
    local_sum = torch.tensor([sum(losses)], dtype=torch.float32, device=device)
    local_count = torch.tensor([len(losses)], dtype=torch.float32, device=device)

    if dist.is_initialized():
        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_count, op=dist.ReduceOp.SUM)

    if local_count.item() == 0:
        return None

    avg_loss = (local_sum / local_count).item()
    return avg_loss


def print_r(rank, arg):
    if rank == dist.get_rank():
        print(arg)


def main(args):
    logging.basicConfig(level=logging.INFO, filename="pytorch_log.txt")
    set_seed(args.seed)

    # override batch size in ds_config
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)
    ds_config["train_batch_size"] = args.batch_size
    delattr(args, "deepspeed_config")
    # make sure models are properly loaded in zero3
    dschf = HfDeepSpeedConfig(ds_config)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16
    )
    model.gradient_checkpointing_enable()

    """
    # the code below allows you to train only part of the parameters
    # we haven't parameterize this part yet, so uncomment down below and modify the code manually

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
    """

    # Load Alpaca 52K dataset and split into train/eval
    dataset = load_dataset(args.dataset_name)
    split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=args.seed)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    tokenized_train_dataset = train_dataset.map(
        lambda x: preprocess_alpaca(x, tokenizer), batched=False
    )
    tokenized_eval_dataset = eval_dataset.map(
        lambda x: preprocess_alpaca(x, tokenizer), batched=False
    )

    # Create DataLoader - let DeepSpeed handle the actual batching
    train_dataloader = DataLoader(
        tokenized_train_dataset,
        batch_size=1,  # This will be overridden by DeepSpeed config
        collate_fn=default_data_collator,
        shuffle=True,
    )
    eval_dataloader = DataLoader(
        tokenized_eval_dataset,
        batch_size=1,  # small eval batch for stability
        collate_fn=default_data_collator,
        shuffle=False,
    )

    # DeepSpeed will automatically parse the config file passed via --deepspeed argument
    model_engine, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_data=tokenized_train_dataset,
        collate_fn=default_data_collator,
        config=ds_config,
    )

    model_engine.train()
    global_step = 0
    total_time = 0
    total_count = 0

    # skip unnecessary evaluation
    save_checkpoint_p = True
    if args.bench_start >= 0 and args.bench_steps > 0:
        save_checkpoint_p = False
    if args.profile_start >= 0:
        save_checkpoint_p = False

    if args.profile_start >= 0:
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
        )
    else:
        prof = None

    # setup logging
    if args.wandb_name != None and dist.get_rank() == 0:
        wandb.init(project="deepspeed_finetune_demo", name=args.wandb_name)

    global_samples = 0
    for epoch in range(args.num_train_epochs):
        print_r(0, f"Starting epoch {epoch + 1}/{args.num_train_epochs}")

        for step, batch in enumerate(train_dataloader):
            if prof != None and global_step == args.profile_start:
                prof.start()
            if prof != None and global_step - args.profile_start == args.profile_steps:
                prof.stop()
                # print profile
                if dist.get_rank() == 0:
                    prof.export_chrome_trace("trace.json")
                    print(
                        prof.key_averages().table(
                            sort_by="self_cuda_time_total", row_limit=10
                        )
                    )
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

            if dist.get_rank() == 0:
                wandb.log({"global_samples": global_samples, "train-loss": loss})
            if global_step % 10 == 0:  # Print every 10 steps
                msg = f"Step {global_step}, Loss: {loss.item():.4f}, Time: {step_time * 1000:.0f}ms"
                print_r(0, msg)
                if dist.get_rank() == 0:
                    logging.info(msg)

            # Evaluation after every eval_steps
            if (
                args.eval_steps > 0
                and global_step % args.eval_steps == 0
                and save_checkpoint_p
            ):
                eval_loss = evaluate(model_engine, eval_dataloader)
                if dist.get_rank() == 0:
                    if eval_loss is not None:
                        eval_loss_val = float(eval_loss)
                        if args.wandb_name != None:
                            wandb.log(
                                {
                                    "global_samples": global_samples,
                                    "eval-loss": eval_loss_val,
                                }
                            )
                        eval_msg = f"[Eval @ step {global_step}] Eval Loss: {eval_loss_val:.4f}"
                        print(eval_msg, flush=True)
                        logging.info(eval_msg)
                    else:
                        eval_msg = f"[Eval @ step {global_step}] Eval Loss unavailable (no eval batches processed)"
                        print(eval_msg, flush=True)
                        logging.info(eval_msg)
            global_step += 1
            if prof != None:
                prof.step()

        if args.bench_start >= 0 and args.bench_steps > 0:
            if global_step >= args.bench_start + args.bench_steps - 1:
                break

    if args.bench_start >= 0 and args.bench_steps > 0:
        print_r(0, f"Average iteration time = {total_time / total_count}")

    if save_checkpoint_p:
        # Save model using DeepSpeed's save_checkpoint method
        # on zero3, it is necessary for each rank to save the checkpoint
        # for other stage, we just save on all ranks anyway
        output_dir_rank = os.path.join(args.output_dir, f"{dist.get_rank()}")
        model_engine.save_checkpoint(output_dir_rank)
        tokenizer.save_pretrained(output_dir_rank)

    print_r(0, "Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="tatsu-lab/alpaca")
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--profile_start", type=int, default=-1)
    parser.add_argument("--profile_steps", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bench_start", type=int, default=-1)
    parser.add_argument("--bench_steps", type=int, default=100)
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=0,
        help="Run evaluation every N steps (0 disables)",
    )
    parser.add_argument("--wandb_name", type=str, default=None)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    main(args)
