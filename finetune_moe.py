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

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import wandb


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)






def preprocess_alpaca(example, tokenizer, max_length=512, mask_instruction_input=True):
    """
    Preprocess Alpaca examples for training.

    Args:
        example: The example dictionary with 'instruction', 'input', 'output'
        tokenizer: The tokenizer
        max_length: Maximum sequence length
        mask_instruction_input: If True, mask instruction and input parts in labels (standard approach)
                               If False, use full sequence as labels (experimental approach)
    """
    # Build the full prompt
    prompt = f"### Instruction:\n{example['instruction']}\n\n"
    if example.get("input", ""):
        prompt += f"### Input:\n{example['input']}\n\n"
    prompt += f"### Response:\n{example['output']}"

    # Tokenize the full prompt
    tokenized = tokenizer(prompt, truncation=True, max_length=max_length, padding="max_length")

    if mask_instruction_input:
        # Standard approach: mask instruction and input parts, only keep response part
        input_ids = tokenized["input_ids"]

        # Tokenize each part separately to find their lengths
        instruction_part = f"### Instruction:\n{example['instruction']}\n\n"
        tokenized_instruction = tokenizer(instruction_part, add_special_tokens=False)
        instruction_len = len(tokenized_instruction["input_ids"])

        input_part = ""
        if example.get("input", ""):
            input_part = f"### Input:\n{example['input']}\n\n"
            tokenized_input = tokenizer(input_part, add_special_tokens=False)
            input_len = len(tokenized_input["input_ids"])
        else:
            input_len = 0

        response_part = f"### Response:\n{example['output']}"
        tokenized_response = tokenizer(response_part, add_special_tokens=False)
        response_len = len(tokenized_response["input_ids"])

        # Create labels with -100 for instruction and input parts, actual tokens for response part
        labels = [-100] * len(tokenized["input_ids"])

        # Only the response part should have actual token IDs, others should be -100
        response_start_idx = instruction_len + input_len
        response_end_idx = min(response_start_idx + response_len, len(tokenized["input_ids"]))

        #for i in range(response_start_idx, response_end_idx):
        for i in range(0, response_end_idx):
            if i < len(tokenized["input_ids"]):
                labels[i] = tokenized["input_ids"][i]

        tokenized["labels"] = labels
    else:
        # Experimental approach: use full sequence as labels (no masking)
        tokenized["labels"] = tokenized["input_ids"].copy()

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

    model_engine.train()

    # Gather total loss and total count from all ranks
    device = model_engine.device
    local_sum = torch.tensor([sum(losses)], dtype=torch.float32, device=device)
    local_count = torch.tensor([len(losses)], dtype=torch.float32, device=device)

    if dist.is_initialized():
        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_count, op=dist.ReduceOp.SUM)

    if local_count.item() == 0:
        return None, None

    avg_loss = (local_sum / local_count).item()

    return avg_loss, None

# Gate analysis functions (commented out as per requirements)
'''
def analyze_gate_outputs(model, tokenizer, test_samples, device="cuda"):
    """
    Analyze the gate outputs of a MoE model to understand expert selection patterns.

    Args:
        model: The MoE model
        tokenizer: The tokenizer
        test_samples: List of test samples to analyze
        device: Device to run the analysis on

    Returns:
        Dictionary containing analysis results
    """
    import torch
    import numpy as np
    from collections import defaultdict

    model.eval()
    gate_analysis_results = {
        'expert_usage_counts': defaultdict(int),
        'average_router_probs': [],
        'routing_entropy': [],
        'sample_by_sample_analysis': []
    }

    with torch.no_grad():
        for i, sample in enumerate(test_samples):
            # Tokenize input
            inputs = tokenizer(sample['input_text'], return_tensors="pt", truncation=True, padding=True)
            input_ids = inputs["input_ids"].to(device)

            # Forward pass to get model outputs
            outputs = model(input_ids=input_ids)

            # Look for router/gate outputs in the model
            # This varies depending on the specific MoE implementation
            hidden_states = outputs.hidden_states if hasattr(outputs, 'hidden_states') else None

            # Try to access router logits if available in the model output
            if hasattr(outputs, 'router_logits'):
                router_logits = outputs.router_logits  # Tuple of tensors for each layer

                for layer_idx, layer_router_logits in enumerate(router_logits):
                    # Apply softmax to get probabilities
                    router_probs = torch.softmax(layer_router_logits, dim=-1)

                    # Calculate routing entropy (measure of uncertainty in expert selection)
                    entropy = -torch.sum(router_probs * torch.log(router_probs + 1e-10), dim=-1)
                    gate_analysis_results['routing_entropy'].extend(entropy.cpu().numpy())

                    # Get selected experts (top-k for each token)
                    top_k_vals, top_k_indices = torch.topk(router_probs, k=min(2, router_probs.size(-1)), dim=-1)

                    # Count expert usage
                    flat_indices = top_k_indices.view(-1).cpu().numpy()
                    for exp_id in flat_indices:
                        gate_analysis_results['expert_usage_counts'][exp_id] += 1

                # Calculate average router probabilities
                avg_probs = torch.mean(torch.cat([torch.softmax(logit, dim=-1) for logit in router_logits]), dim=0)
                gate_analysis_results['average_router_probs'].append(avg_probs.cpu().numpy())

            # Store sample-specific analysis
            sample_analysis = {
                'sample_idx': i,
                'input_length': len(input_ids[0]),
                'selected_experts': [] if not hasattr(outputs, 'router_logits') else
                                  [logit.topk(k=min(2, logit.size(-1)), dim=-1)[1].cpu().numpy()
                                   for logit in outputs.router_logits]
            }
            gate_analysis_results['sample_by_sample_analysis'].append(sample_analysis)

    return gate_analysis_results


def compare_gate_analysis(before_training_analysis, after_training_analysis):
    """
    Compare gate analysis results before and after training.

    Args:
        before_training_analysis: Gate analysis results before training
        after_training_analysis: Gate analysis results after training

    Returns:
        Dictionary containing comparison results
    """
    comparison_results = {}

    # Compare expert usage patterns
    before_expert_counts = before_training_analysis.get('expert_usage_counts', {})
    after_expert_counts = after_training_analysis.get('expert_usage_counts', {})

    # Calculate expert utilization statistics
    all_experts = set(list(before_expert_counts.keys()) + list(after_expert_counts.keys()))

    expert_usage_comparison = {}
    for expert_id in all_experts:
        before_count = before_expert_counts.get(expert_id, 0)
        after_count = after_expert_counts.get(expert_id, 0)
        expert_usage_comparison[expert_id] = {
            'before': before_count,
            'after': after_count,
            'change': after_count - before_count,
            'change_percentage': ((after_count - before_count) / (before_count + 1e-10)) * 100
        }

    comparison_results['expert_usage_comparison'] = expert_usage_comparison

    # Compare routing entropy
    before_entropy = before_training_analysis.get('routing_entropy', [])
    after_entropy = after_training_analysis.get('routing_entropy', [])

    comparison_results['entropy_stats'] = {
        'before_mean': np.mean(before_entropy) if before_entropy else 0,
        'after_mean': np.mean(after_entropy) if after_entropy else 0,
        'before_std': np.std(before_entropy) if before_entropy else 0,
        'after_std': np.std(after_entropy) if after_entropy else 0
    }

    # Compare average router probabilities
    before_avg_probs = before_training_analysis.get('average_router_probs', [])
    after_avg_probs = after_training_analysis.get('average_router_probs', [])

    comparison_results['probability_comparison'] = {
        'before_shape': [p.shape for p in before_avg_probs],
        'after_shape': [p.shape for p in after_avg_probs]
    }

    return comparison_results
'''

def main(args):
    logging.basicConfig(level=logging.INFO, filename='pytorch_log.txt')
    set_seed(args.seed)

    # override batch size in ds_config
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)
    ds_config["train_batch_size"] = args.batch_size

    # Extract micro batch size per gpu from config to use for evaluation
    train_micro_batch_size_per_gpu = ds_config.get("train_micro_batch_size_per_gpu", 1)

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

    # Check if the model has MoE gates by looking for 'gate' in parameter names
    has_moe_gates = any('gate' in name.lower() and 'gate_proj' not in name.lower() for name in model.state_dict().keys())

    if has_moe_gates:
        print("Detected MoE model with gate parameters. Freezing all parameters except gate parameters.")
        # Freeze all parameters except gate parameters BEFORE DeepSpeed initialization
        # This needs to be done before passing to DeepSpeed
        for name, param in model.named_parameters():
            if 'gate' in name.lower() and not 'gate_proj' in name.lower():
                param.requires_grad = True
                print(f"Unfrozen parameter: {name}")
            else:
                param.requires_grad = False
    else:
        print("Detected non-MoE model. Using default parameter freezing behavior.")
        # For non-MoE models, we don't modify the parameter freezing behavior
        # DeepSpeed will handle parameter management according to its configuration
        pass  # Do nothing - let DeepSpeed handle parameters as configured

    # Enable input gradient requirements to ensure gradient flow
    # This is needed when using gradient checkpointing with partially frozen models
    model.enable_input_require_grads()

    # Load Alpaca 52K dataset and split into train/eval
    dataset = load_dataset("tatsu-lab/alpaca")
    # Load the codealpaca dataset
    #dataset = load_dataset("theblackcat102/evol-codealpaca-v1")
    split_dataset = dataset["train"].train_test_split(test_size=0.01, seed=args.seed)
    #train_dataset = split_dataset["train"]
    split_dataset2 = split_dataset["train"].train_test_split(test_size=0.01, seed=args.seed)
    #split_dataset3 = split_dataset2["train"].train_test_split(test_size=0.9, seed=args.seed)
    eval_dataset = split_dataset["test"]
    train_dataset = split_dataset2["train"]
    test_dataset = split_dataset2["test"]

    # Limit training dataset size if specified
    if args.train_dataset_size is not None and args.train_dataset_size < len(train_dataset):
        train_dataset = train_dataset.select(range(args.train_dataset_size))
        print(f"Training dataset limited to {args.train_dataset_size} samples")
    else:
        print(f"Using full training dataset with {len(train_dataset)} samples")

    # Determine whether to mask instruction and input parts based on command line argument
    mask_instruction_input_during_training = not args.no_mask_instruction_input

    tokenized_train_dataset = train_dataset.map(
        lambda x: preprocess_alpaca(x, tokenizer, mask_instruction_input=mask_instruction_input_during_training),  # Use parameter to control masking
        batched=False
    )
    tokenized_eval_dataset = eval_dataset.map(
        lambda x: preprocess_alpaca(x, tokenizer, mask_instruction_input=True),  # Always mask for evaluation to measure response quality
        batched=False
    )
    tokenized_test_dataset = test_dataset.map(
        lambda x: preprocess_alpaca(x, tokenizer, mask_instruction_input=True),  # Always mask for testing to measure response quality
        batched=False
    )

    # Create DataLoader - let DeepSpeed handle the actual batching
    train_dataloader = DataLoader(
        tokenized_train_dataset,
        batch_size=1,  # This will be overridden by DeepSpeed config
        collate_fn=default_data_collator,
        shuffle=True
    )
    eval_dataloader = DataLoader(
        tokenized_eval_dataset,
        batch_size=train_micro_batch_size_per_gpu,  # Use same micro batch size as training for efficiency
        collate_fn=default_data_collator,
        shuffle=False
    )
    test_dataloader = DataLoader(
        tokenized_test_dataset,
        batch_size=train_micro_batch_size_per_gpu,  # Use same micro batch size as training for efficiency
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


    # Skip sampling and generation for now to speed up training
    test_samples = []
    before_training_outputs = []

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
    #for epoch in range(args.num_train_epochs):
    eval_loss, eval_accuracy = evaluate(model_engine, eval_dataloader)
    if dist.get_rank() == 0:
        print (f"Eval loss {eval_loss} @ global_samples {global_samples}")
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
            if (global_samples == 0 and dist.get_rank() == 0):
                print (batch)
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

            # Evaluate based on eval_steps parameter instead of hardcoded points
            if args.eval_steps > 0 and global_step > 0 and global_step % args.eval_steps == 0:
                eval_loss, eval_accuracy = evaluate(model_engine, eval_dataloader)
                if dist.get_rank() == 0:
                    print (f"Eval loss {eval_loss} @ global_step {global_step}, global_samples {global_samples}")
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
    checkpoint_path = f"{args.output_dir}/checkpoint_rank_{dist.get_rank()}"
    model_engine.save_checkpoint(checkpoint_path)
    if dist.get_rank() == 0:
        tokenizer_path = f"{args.output_dir}/tokenizer"
        tokenizer.save_pretrained(tokenizer_path)
        print(f"Checkpoint saved to: {checkpoint_path}")
        print(f"Tokenizer saved to: {tokenizer_path}")



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
    parser.add_argument("--train_dataset_size", type=int, default=None, help="Specify the size of the training dataset to use (None means use full dataset)")
    parser.add_argument("--no_mask_instruction_input", action="store_true", help="Don't mask instruction and input parts during training (experimental)")
    parser.add_argument("--bench_start", type=int, default=-1)
    parser.add_argument("--bench_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=0, help="Run evaluation every N steps (0 disables)")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    main(args)
