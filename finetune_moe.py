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


# def analyze_gate_outputs(model_engine, eval_dataloader, tokenizer, step_name="before_training"):
#     """
#     Analyze the gate outputs of the MoE model.
#
#     Args:
#         model_engine: The DeepSpeed model engine
#         eval_dataloader: Dataloader for evaluation data
#         tokenizer: Tokenizer for processing text
#         step_name: Name to identify the analysis step (e.g., 'before_training', 'after_training')
#
#     Returns:
#         Dictionary containing analysis results
#     """
#     import torch
#     from tqdm import tqdm
#     from deepspeed import comm as dist
#
#     # Dictionary to store gate outputs
#     gate_outputs = {}
#
#     # Find all modules with 'gate' in their name and register hooks to capture their outputs
#     def register_gate_hooks(model):
#         hooks_registered = []
#         for name, module in model.named_modules():
#             if 'gate' in name.lower() and not 'gate_proj' in name.lower():
#                 def make_hook(n):
#                     def hook_fn(module, input, output):
#                         # Store the output of the gate layer
#                         if n not in gate_outputs:
#                             gate_outputs[n] = []
#                         # Detach from computation graph and move to CPU for storage
#                         if isinstance(output, torch.Tensor):
#                             gate_outputs[n].append(output.detach().cpu())
#                         elif isinstance(output, tuple):
#                             # If output is a tuple (like from some activation functions), take the first tensor
#                             if len(output) > 0 and isinstance(output[0], torch.Tensor):
#                                 gate_outputs[n].append(output[0].detach().cpu())
#                     return hook_fn
#
#                 hook = module.register_forward_hook(make_hook(name))
#                 hooks_registered.append((name, hook))
#         return hooks_registered
#
#     # Store original training state
#     was_training = model_engine.training
#
#     # Register hooks only during analysis
#     model_to_analyze = model_engine.module if hasattr(model_engine, 'module') else model_engine
#     registered_hooks = register_gate_hooks(model_to_analyze)
#
#     try:
#         model_engine.eval()
#         world_size = dist.get_world_size() if dist.is_initialized() else 1
#         rank = dist.get_rank() if dist.is_initialized() else 0
#
#         with torch.no_grad():
#             # Process a few batches to collect gate outputs
#             for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc=f"Collecting gate outputs [{step_name}]") if rank == 0 else eval_dataloader):
#                 if batch_idx >= 5:  # Limit to first 5 batches to reduce impact on performance
#                     break
#                 if batch_idx % world_size != rank:
#                     continue
#
#                 batch = {k: v.to(model_engine.device) for k, v in batch.items()}
#                 outputs = model_engine(**batch)
#                 # We only need the forward pass to trigger the hooks, not the output
#     finally:
#         # Remove hooks immediately after analysis - ensure they're always removed
#         for name, hook in registered_hooks:
#             hook.remove()
#
#         # Restore original training state
#         if was_training:
#             model_engine.train()
#
#     # Analyze the collected gate outputs
#     analysis_results = {}
#
#     for gate_name, outputs_list in gate_outputs.items():
#         if len(outputs_list) == 0:
#             continue
#
#         # Concatenate all outputs for this gate
#         all_outputs = torch.cat(outputs_list, dim=0)
#
#         # Calculate statistics
#         mean_output = all_outputs.mean().item()
#         std_output = all_outputs.std().item()
#         min_output = all_outputs.min().item()
#         max_output = all_outputs.max().item()
#
#         # Calculate activation statistics
#         num_total_elements = all_outputs.numel()
#         num_activated = (all_outputs != 0).sum().item()
#         activation_ratio = num_activated / num_total_elements if num_total_elements > 0 else 0
#
#         analysis_results[gate_name] = {
#             'mean': mean_output,
#             'std': std_output,
#             'min': min_output,
#             'max': max_output,
#             'activation_ratio': activation_ratio,
#             'shape': list(all_outputs.shape)
#         }
#
#     # Clear the gate_outputs to free memory
#     gate_outputs.clear()
#
#     # Print summary
#     if rank == 0:
#         print(f"\n=== Gate Output Analysis - {step_name} ===")
#         for gate_name, stats in analysis_results.items():
#             print(f"Gate: {gate_name}")
#             print(f"  Shape: {stats['shape']}")
#             print(f"  Mean: {stats['mean']:.6f}")
#             print(f"  Std: {stats['std']:.6f}")
#             print(f"  Min: {stats['min']:.6f}")
#             print(f"  Max: {stats['max']:.6f}")
#             print(f"  Activation Ratio: {stats['activation_ratio']:.6f}")
#             print()
#
#     return analysis_results
#
#
# def compare_gate_analysis(before_results, after_results):
#     """
#     Compare gate analysis results before and after fine-tuning.
#
#     Args:
#         before_results: Analysis results from before fine-tuning
#         after_results: Analysis results from after fine-tuning
#
#     Returns:
#         Dictionary with comparison results
#     """
#     from deepspeed import comm as dist
#
#     rank = dist.get_rank() if dist.is_initialized() else 0
#
#     comparison_results = {}
#
#     # Find common gate names in both results
#     common_gates = set(before_results.keys()) & set(after_results.keys())
#
#     for gate_name in common_gates:
#         before_stats = before_results[gate_name]
#         after_stats = after_results[gate_name]
#
#         comparison_results[gate_name] = {
#             'mean_change': after_stats['mean'] - before_stats['mean'],
#             'std_change': after_stats['std'] - before_stats['std'],
#             'min_change': after_stats['min'] - before_stats['min'],
#             'max_change': after_stats['max'] - before_stats['max'],
#             'activation_ratio_change': after_stats['activation_ratio'] - before_stats['activation_ratio'],
#             'before': before_stats,
#             'after': after_stats
#         }
#
#     # Print comparison summary
#     if rank == 0:
#         print("\n=== Gate Output Changes After Fine-tuning ===")
#         for gate_name, comparison in comparison_results.items():
#             print(f"Gate: {gate_name}")
#             print(f"  Mean Change: {comparison['mean_change']:.6f}")
#             print(f"  Std Change: {comparison['std_change']:.6f}")
#             print(f"  Min Change: {comparison['min_change']:.6f}")
#             print(f"  Max Change: {comparison['max_change']:.6f}")
#             print(f"  Activation Ratio Change: {comparison['activation_ratio_change']:.6f}")
#             print()
#
#     return comparison_results


def sample_test_examples(test_dataset, tokenizer, num_samples=10, seed=42):
    """
    Sample random examples from test dataset for comparison.

    Args:
        test_dataset: The test dataset
        tokenizer: The tokenizer
        num_samples: Number of samples to take (default 10)
        seed: Random seed for reproducibility

    Returns:
        List of dictionaries containing input text and labels
    """
    import random
    from tqdm import tqdm
    random.seed(seed)

    # Randomly sample indices
    sampled_indices = random.sample(range(len(test_dataset)), num_samples)

    samples = []
    for idx in tqdm(sampled_indices, desc="Sampling test examples", total=num_samples):
        example = test_dataset[idx]

        # Reconstruct the input text from the example
        prompt = f"### Instruction:\n{example['instruction']}\n\n"
        if example.get("input", ""):
            prompt += f"### Input:\n{example['input']}\n\n"
        prompt += f"### Response:\n"

        # Get the full response (ground truth)
        full_prompt = f"### Instruction:\n{example['instruction']}\n\n"
        if example.get("input", ""):
            full_prompt += f"### Input:\n{example['input']}\n\n"
        full_prompt += f"### Response:\n{example['output']}"

        samples.append({
            'input_text': prompt,
            'full_text': full_prompt,
            'ground_truth': example['output'],
            'original_idx': idx
        })

    return samples


def generate_model_outputs(model_engine, samples, tokenizer, max_new_tokens=256):
    """
    Generate outputs from the model for the given samples.

    Args:
        model_engine: The DeepSpeed model engine
        samples: List of sample dictionaries
        tokenizer: The tokenizer
        max_new_tokens: Maximum number of new tokens to generate

    Returns:
        List of generated outputs
    """
    import torch
    from deepspeed import comm as dist
    from tqdm import tqdm

    rank = dist.get_rank() if dist.is_initialized() else 0

    # Store original training state
    was_training = model_engine.training
    model_engine.eval()

    generated_outputs = []

    # Use tqdm for progress bar if on rank 0
    sample_iterator = tqdm(enumerate(samples), total=len(samples), desc="Generating outputs", disable=(rank != 0))

    for i, sample in sample_iterator:
        # Tokenize input
        inputs = tokenizer(sample['input_text'], return_tensors="pt", truncation=True, padding=True)
        input_ids = inputs["input_ids"].to(model_engine.device)

        # Generate output
        with torch.no_grad():
            outputs = model_engine.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode the generated part only (excluding input)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the generated part (after the input)
        if sample['input_text'] in generated_text:
            generated_part = generated_text.split(sample['input_text'], 1)[1].strip()
        else:
            # If the input text is not found in the generated text, return the whole generated text
            generated_part = generated_text[len(tokenizer.decode(input_ids[0], skip_special_tokens=True)):]

        generated_outputs.append(generated_part)

    # Restore original training state
    if was_training:
        model_engine.train()

    return generated_outputs


def format_comparison_output(samples, before_outputs, after_outputs):
    """
    Format the comparison output in a human-readable format.

    Args:
        samples: List of sample dictionaries
        before_outputs: List of outputs before training
        after_outputs: List of outputs after training
    """
    print("\n" + "="*80)
    print("MoE Model Output Comparison - Sample Analysis")
    print("="*80)
    print()

    for i, (sample, before_out, after_out) in enumerate(zip(samples, before_outputs, after_outputs)):
        print(f"Sample #{i+1}:")
        print(f"Input: {repr(sample['input_text'])}")
        print("-" * 80)
        print(f"BEFORE Training: {repr(before_out)}")
        print("-" * 80)
        print(f"AFTER Training:  {repr(after_out)}")
        print("-" * 80)
        print(f"GROUND TRUTH:    {repr(sample['ground_truth'])}")
        print("=" * 80)
        print()


def preprocess_alpaca(example, tokenizer, max_length=512):
    prompt = f"### Instruction:\n{example['instruction']}\n\n"
    if example.get("input", ""):
        prompt += f"### Input:\n{example['input']}\n\n"
    prompt += f"### Response:\n{example['output']}"
    tokenized = tokenizer(prompt, truncation=True, max_length=max_length, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def calculate_accuracy(model_engine, eval_dataloader, tokenizer):
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

        all_predictions = []
        all_references = []
        for batch_idx, batch in enum:
            #if batch_idx % world_size != rank:
                #continue
            # in zero stage 3, you need other rank to participate to continue inference
            #if batch_idx >= len(eval_dataloader) - (len(eval_dataloader) % world_size):
                #continue

            batch = {k: v.to(model_engine.device) for k, v in batch.items()}

            # Get model predictions
            outputs = model_engine(**batch)
            logits = outputs.logits

            # Get predictions (highest probability tokens)
            predictions = torch.argmax(logits, dim=-1)

            # Get labels (ignore padding tokens - typically -100)
            labels = batch["labels"]

            assert (len(predictions)==len(labels))
            for i in range(len(predictions)):
                all_predictions.append(predictions[i])
                all_references.append(labels[i])

            # Create mask for non-padding tokens
            mask = labels != -100  # -100 is typically used for ignored tokens in labels

            # Count correct predictions only for non-padding tokens
            correct_predictions = (predictions == labels) & mask
            total_correct += correct_predictions.sum().item()
            total_tokens += mask.sum().item()

            processed_batches += 1

            # 先把张量搬到 CPU 并转成纯 Python 列表
        pred_ids = [p.detach().cpu().tolist() for p in all_predictions]  # List[List[int]]

        # 如果 label 是字符串，就可以跳过 decode
        label_ids = [p.detach().cpu().tolist() for p in all_references]  # List[List[int]]

        # 用 tokenizer 批量 decode 成字符串
        # 注意：skip_special_tokens=True 可以去掉 <pad>, <bos>, <eos> 等特殊符号
        pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        ref_texts  = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # 如果每个样本有多个参考答案，组织成 List[List[str]] 的结构；否则也可以单参考：
        references = [[r] for r in ref_texts]  # 单参考
        predictions = pred_texts

        preds_tok = [p.split() for p in predictions]
        refs_tok = [[r.split() for r in ref_group] for ref_group in references]

        corpus_bleu_4 = corpus_bleu(refs_tok, preds_tok, weights=(0.25, 0.25, 0.25, 0.25))
        print(f"Corpus BLEU-4: {corpus_bleu_4:.4f}")



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
    if args.calc_accuracy:
        if dist.get_rank() == 0:
            print("Calculating baseline accuracy before training...")
        baseline_accuracy = calculate_accuracy(model_engine, test_dataloader, tokenizer)  # Limit batches for speed
        if dist.get_rank() == 0:
            print(f"Baseline accuracy before training: {baseline_accuracy:.4f}")

    # Sample test examples for comparison
    if dist.get_rank() == 0:
        print("Sampling test examples for comparison...")
    test_samples = sample_test_examples(test_dataset, tokenizer, num_samples=10, seed=args.seed)

    # Generate outputs before training
    if dist.get_rank() == 0:
        print("Generating outputs before training...")
    before_training_outputs = generate_model_outputs(model_engine, test_samples, tokenizer)

    # Perform gate output analysis before fine-tuning
    # if dist.get_rank() == 0:
    #     print("Performing gate output analysis before fine-tuning...")
    # gate_analysis_before = analyze_gate_outputs(model_engine, test_dataloader, tokenizer, step_name="before_training")

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

    # Generate outputs after training
    if dist.get_rank() == 0:
        print("Generating outputs after training...")
    after_training_outputs = generate_model_outputs(model_engine, test_samples, tokenizer)

    # Format comparison output
    if dist.get_rank() == 0:
        print("Formatting comparison output...")
    format_comparison_output(test_samples, before_training_outputs, after_training_outputs)

    # Perform gate output analysis after fine-tuning
    # if dist.get_rank() == 0:
    #     print("Performing gate output analysis after fine-tuning...")
    # gate_analysis_after = analyze_gate_outputs(model_engine, test_dataloader, tokenizer, step_name="after_training")

    # Compare gate analysis results
    # if dist.get_rank() == 0:
    #     print("Comparing gate output changes...")
    # gate_comparison = compare_gate_analysis(gate_analysis_before, gate_analysis_after)

    # Save model using DeepSpeed's save_checkpoint method
    model_engine.save_checkpoint(f"{args.output_dir}{dist.get_rank()}")
    tokenizer.save_pretrained(f"{args.output_dir}{dist.get_rank()}")
    print("Model saved.")

    # Calculate final accuracy after training
    if args.calc_accuracy:
        if dist.get_rank() == 0:
            print("Calculating final accuracy after training...")
        final_accuracy = calculate_accuracy(model_engine, test_dataloader, tokenizer)  # Limit batches for speed
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
