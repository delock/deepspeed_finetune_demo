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
    Generate outputs from the model for the given samples using forward pass.
    This function gets the model's response to the input by using the logits
    from the forward pass. For efficiency, we only perform a single forward pass
    and return the logits for the last token position.

    Args:
        model_engine: The DeepSpeed model engine
        samples: List of sample dictionaries
        tokenizer: The tokenizer
        max_new_tokens: Maximum number of new tokens to generate (not used in forward pass)

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

        # Use forward pass to get logits
        with torch.no_grad():
            outputs = model_engine(input_ids=input_ids)
            logits = outputs.logits

        # Get the predicted token for the next position (after the last input token)
        next_token_logits = logits[:, -1, :]  # Get logits for the next token
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # Decode the single predicted token
        predicted_text = tokenizer.decode(next_token_id[0], skip_special_tokens=True)

        generated_outputs.append(predicted_text)

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

        for i in range(response_start_idx, response_end_idx):
            if i < len(tokenized["input_ids"]):
                labels[i] = tokenized["input_ids"][i]

        tokenized["labels"] = labels
    else:
        # Experimental approach: use full sequence as labels (no masking)
        tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized

def calculate_accuracy(model_engine, eval_dataloader, tokenizer):
    """
    Calculate accuracy by comparing model predictions with true labels.
    Accuracy is computed as the percentage of correctly predicted tokens.
    Distributed implementation: each rank processes a subset of samples.
    Returns both full sequence accuracy and response-only accuracy.
    """
    import torch
    from tqdm import tqdm
    from deepspeed import comm as dist

    model_engine.eval()
    total_correct_full = 0
    total_tokens_full = 0
    total_correct_response = 0
    total_tokens_response = 0
    processed_batches = 0

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    # Lists to store prompts and targets for potential BLEU evaluation
    all_predictions = []
    all_references = []

    # Collect all batches first, then distribute among ranks
    all_batches = []
    batch_count = 0

    with torch.no_grad():
        # Collect all batches from the dataloader
        for batch_idx, batch in enumerate(eval_dataloader):
            all_batches.append(batch)
            batch_count += 1

    # Record original batch count before padding
    original_batches_count = len(all_batches)

    # Pad batches to ensure total number is divisible by world_size for ZeRO-3 compatibility
    remainder = original_batches_count % world_size
    if remainder != 0:
        # Add copies of the first few batches to make total_batches divisible by world_size
        batches_needed = world_size - remainder
        for i in range(batches_needed):
            all_batches.append(all_batches[i])

    # Calculate total number of batches each rank will iterate through
    # This ensures all ranks perform the same number of iterations for synchronization
    total_iterations = len(all_batches)

    # Initialize progress bar with total number of batches across all ranks
    if rank == 0:
        progress_bar = tqdm(total=total_iterations,
                           desc=f"Accuracy Calc [rank {rank}]",
                           leave=False)
    else:
        progress_bar = None

    # All ranks iterate through all padded batches, but only accumulate results for original batches
    for batch_idx in range(total_iterations):
        batch = all_batches[batch_idx]

        # Print batch structure for debugging (only first batch on rank 0)
        if rank == 0 and batch_idx == 0:
            print(f"\n[Batch Debug] Keys: {list(batch.keys())}")
            for k, v in batch.items():
                if k in ['input_ids', 'labels']:
                    # Decode text and stop at special control characters
                    sample_ids = v[0] if len(v.shape) > 1 and v.shape[0] > 0 else v
                    sample_ids_list = sample_ids.cpu().tolist() if hasattr(sample_ids, 'tolist') else [sample_ids]

                    # Filter out invalid token IDs to prevent overflow error
                    # Valid token IDs should be in range [0, vocab_size)
                    vocab_size = tokenizer.vocab_size
                    filtered_ids = []
                    for token_id in sample_ids_list:
                        # Check if token_id is valid (non-negative and within vocab range)
                        if isinstance(token_id, int) and 0 <= token_id < vocab_size:
                            filtered_ids.append(token_id)
                        elif isinstance(token_id, int) and token_id == -100:
                            # Skip -100 tokens (used for ignored positions)
                            continue
                        elif isinstance(token_id, int) and token_id < 0:
                            # For other negative values, skip them
                            continue
                        else:
                            # If it's not an int, skip it
                            continue

                    if filtered_ids:  # Only decode if there are valid tokens
                        decoded_text = tokenizer.decode(filtered_ids, skip_special_tokens=True)  # Skip special tokens to avoid control chars
                        # Find first control character and truncate there
                        clean_text = ""
                        for char in decoded_text:
                            if ord(char) < 32 and char not in ['\n', '\t']:  # Control chars except newline and tab
                                break
                            clean_text += char
                        print(f"[Batch Debug] {k}: shape={v.shape}, decoded_text={repr(clean_text)}")  # Show clean text without length limit
                    else:
                        print(f"[Batch Debug] {k}: shape={v.shape}, dtype={v.dtype}, no valid tokens to decode")
                else:
                    print(f"[Batch Debug] {k}: shape={v.shape}, dtype={v.dtype}")
            print()

        batch = {k: v.to(model_engine.device) for k, v in batch.items()}

        # ALL ranks execute model forward pass for synchronization in ZeRO-3
        outputs = model_engine(**batch)
        logits = outputs.logits

        # Get predictions (highest probability tokens)
        predictions = torch.argmax(logits, dim=-1)

        # Get labels (ignore padding tokens - typically -100)
        labels = batch["labels"]

        assert (len(predictions)==len(labels))

        # Accumulate results only for original batches (not padded ones), and only for assigned batches
        is_original_batch = batch_idx < original_batches_count
        is_assigned_to_this_rank = batch_idx % world_size == rank

        if is_assigned_to_this_rank:
            for i in range(len(predictions)):
                all_predictions.append(predictions[i])
                all_references.append(labels[i])

            # Get input_ids for full sequence comparison
            input_ids = batch["input_ids"]

            # Create mask for full sequence (all non-padding tokens)
            # This includes instruction, input, and response parts
            mask_full = input_ids != tokenizer.pad_token_id  # Use input_ids to identify all non-padding tokens

            # Count correct predictions for full sequence (comparing predictions with input_ids)
            correct_predictions_full = (predictions == input_ids) & mask_full
            if is_original_batch:  # Only accumulate for original batches
                total_correct_full += correct_predictions_full.sum().item()
                total_tokens_full += mask_full.sum().item()

            # For response-only accuracy, we only consider the response part
            # In our preprocessing, only the response part has non-masked tokens (not -100)
            # So mask_response is the same as mask_full in our current setup, but conceptually different
            mask_response = labels != -100  # -100 is used for ignored tokens in labels
            correct_predictions_response = (predictions == labels) & mask_response  # Compare with labels for response part only
            if is_original_batch:  # Only accumulate for original batches
                total_correct_response += correct_predictions_response.sum().item()
                total_tokens_response += mask_response.sum().item()

            if is_original_batch:
                processed_batches += 1

        if progress_bar:
            progress_bar.update(1)

    if progress_bar:
        progress_bar.close()

    # Calculate token-level accuracies
    # 先把张量搬到 CPU 并转成纯 Python 列表
    pred_ids = [p.detach().cpu().tolist() for p in all_predictions]  # List[List[int]]

    # 如果 label 是字符串，就可以跳过 decode
    label_ids = [p.detach().cpu().tolist() for p in all_references]  # List[List[int]]

    # Filter out invalid token IDs to prevent overflow error
    # Valid token IDs should be in range [0, vocab_size) or -100 (ignored tokens)
    vocab_size = tokenizer.vocab_size
    filtered_pred_ids = []
    filtered_label_ids = []

    for pred_seq in pred_ids:
        filtered_seq = []
        for token_id in pred_seq:
            if isinstance(token_id, int) and 0 <= token_id < vocab_size:
                filtered_seq.append(token_id)
            elif isinstance(token_id, int) and token_id == -100:
                # Keep -100 tokens but replace with a valid token ID for decoding
                # We'll use the pad token ID as a placeholder
                filtered_seq.append(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
            elif isinstance(token_id, int) and token_id < 0:
                # For other negative values, use pad token as placeholder
                filtered_seq.append(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
            else:
                # If it's not an int, try to convert or use pad token
                try:
                    int_token_id = int(token_id)
                    if 0 <= int_token_id < vocab_size:
                        filtered_seq.append(int_token_id)
                    else:
                        filtered_seq.append(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
                except:
                    filtered_seq.append(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
        filtered_pred_ids.append(filtered_seq)

    for label_seq in label_ids:
        filtered_seq = []
        for token_id in label_seq:
            if isinstance(token_id, int) and 0 <= token_id < vocab_size:
                filtered_seq.append(token_id)
            elif isinstance(token_id, int) and token_id == -100:
                # Keep -100 tokens but replace with a valid token ID for decoding
                # We'll use the pad token ID as a placeholder
                filtered_seq.append(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
            elif isinstance(token_id, int) and token_id < 0:
                # For other negative values, use pad token as placeholder
                filtered_seq.append(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
            else:
                # If it's not an int, try to convert or use pad token
                try:
                    int_token_id = int(token_id)
                    if 0 <= int_token_id < vocab_size:
                        filtered_seq.append(int_token_id)
                    else:
                        filtered_seq.append(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
                except:
                    filtered_seq.append(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
        filtered_label_ids.append(filtered_seq)

    # 用 tokenizer 批量 decode 成字符串
    # 注意：skip_special_tokens=True 可以去掉 <pad>, <bos>, <eos> 等特殊符号
    pred_texts = tokenizer.batch_decode(filtered_pred_ids, skip_special_tokens=True)
    ref_texts  = tokenizer.batch_decode(filtered_label_ids, skip_special_tokens=True)

    # 输出第一个样本的预测和参考文本，用于调试 (only on rank 0)
    if rank == 0 and len(pred_texts) > 0 and len(ref_texts) > 0:
        print(f"\n--- First Sample Analysis ---")
        print(f"Predicted text: {repr(pred_texts[0])}")
        print(f"Reference text: {repr(ref_texts[0])}")
        print(f"--- End First Sample Analysis ---\n")

    model_engine.train()

    # Gather totals from all ranks
    device = model_engine.device
    local_correct_full = torch.tensor([total_correct_full], dtype=torch.float32, device=device)
    local_total_full = torch.tensor([total_tokens_full], dtype=torch.float32, device=device)
    local_correct_response = torch.tensor([total_correct_response], dtype=torch.float32, device=device)
    local_total_response = torch.tensor([total_tokens_response], dtype=torch.float32, device=device)

    if dist.is_initialized():
        dist.all_reduce(local_correct_full, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_total_full, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_correct_response, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_total_response, op=dist.ReduceOp.SUM)

    # Calculate accuracies
    if local_total_full.item() == 0:
        accuracy_full = 0.0  # Return 0 accuracy if no valid tokens
    else:
        accuracy_full = (local_correct_full / local_total_full).item()

    if local_total_response.item() == 0:
        accuracy_response = 0.0  # Return 0 accuracy if no valid tokens
    else:
        accuracy_response = (local_correct_response / local_total_response).item()

    # Print token counts on rank 0
    if rank == 0:
        print(f"Full Sequence - Correctly predicted tokens: {int(local_correct_full.item())}")
        print(f"Full Sequence - Total tokens to predict: {int(local_total_full.item())}")
        print(f"Full Sequence Accuracy: {accuracy_full:.6f}")
        print(f"Response Only - Correctly predicted tokens: {int(local_correct_response.item())}")
        print(f"Response Only - Total tokens to predict: {int(local_total_response.item())}")
        print(f"Response Only Accuracy: {accuracy_response:.6f}")

    return accuracy_full, accuracy_response


def calculate_accuracy_detailed(model_engine, eval_dataloader, tokenizer):
    """
    Calculate accuracy by comparing model predictions with true labels.
    Returns both full sequence accuracy and response-only accuracy.
    Accuracy is computed as the percentage of correctly predicted tokens.
    Distributed implementation: each rank processes a subset of samples.
    """
    import torch
    from tqdm import tqdm
    from deepspeed import comm as dist

    model_engine.eval()
    total_correct_full = 0
    total_tokens_full = 0
    total_correct_response = 0
    total_tokens_response = 0
    processed_batches = 0

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    # Lists to store prompts and targets for potential BLEU evaluation
    all_predictions = []
    all_references = []

    # Collect all batches first, then distribute among ranks
    all_batches = []
    batch_count = 0

    with torch.no_grad():
        # Collect all batches from the dataloader
        for batch_idx, batch in enumerate(eval_dataloader):
            all_batches.append(batch)
            batch_count += 1

    # Record original batch count before padding
    original_batches_count = len(all_batches)

    # Pad batches to ensure total number is divisible by world_size for ZeRO-3 compatibility
    remainder = original_batches_count % world_size
    if remainder != 0:
        # Add copies of the first few batches to make total_batches divisible by world_size
        batches_needed = world_size - remainder
        for i in range(batches_needed):
            all_batches.append(all_batches[i])

    # Calculate total number of batches each rank will iterate through
    # This ensures all ranks perform the same number of iterations for synchronization
    total_iterations = len(all_batches)

    # Initialize progress bar with total number of batches across all ranks
    if rank == 0:
        progress_bar = tqdm(total=total_iterations,
                           desc=f"Detailed Accuracy Calc [rank {rank}]",
                           leave=False)
    else:
        progress_bar = None

    # All ranks iterate through all padded batches, but only accumulate results for original batches
    for batch_idx in range(total_iterations):
        batch = all_batches[batch_idx]

        # Print batch structure for debugging (only first batch on rank 0)
        if rank == 0 and batch_idx == 0:
            print(f"\n[Batch Debug] Keys: {list(batch.keys())}")
            for k, v in batch.items():
                if k in ['input_ids', 'labels']:
                    # Decode text and stop at special control characters
                    sample_ids = v[0] if len(v.shape) > 1 and v.shape[0] > 0 else v
                    sample_ids_list = sample_ids.cpu().tolist() if hasattr(sample_ids, 'tolist') else [sample_ids]

                    # Filter out invalid token IDs to prevent overflow error
                    # Valid token IDs should be in range [0, vocab_size)
                    vocab_size = tokenizer.vocab_size
                    filtered_ids = []
                    for token_id in sample_ids_list:
                        # Check if token_id is valid (non-negative and within vocab range)
                        if isinstance(token_id, int) and 0 <= token_id < vocab_size:
                            filtered_ids.append(token_id)
                        elif isinstance(token_id, int) and token_id == -100:
                            # Skip -100 tokens (used for ignored positions)
                            continue
                        elif isinstance(token_id, int) and token_id < 0:
                            # For other negative values, skip them
                            continue
                        else:
                            # If it's not an int, skip it
                            continue

                    if filtered_ids:  # Only decode if there are valid tokens
                        decoded_text = tokenizer.decode(filtered_ids, skip_special_tokens=True)  # Skip special tokens to avoid control chars
                        # Find first control character and truncate there
                        clean_text = ""
                        for char in decoded_text:
                            if ord(char) < 32 and char not in ['\n', '\t']:  # Control chars except newline and tab
                                break
                            clean_text += char
                        print(f"[Batch Debug] {k}: shape={v.shape}, decoded_text={repr(clean_text)}")  # Show clean text without length limit
                    else:
                        print(f"[Batch Debug] {k}: shape={v.shape}, dtype={v.dtype}, no valid tokens to decode")
                else:
                    print(f"[Batch Debug] {k}: shape={v.shape}, dtype={v.dtype}")
            print()

        batch = {k: v.to(model_engine.device) for k, v in batch.items()}

        # ALL ranks execute model forward pass for synchronization in ZeRO-3
        outputs = model_engine(**batch)
        logits = outputs.logits

        # Get predictions (highest probability tokens)
        predictions = torch.argmax(logits, dim=-1)

        # Get labels (ignore padding tokens - typically -100)
        labels = batch["labels"]

        assert (len(predictions)==len(labels))

        # Accumulate results only for original batches (not padded ones), and only for assigned batches
        is_original_batch = batch_idx < original_batches_count
        is_assigned_to_this_rank = batch_idx % world_size == rank

        if is_assigned_to_this_rank:
            for i in range(len(predictions)):
                all_predictions.append(predictions[i])
                all_references.append(labels[i])

            # Create mask for full sequence (all non-padding tokens)
            # This includes instruction, input, and response parts
            mask_full = input_ids != tokenizer.pad_token_id  # Use input_ids to identify all non-padding tokens

            # Count correct predictions for full sequence
            correct_predictions_full = (predictions == input_ids) & mask_full  # Compare with input_ids for full sequence
            if is_original_batch:  # Only accumulate for original batches
                total_correct_full += correct_predictions_full.sum().item()
                total_tokens_full += mask_full.sum().item()

            # For response-only accuracy, we only consider the response part
            # In our current setup, non-masked tokens in labels (not -100) are the response tokens
            mask_response = labels != -100  # -100 is used for ignored tokens in labels
            correct_predictions_response = (predictions == labels) & mask_response  # Compare with labels for response part only
            if is_original_batch:  # Only accumulate for original batches
                total_correct_response += correct_predictions_response.sum().item()
                total_tokens_response += mask_response.sum().item()

            if is_original_batch:
                processed_batches += 1

        if progress_bar:
            progress_bar.update(1)

    if progress_bar:
        progress_bar.close()

    model_engine.train()

    # Gather totals from all ranks
    device = model_engine.device
    local_correct_full = torch.tensor([total_correct_full], dtype=torch.float32, device=device)
    local_total_full = torch.tensor([total_tokens_full], dtype=torch.float32, device=device)
    local_correct_response = torch.tensor([total_correct_response], dtype=torch.float32, device=device)
    local_total_response = torch.tensor([total_tokens_response], dtype=torch.float32, device=device)

    if dist.is_initialized():
        dist.all_reduce(local_correct_full, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_total_full, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_correct_response, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_total_response, op=dist.ReduceOp.SUM)

    # Calculate accuracies
    if local_total_full.item() == 0:
        accuracy_full = 0.0  # Return 0 accuracy if no valid tokens
    else:
        accuracy_full = (local_correct_full / local_total_full).item()

    if local_total_response.item() == 0:
        accuracy_response = 0.0  # Return 0 accuracy if no valid tokens
    else:
        accuracy_response = (local_correct_response / local_total_response).item()

    # Print token counts on rank 0
    if rank == 0:
        print(f"Full Sequence - Correctly predicted tokens: {int(local_correct_full.item())}")
        print(f"Full Sequence - Total tokens to predict: {int(local_total_full.item())}")
        print(f"Full Sequence Accuracy: {accuracy_full:.6f}")
        print(f"Response Only - Correctly predicted tokens: {int(local_correct_response.item())}")
        print(f"Response Only - Total tokens to predict: {int(local_total_response.item())}")
        print(f"Response Only Accuracy: {accuracy_response:.6f}")

    return accuracy_full, accuracy_response


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
        baseline_accuracy_full, baseline_accuracy_response = calculate_accuracy(model_engine, test_dataloader, tokenizer)  # Limit batches for speed
        if dist.get_rank() == 0:
            print(f"Baseline full sequence accuracy before training: {baseline_accuracy_full:.4f}")
            print(f"Baseline response-only accuracy before training: {baseline_accuracy_response:.4f}")

    # Sample test examples for comparison
    # if dist.get_rank() == 0:
    #     print("Sampling test examples for comparison...")
    # test_samples = sample_test_examples(test_dataset, tokenizer, num_samples=10, seed=args.seed)
    #
    # # Generate outputs before training
    # # if dist.get_rank() == 0:
    # #     print("Generating outputs before training...")
    # # before_training_outputs = generate_model_outputs(model_engine, test_samples, tokenizer)

    # Skip sampling and generation for now to speed up training
    test_samples = []
    before_training_outputs = []

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
    # if dist.get_rank() == 0:
    #     print("Generating outputs after training...")
    # after_training_outputs = generate_model_outputs(model_engine, test_samples, tokenizer)
    #
    # # Format comparison output
    # # if dist.get_rank() == 0:
    # #     print("Formatting comparison output...")
    # # format_comparison_output(test_samples, before_training_outputs, after_training_outputs)

    # Skip comparison for now to speed up training
    print("Skipping comparison output for training run")

    # Perform gate output analysis after fine-tuning
    # if dist.get_rank() == 0:
    #     print("Performing gate output analysis after fine-tuning...")
    # gate_analysis_after = analyze_gate_outputs(model_engine, test_dataloader, tokenizer, step_name="after_training")

    # Compare gate analysis results
    # if dist.get_rank() == 0:
    #     print("Comparing gate output changes...")
    # gate_comparison = compare_gate_analysis(gate_analysis_before, gate_analysis_after)

    # Save model using DeepSpeed's save_checkpoint method
    #model_engine.save_checkpoint(f"{args.output_dir}{dist.get_rank()}")
    #tokenizer.save_pretrained(f"{args.output_dir}{dist.get_rank()}")
    #print("Model saved.")

    # Calculate final accuracy after training
    if args.calc_accuracy:
        if dist.get_rank() == 0:
            print("Calculating final accuracy after training...")
        final_accuracy_full, final_accuracy_response = calculate_accuracy(model_engine, test_dataloader, tokenizer)  # Limit batches for speed
        if dist.get_rank() == 0:
            print(f"Final full sequence accuracy after training: {final_accuracy_full:.4f}")
            print(f"Final response-only accuracy after training: {final_accuracy_response:.4f}")

        # Calculate accuracy improvement if baseline accuracy is available
        if 'baseline_accuracy_full' in locals() and dist.get_rank() == 0:
            print(f"Full sequence accuracy improvement: {final_accuracy_full - baseline_accuracy_full:.4f}")
        if 'baseline_accuracy_response' in locals() and dist.get_rank() == 0:
            print(f"Response-only accuracy improvement: {final_accuracy_response - baseline_accuracy_response:.4f}")


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
    parser.add_argument("--calc_accuracy", action="store_true", help="Calculate accuracy during evaluation (slower)")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    main(args)
