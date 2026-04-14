#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate HumanEval completions using HuggingFace transformers (single GPU).
Outputs evalplus-compatible JSONL.

Usage:
    # Baseline (code completion, no instruction wrapping):
    python gen_humaneval.py --model moonshotai/Moonlight-16B-A3B --output evalplus_results/baseline

    # Fine-tuned model (Alpaca instruction format):
    python gen_humaneval.py --model hf_model_muon --output evalplus_results/muon --instruction
"""

import argparse
import json
import os

import torch
from evalplus.data import get_human_eval_plus
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_completion(model, tokenizer, prompt, max_new_tokens=512, temperature=0.0):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            top_p=0.95 if temperature > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Only decode the newly generated tokens
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    completion = tokenizer.decode(generated, skip_special_tokens=True)
    # Stop at common end markers
    for stop in ["\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint("]:
        if stop in completion:
            completion = completion[:completion.index(stop)]
    return completion


def wrap_instruction_prompt(code_prompt):
    """Wrap a HumanEval code prompt in Alpaca instruction format.

    The code_prompt (function signature + docstring) is included both in the
    instruction (so the model knows what to complete) and as the start of the
    response (so the model continues writing the function body).
    """
    instruction = f"### Instruction:\nComplete the following Python function.\n\n{code_prompt}\n\n### Response:\n{code_prompt}"
    return instruction


def load_existing_samples(out_path):
    """Load already-generated samples to support resuming interrupted runs."""
    done = {}
    if not os.path.exists(out_path):
        return done
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            task_id = sample["task_id"]
            done.setdefault(task_id, []).append(sample["completion"])
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument(
        "--instruction",
        action="store_true",
        help="Wrap prompts in Alpaca instruction format (for fine-tuned models)",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "samples.jsonl")

    # Resume support: skip task_ids already written
    done = load_existing_samples(out_path)
    if done:
        print(f"Resuming: {len(done)} task(s) already done, skipping them.")

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    print("Loading HumanEval+ dataset")
    problems = get_human_eval_plus()

    mode = "instruction" if args.instruction else "completion"
    print(f"Generation mode: {mode}")

    # Append to existing file so intermediate results are never lost
    total_written = sum(len(v) for v in done.values())
    with open(out_path, "a") as f_out:
        for task_id, problem in tqdm(problems.items(), desc="Generating"):
            if task_id in done and len(done[task_id]) >= args.n_samples:
                continue
            code_prompt = problem["prompt"]
            if args.instruction:
                prompt = wrap_instruction_prompt(code_prompt)
            else:
                prompt = code_prompt
            already = len(done.get(task_id, []))
            for _ in range(args.n_samples - already):
                completion = generate_completion(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
                sample = dict(task_id=task_id, completion=completion)
                f_out.write(json.dumps(sample) + "\n")
                f_out.flush()
                total_written += 1

    print(f"Done. {total_written} samples in {out_path}")


if __name__ == "__main__":
    main()
