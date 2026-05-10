#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Evaluate TruthfulQA MC2 using vLLM.

For each question, we compute the probability of each choice and report
the fraction of correct choices that receive the highest probability (MC2 accuracy).

Usage:
    python evaluate/truthfulqa/gen_truthfulqa.py --model moonshotai/Moonlight-16B-A3B --output eval_results/tqa_baseline --tp 2
"""

import argparse
import json
import os

os.environ.setdefault("VLLM_DISABLE_CUSTOM_ALL_REDUCE", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from datasets import load_dataset
from vllm import LLM, SamplingParams


def format_prompt(question):
    return f"### Instruction:\n{question}\n\n### Response:\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--tp", type=int, default=8, help="Tensor parallel size")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n_samples", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "samples.jsonl")

    print(f"Loading model: {args.model} (tp={args.tp})")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        disable_custom_all_reduce=True,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_new_tokens,
        logprobs=20,
    )

    print("Loading TruthfulQA dataset (multiple_choice)")
    dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation", trust_remote_code=True)
    print(f"Loaded {len(dataset)} examples")

    prompts = []
    meta = []
    for i, example in enumerate(dataset):
        question = example["question"]
        mc2 = example["mc2_targets"]
        choices = mc2["choices"]
        labels = mc2["labels"]
        prompt = format_prompt(question)
        for _ in range(args.n_samples):
            prompts.append(prompt)
            meta.append({
                "task_id": i,
                "question": question,
                "choices": choices,
                "labels": labels,
            })

    print(f"Generating {len(prompts)} completions")
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    with open(out_path, "w") as f_out:
        for m, output in zip(meta, outputs):
            completion = output.outputs[0].text.strip()
            logprobs_dict = {}
            if output.outputs[0].logprobs and output.outputs[0].logprobs[0]:
                for token, lp in output.outputs[0].logprobs[0].items():
                    logprobs_dict[token.token] = lp.logprob
            sample = {
                "task_id": m["task_id"],
                "question": m["question"],
                "choices": m["choices"],
                "labels": m["labels"],
                "raw_completion": completion,
                "first_token_logprobs": logprobs_dict,
            }
            f_out.write(json.dumps(sample) + "\n")

    print(f"Done. {len(outputs)} samples in {out_path}")


if __name__ == "__main__":
    main()
