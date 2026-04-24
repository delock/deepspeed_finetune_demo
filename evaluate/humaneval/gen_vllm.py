#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate HumanEval/MBPP completions using vLLM (multi-GPU tensor parallel).
Outputs evalplus-compatible JSONL.

Usage:
    python gen_humaneval_vllm.py --model moonshotai/Moonlight-16B-A3B --output evalplus_results/baseline --tp 8
    python gen_humaneval_vllm.py --model moonshotai/Moonlight-16B-A3B --output evalplus_results/baseline --tp 8 --dataset mbpp
"""

import argparse
import json
import os

os.environ.setdefault("VLLM_DISABLE_CUSTOM_ALL_REDUCE", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from evalplus.data import get_human_eval_plus, get_mbpp_plus
from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="humaneval", choices=["humaneval", "mbpp"])
    parser.add_argument("--tp", type=int, default=8, help="Tensor parallel size")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument(
        "--instruction",
        action="store_true",
        help="Wrap prompts in Alpaca instruction format",
    )
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
        enforce_eager=True,  # disable CUDA graphs to avoid worker crashes
        disable_custom_all_reduce=True,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature if args.temperature > 0 else 0,
        max_tokens=args.max_new_tokens,
        top_p=0.95 if args.temperature > 0 else 1.0,
        stop=["\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint("],
    )

    print(f"Loading {args.dataset}+ dataset")
    if args.dataset == "humaneval":
        problems = get_human_eval_plus()
    else:
        problems = get_mbpp_plus()

    # Build prompts
    prompts = []
    task_ids = []
    for task_id, problem in problems.items():
        if args.dataset == "humaneval":
            code_prompt = problem["prompt"]
            if args.instruction:
                prompt = (
                    f"### Instruction:\nComplete the following Python function.\n\n"
                    f"{code_prompt}\n\n### Response:\n{code_prompt}"
                )
            else:
                prompt = code_prompt
        else:  # mbpp
            description = problem["prompt"]
            code_prompt = problem.get("prompt", "")
            if args.instruction:
                prompt = (
                    f"### Instruction:\n{description}\n\n### Response:\n"
                )
            else:
                prompt = description + "\n"
        for _ in range(args.n_samples):
            prompts.append(prompt)
            task_ids.append(task_id)

    mode = "instruction" if args.instruction else "completion"
    print(f"Generation mode: {mode}, {len(prompts)} prompts")

    # Generate all at once - vllm handles batching internally
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    # Write results
    with open(out_path, "w") as f_out:
        for task_id, output in zip(task_ids, outputs):
            completion = output.outputs[0].text
            sample = dict(task_id=task_id, completion=completion)
            f_out.write(json.dumps(sample) + "\n")

    print(f"Done. {len(outputs)} samples in {out_path}")


if __name__ == "__main__":
    main()
