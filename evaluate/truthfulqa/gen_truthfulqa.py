#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Compute TruthfulQA MC2 log-likelihoods using vLLM prompt_logprobs.

For each question-choice pair, we build prompt = instruction + choice,
then use prompt_logprobs to get the logprob of every token conditioned on
all previous tokens.  The sum of logprobs for the choice tokens gives the
log-likelihood of that choice.  These are saved to a JSONL file for MC2
scoring in eval_truthfulqa.py.

Usage:
    python evaluate/truthfulqa/gen_truthfulqa.py \
        --model moonshotai/Moonlight-16B-A3B \
        --output eval_results/tqa_baseline --tp 2
"""

import argparse
import json
import math
import os

os.environ.setdefault("VLLM_DISABLE_CUSTOM_ALL_REDUCE", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from datasets import load_dataset
from vllm import LLM, SamplingParams

PROMPT_PREFIX = "### Instruction:\n"
PROMPT_SUFFIX = "\n\n### Response:\n"


def build_texts(question, choices):
    prefix = f"{PROMPT_PREFIX}{question}{PROMPT_SUFFIX}"
    return [prefix + c for c in choices], prefix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--tp", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "mc2_loglikes.jsonl")

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

    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0,
        prompt_logprobs=1,
    )

    print("Loading TruthfulQA dataset (multiple_choice)")
    dataset = load_dataset(
        "truthfulqa/truthful_qa",
        "multiple_choice",
        split="validation",
        trust_remote_code=True,
    )
    print(f"Loaded {len(dataset)} examples")

    results = []
    for i, example in enumerate(dataset):
        question = example["question"]
        mc2 = example["mc2_targets"]
        choices = mc2["choices"]
        labels = mc2["labels"]

        texts, prefix = build_texts(question, choices)
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        prefix_len = len(prefix_tokens)

        outputs = llm.generate(texts, sampling_params, use_tqdm=False)

        loglikes = []
        for j, output in enumerate(outputs):
            prompt_logprobs = output.prompt_logprobs
            if prompt_logprobs is None:
                loglikes.append(-math.inf)
                continue

            choice_tokens = tokenizer.encode(choices[j], add_special_tokens=False)
            n_choice = len(choice_tokens)

            total_ll = 0.0
            valid = 0
            for k in range(prefix_len, min(prefix_len + n_choice, len(prompt_logprobs))):
                token_lp = prompt_logprobs[k]
                if token_lp is None:
                    continue
                token_id = choice_tokens[k - prefix_len]
                if token_id in token_lp:
                    total_ll += token_lp[token_id].logprob
                    valid += 1
                else:
                    total_ll += math.log(1e-10)
                    valid += 1

            loglikes.append(total_ll if valid > 0 else -math.inf)

        results.append({
            "task_id": i,
            "question": question,
            "choices": choices,
            "labels": labels,
            "loglikes": loglikes,
        })

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(dataset)} questions")

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Done. {len(results)} questions -> {out_path}")


if __name__ == "__main__":
    main()
