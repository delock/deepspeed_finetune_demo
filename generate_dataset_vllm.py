#!/usr/bin/env python3
"""
使用vLLM生成训练数据集
"""

import argparse
import json
import asyncio
from tqdm import tqdm
import torch
from transformers import AutoTokenizer

def load_prompt_content(prompt_file):
    """加载prompt文件内容"""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read().strip()

def generate_training_data_vllm(teacher_model_name, prompt_file, num_samples, output_file):
    """使用vLLM根据prompt生成训练数据"""
    try:
        from vllm import LLM
        print(f"Using vLLM for dataset generation with model: {teacher_model_name}")

        # Create the LLM instance with multi-GPU support
        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        tensor_parallel_size = available_gpus  # Use all available GPUs for maximum parallelization

        llm = LLM(
            model=teacher_model_name,
            tensor_parallel_size=tensor_parallel_size,
            dtype="bfloat16",
            enforce_eager=True,  # Skip CUDA graph capture to reduce memory usage
            gpu_memory_utilization=0.9,  # Further reduce GPU memory utilization to avoid OOM
            max_model_len=2048,  # Further reduce max model length to reduce memory usage
            max_num_batched_tokens=4096  # Limit batched tokens to reduce memory usage
        )

        # Load the prompt
        prompt_content = load_prompt_content(prompt_file)

        print(f"Generating {num_samples} training samples using vLLM...")

        # Create prompts for generation
        prompts = []
        for i in range(min(num_samples, 100)):  # Limit to 100 samples per batch to reduce memory usage
            # Create a unique variation of the prompt for each sample
            prompt_variation = f"{prompt_content}\n\nSample #{i+1}: Generate a Python programming question and answer."
            prompts.append(prompt_variation)

        from vllm import SamplingParams

        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=128,  # Reduce max tokens to save memory
            stop=["\n\n", "###"]  # Stop at natural boundaries
        )

        # Generate responses using vLLM
        outputs = llm.generate(
            prompts,
            sampling_params=sampling_params
        )

        # Process the outputs
        generated_data = []
        tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text

            # Initialize default values
            instruction_part = f"Python programming question #{i+1}"
            input_part = ""
            output_part = generated_text

            # Try to parse the generated text to extract the actual problem, input, and solution
            if "Problem:" in generated_text and "Input:" in generated_text and "Output:" in generated_text and "'''" in generated_text:
                try:
                    # Split the text to extract different parts
                    # Format: "Problem: ... \nInput: ... \nOutput:\n'''\ncode\n'''"
                    text_parts = generated_text.split("Output:", 1)
                    if len(text_parts) > 1:
                        # Part before "Output:" contains Problem and Input
                        problem_input_part = text_parts[0]
                        # Part after "Output:" contains the code between triple quotes
                        output_section = text_parts[1]

                        # Extract problem and input from the first part
                        if "Input:" in problem_input_part:
                            prob_input_split = problem_input_part.split("Input:", 1)
                            instruction_part = "Problem:" + prob_input_split[0].split("Problem:", 1)[-1].strip()
                            input_part = prob_input_split[1].strip()
                        else:
                            # If no Input section, just extract Problem
                            if "Problem:" in problem_input_part:
                                instruction_part = "Problem:" + problem_input_part.split("Problem:", 1)[-1].strip()

                        # Extract code solution between triple quotes
                        if "'''" in output_section:
                            # Find the first and second occurrence of '''
                            first_quote = output_section.find("'''")
                            second_quote = output_section.find("'''", first_quote + 3)
                            if second_quote != -1:
                                output_part = output_section[first_quote + 3:second_quote].strip()
                            else:
                                # If only one closing ''', take from opening to end
                                output_part = output_section[first_quote + 3:].strip()
                        else:
                            # If no triple quotes, use the entire output section
                            output_part = output_section.strip()
                except:
                    # If parsing fails, use the original approach
                    instruction_part = f"Python programming question #{i+1}"
                    output_part = generated_text
            else:
                # If the format isn't as expected, use the full generated text as output
                instruction_part = f"Python programming question #{i+1}"
                output_part = generated_text

            # Create a sample with instruction, input, and output
            sample = {
                "instruction": instruction_part,
                "input": input_part,
                "output": output_part[:1000] if len(output_part) > 1000 else output_part  # Limit length
            }

            generated_data.append(sample)

        # If we need more samples than processed in the first batch, continue generating
        remaining_samples = num_samples - len(generated_data)
        while remaining_samples > 0:
            batch_size = min(remaining_samples, 100)  # Process in batches of 100
            additional_prompts = []
            for i in range(len(generated_data), len(generated_data) + batch_size):
                prompt_variation = f"{prompt_content}\n\nSample #{i+1}: Generate a Python programming question and answer."
                additional_prompts.append(prompt_variation)

            additional_outputs = llm.generate(
                additional_prompts,
                sampling_params=sampling_params
            )

            for i, output in enumerate(additional_outputs):
                generated_text = output.outputs[0].text

                # Initialize default values
                instruction_part = f"Python programming question #{len(generated_data) + 1}"
                input_part = ""
                output_part = generated_text

                # Try to parse the generated text to extract the actual problem, input, and solution
                if "Problem:" in generated_text and "Input:" in generated_text and "Output:" in generated_text and "'''" in generated_text:
                    try:
                        # Split the text to extract different parts
                        # Format: "Problem: ... \nInput: ... \nOutput:\n'''\ncode\n'''"
                        text_parts = generated_text.split("Output:", 1)
                        if len(text_parts) > 1:
                            # Part before "Output:" contains Problem and Input
                            problem_input_part = text_parts[0]
                            # Part after "Output:" contains the code between triple quotes
                            output_section = text_parts[1]

                            # Extract problem and input from the first part
                            if "Input:" in problem_input_part:
                                prob_input_split = problem_input_part.split("Input:", 1)
                                instruction_part = "Problem:" + prob_input_split[0].split("Problem:", 1)[-1].strip()
                                input_part = prob_input_split[1].strip()
                            else:
                                # If no Input section, just extract Problem
                                if "Problem:" in problem_input_part:
                                    instruction_part = "Problem:" + problem_input_part.split("Problem:", 1)[-1].strip()

                            # Extract code solution between triple quotes
                            if "'''" in output_section:
                                # Find the first and second occurrence of '''
                                first_quote = output_section.find("'''")
                                second_quote = output_section.find("'''", first_quote + 3)
                                if second_quote != -1:
                                    output_part = output_section[first_quote + 3:second_quote].strip()
                                else:
                                    # If only one closing ''', take from opening to end
                                    output_part = output_section[first_quote + 3:].strip()
                            else:
                                # If no triple quotes, use the entire output section
                                output_part = output_section.strip()
                    except:
                        # If parsing fails, use the original approach
                        instruction_part = f"Python programming question #{len(generated_data) + 1}"
                        output_part = generated_text
                else:
                    # If the format isn't as expected, use the full generated text as output
                    instruction_part = f"Python programming question #{len(generated_data) + 1}"
                    output_part = generated_text

                sample = {
                    "instruction": instruction_part,
                    "input": input_part,
                    "output": output_part[:1000] if len(output_part) > 1000 else output_part  # Limit length
                }
                generated_data.append(sample)

            remaining_samples -= batch_size

        print(f"Saving generated dataset to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(generated_data, f, ensure_ascii=False, indent=2)

        print(f"Generated {len(generated_data)} samples successfully!")

    except ImportError:
        print("vLLM not available, falling back to transformers generation...")
        # Fallback to the original method if vLLM is not available
        from transformers import AutoModelForCausalLM
        from datasets import Dataset

        tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(teacher_model_name, torch_dtype=torch.bfloat16)
        model.eval()
        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

        prompt_content = load_prompt_content(prompt_file)
        generated_data = []

        for i in tqdm(range(num_samples), desc="Generating samples"):
            # Create a unique variation of the prompt for each sample
            instruction_variation = f"{prompt_content}\n\nSample #{i+1}: Generate a Python programming question and answer."

            # Tokenize the instruction
            inputs = tokenizer(instruction_variation, return_tensors="pt", truncation=True, max_length=512)
            input_ids = inputs["input_ids"].to(model.device)

            # Generate response with faster parameters
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=128,  # Reduced from 256 to 128 for faster generation
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    num_beams=1,  # Use greedy decoding instead of beam search
                    early_stopping=False
                )

            # Decode the generated part only
            full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_part = full_output[len(instruction_variation):].strip()

            # Create a sample with instruction, input (empty), and output
            sample = {
                "instruction": f"Python programming question #{i+1}",
                "input": "",
                "output": response_part[:500] if len(response_part) > 500 else response_part  # Limit length
            }

            generated_data.append(sample)

        print(f"Saving generated dataset to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(generated_data, f, ensure_ascii=False, indent=2)

        print(f"Generated {len(generated_data)} samples successfully!")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"vLLM OOM error: {str(e)}, falling back to transformers generation...")
            # Fallback to transformers if vLLM runs out of memory
            from transformers import AutoModelForCausalLM
            from datasets import Dataset

            tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(teacher_model_name, torch_dtype=torch.bfloat16)
            model.eval()
            model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

            prompt_content = load_prompt_content(prompt_file)
            generated_data = []

            for i in tqdm(range(num_samples), desc="Generating samples"):
                # Create a unique variation of the prompt for each sample
                instruction_variation = f"{prompt_content}\n\nSample #{i+1}: Generate a Python programming question and answer."

                # Tokenize the instruction
                inputs = tokenizer(instruction_variation, return_tensors="pt", truncation=True, max_length=512)
                input_ids = inputs["input_ids"].to(model.device)

                # Generate response with faster parameters
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids,
                        max_new_tokens=128,  # Reduced from 256 to 128 for faster generation
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        num_beams=1,  # Use greedy decoding instead of beam search
                        early_stopping=False
                    )

                # Decode the generated part only
                full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response_part = full_output[len(instruction_variation):].strip()

                # Create a sample with instruction, input (empty), and output
                sample = {
                    "instruction": f"Python programming question #{i+1}",
                    "input": "",
                    "output": response_part[:500] if len(response_part) > 500 else response_part  # Limit length
                }

                generated_data.append(sample)

            print(f"Saving generated dataset to: {output_file}")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(generated_data, f, ensure_ascii=False, indent=2)

            print(f"Generated {len(generated_data)} samples successfully using transformers!")
        else:
            # Re-raise the error if it's not an OOM error
            raise e


def main():
    parser = argparse.ArgumentParser(description="Generate training dataset using vLLM and a prompt")
    parser.add_argument("--model_name", type=str, required=True, help="Teacher model name to use for generation")
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to the prompt file")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of samples to generate")
    parser.add_argument("--output_file", type=str, default="generated_dataset.json", help="Output file for generated dataset")
    
    args = parser.parse_args()
    
    generate_training_data_vllm(
        teacher_model_name=args.model_name,
        prompt_file=args.prompt_file,
        num_samples=args.num_samples,
        output_file=args.output_file
    )


if __name__ == "__main__":
    main()
