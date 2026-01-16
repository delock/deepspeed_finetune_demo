#!/usr/bin/env python3
"""
使用teacher模型和prompt文件生成训练数据集
"""

import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def load_prompt_content(prompt_file):
    """加载prompt文件内容"""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read().strip()

def generate_training_data(teacher_model_name, prompt_file, num_samples, output_file):
    """使用teacher模型根据prompt生成训练数据"""
    print(f"Loading teacher model: {teacher_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name, 
        torch_dtype=torch.bfloat16,
        device_map="auto"  # Automatically use available GPUs
    )
    
    model.eval()
    
    print(f"Loading prompt from: {prompt_file}")
    prompt_content = load_prompt_content(prompt_file)
    
    print(f"Generating {num_samples} training samples...")
    generated_data = []
    
    # For demonstration purposes, we'll create a simple generation loop
    # In practice, this would involve more sophisticated prompting and sampling
    
    # Create a simple prompt template based on the provided prompt
    base_instruction = prompt_content
    
    for i in tqdm(range(num_samples), desc="Generating samples"):
        # Create a unique variation of the prompt for each sample
        instruction_variation = f"{base_instruction}\n\nSample #{i+1}: Generate a Python programming question and answer."
        
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
        
        # Decode the full output
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Try to extract instruction and response parts
        # This is a simplified approach - in practice, you'd need more sophisticated parsing
        if "\n\nSample #" in full_output:
            # Extract the response part after the prompt
            response_part = full_output.split("\n\nSample #")[1]
            response_part = response_part.split(":", 1)[1].strip() if ":" in response_part else response_part
            
            # Further split to get just the generated content
            if "\n\n" in response_part:
                response_part = response_part.split("\n\n", 1)[0]
        else:
            response_part = full_output[len(instruction_variation):].strip()
        
        # Create a sample with instruction, input (empty), and output
        sample = {
            "instruction": f"Python programming question #{i+1}",
            "input": "",  # Could be populated with specific code snippets or scenarios
            "output": response_part[:500] if len(response_part) > 500 else response_part  # Limit length
        }
        
        generated_data.append(sample)
    
    print(f"Saving generated dataset to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(generated_data, f, ensure_ascii=False, indent=2)
    
    print(f"Generated {len(generated_data)} samples successfully!")

def main():
    parser = argparse.ArgumentParser(description="Generate training dataset using a teacher model and prompt")
    parser.add_argument("--model_name", type=str, required=True, help="Teacher model name to use for generation")
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to the prompt file")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of samples to generate")
    parser.add_argument("--output_file", type=str, default="generated_dataset.json", help="Output file for generated dataset")
    
    args = parser.parse_args()
    
    generate_training_data(
        teacher_model_name=args.model_name,
        prompt_file=args.prompt_file,
        num_samples=args.num_samples,
        output_file=args.output_file
    )

if __name__ == "__main__":
    main()