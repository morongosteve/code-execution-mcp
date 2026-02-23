"""Example: Loading PEFT/LoRA adapters with quantization.

Shows the sequence of MCP tool calls for loading base models with LoRA adapters,
including 4-bit and 8-bit quantization options.
"""

# Example 1: Load a PEFT adapter with 4-bit quantization
load_4bit = {
    "tool": "hf_load_peft_model",
    "args": {
        "base_model": "meta-llama/Meta-Llama-3-8B",
        "adapter_id": "my-org/llama3-lora-adapter",
        "quantize": "4bit",
        "max_new_tokens": 256,
    }
}

# Example 2: Load a PEFT adapter with 8-bit quantization
load_8bit = {
    "tool": "hf_load_peft_model",
    "args": {
        "base_model": "mistralai/Mistral-7B-v0.1",
        "adapter_id": "my-org/mistral-lora-finetuned",
        "quantize": "8bit",
        "max_new_tokens": 512,
    }
}

# Example 3: Load without quantization (full precision)
load_full = {
    "tool": "hf_load_peft_model",
    "args": {
        "base_model": "meta-llama/Meta-Llama-3-8B",
        "adapter_id": "my-org/llama3-lora-adapter",
    }
}

# Example 4: Generate text with the loaded PEFT model
generate = {
    "tool": "hf_text_generate",
    "args": {
        "prompt": "Summarize the key benefits of transfer learning:",
        "model": "meta-llama/Meta-Llama-3-8B",
    }
}

# Example 5: Fine-tune a new LoRA adapter
finetune = {
    "tool": "hf_finetune",
    "args": {
        "model_name": "meta-llama/Meta-Llama-3-8B",
        "dataset_name": "tatsu-lab/alpaca",
        "output_dir": "./my-lora-adapter",
        "num_epochs": 1,
        "learning_rate": 2e-4,
    }
}

# Example 6: Benchmark the model
benchmark = {
    "tool": "hf_benchmark",
    "args": {
        "model": "meta-llama/Meta-Llama-3-8B",
        "num_runs": 10,
    }
}

print("PEFT/LoRA Adapter Example")
print("=" * 50)
print("Quantization options:")
print("  - 4bit: Lowest memory usage, slight quality trade-off")
print("  - 8bit: Balanced memory and quality")
print("  - None: Full precision, highest quality, most memory")
print()
print("Workflow: Load Base + Adapter -> Generate -> Benchmark")
print("See README.md for full documentation.")
