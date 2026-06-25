"""Example: Image generation using HuggingFace diffusion models.

Shows the sequence of MCP tool calls for generating images via API or locally.
"""

# Example 1: Generate an image via HuggingFace Inference API
api_image = {
    "tool": "hf_generate_image",
    "args": {
        "prompt": "A futuristic city at sunset, digital art, highly detailed",
        "backend": "api",
    },
}

# Example 2: Generate an image locally with a specific model
local_image = {
    "tool": "hf_generate_image",
    "args": {
        "prompt": "A cat wearing a top hat, watercolor painting style",
        "backend": "local",
        "model": "stabilityai/stable-diffusion-xl-base-1.0",
        "output_path": "cat_tophat.png",
    },
}

# Example 3: Generate with FLUX model via API
flux_image = {
    "tool": "hf_generate_image",
    "args": {
        "prompt": "An astronaut riding a horse on Mars, photorealistic",
        "backend": "api",
        "model": "black-forest-labs/FLUX.1-schnell",
    },
}

# Example 4: Check GPU status before local generation
gpu_check = {"tool": "hf_gpu_status", "args": {}}

print("Image Generation Example")
print("=" * 50)
print("API backend: No GPU required, uses HuggingFace Inference API")
print("Local backend: Requires GPU with sufficient VRAM (8GB+ recommended)")
print()
print("Supported models include:")
print("  - stabilityai/stable-diffusion-xl-base-1.0 (SDXL)")
print("  - black-forest-labs/FLUX.1-schnell (FLUX)")
print("  - runwayml/stable-diffusion-v1-5 (SD 1.5)")
print()
print("See README.md for full documentation.")
