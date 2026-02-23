"""Example: Basic HuggingFace model loading and text generation.

This example shows how to use the MCP tools programmatically.
In practice, these are called by the AI agent through the MCP protocol.
"""

# Example tool calls that an MCP client would make:

# 1. Load a model via API
load_result = {
    "tool": "hf_load_model",
    "args": {
        "repo_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "backend": "api",
        "max_new_tokens": 256,
        "temperature": 0.7,
    }
}

# 2. Generate text
generate_result = {
    "tool": "hf_text_generate",
    "args": {
        "prompt": "Explain quantum computing in simple terms:",
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    }
}

# 3. Load embeddings
embed_result = {
    "tool": "hf_load_embeddings",
    "args": {
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "backend": "local",
    }
}

# 4. Generate embeddings
embeddings_result = {
    "tool": "hf_embed_texts",
    "args": {
        "texts_json": '["Hello world", "How are you?"]',
        "embed_type": "documents",
    }
}

print("See README.md for full tool documentation.")
print("These tools are meant to be called by an AI agent via MCP protocol.")
