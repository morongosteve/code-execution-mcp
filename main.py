#!/usr/bin/env python3
import os
import sys
from fastmcp import FastMCP
from code_execution_tool import CodeExecutionTool
from huggingface_tools import (
    list_models,
    load_hf_model,
    load_hf_chat_model,
    load_hf_embeddings,
    hf_generate,
    hf_chat,
    hf_embed,
    unload_model,
    hf_pipeline_task,
    hf_dataset_info,
)

# Helpers for debugging
version = f"v0.2.0, Python {sys.version.split(' ')[0]}, executable={sys.executable}"
os.environ["CODE_EXEC_MCP_VERSION"] = version

# Initialize FastMCP server
mcp = FastMCP(
    "code-execution-mcp",
    instructions=(
        f"Execute terminal commands, Python code, and HuggingFace model operations "
        f"on the host system. Includes LangChain-based HuggingFace integration for "
        f"text generation, chat, embeddings, and pipeline tasks. Version: {version}"
    ),
    version=version
)

# Get configuration from environment variables with defaults
EXECUTABLE = os.getenv("CODE_EXEC_EXECUTABLE", "")
INIT_COMMANDS_STR = os.getenv("CODE_EXEC_INIT_COMMANDS", "")
INIT_COMMANDS = [cmd.strip() for cmd in INIT_COMMANDS_STR.split(";") if cmd.strip()] if INIT_COMMANDS_STR else []

# Timeout configuration
FIRST_OUTPUT_TIMEOUT = int(os.getenv("CODE_EXEC_FIRST_OUTPUT_TIMEOUT", "30"))
BETWEEN_OUTPUT_TIMEOUT = int(os.getenv("CODE_EXEC_BETWEEN_OUTPUT_TIMEOUT", "15"))
DIALOG_TIMEOUT = int(os.getenv("CODE_EXEC_DIALOG_TIMEOUT", "5"))
MAX_EXEC_TIMEOUT = int(os.getenv("CODE_EXEC_MAX_EXEC_TIMEOUT", "180"))

# Create single CodeExecutionTool instance for state management
# This preserves Agent Zero's pattern of maintaining shell sessions across calls
code_tool = CodeExecutionTool(
    executable=EXECUTABLE,
    init_commands=INIT_COMMANDS,
    first_output_timeout=FIRST_OUTPUT_TIMEOUT,
    between_output_timeout=BETWEEN_OUTPUT_TIMEOUT,
    dialog_timeout=DIALOG_TIMEOUT,
    max_exec_timeout=MAX_EXEC_TIMEOUT
)


@mcp.tool()
async def execute_terminal(command: str, session: int = 0) -> str:
    try:
        result = await code_tool.execute_terminal_command(session=session, command=command)
        return result or "[No output]"
    except Exception as e:
        return f"Error executing terminal command: {str(e)}"


@mcp.tool()
async def execute_python(code: str, session: int = 0) -> str:
    try:
        result = await code_tool.execute_python_code(session=session, code=code)
        return result or "[No output]"
    except Exception as e:
        return f"Error executing Python code: {str(e)}"


@mcp.tool()
async def get_output(session: int = 0) -> str:
    try:
        result = await code_tool.get_terminal_output(session=session)
        return result or "[No output]"
    except Exception as e:
        return f"Error getting terminal output: {str(e)}"


@mcp.tool()
async def reset_terminal(session: int = 0, reason: str | None = None) -> str:
    try:
        result = await code_tool.reset_terminal(session=session, reason=reason)
        return result or "[No output]"
    except Exception as e:
        return f"Error resetting terminal: {str(e)}"


# --- HuggingFace + LangChain integration tools ---

@mcp.tool()
async def hf_list_models() -> str:
    """List all currently loaded HuggingFace models and embeddings."""
    return await list_models()


@mcp.tool()
async def hf_load_model(
    repo_id: str,
    task: str = "text-generation",
    backend: str = "api",
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    alias: str = "",
) -> str:
    """Load a HuggingFace model via LangChain for text generation.

    Args:
        repo_id: HuggingFace model repository ID (e.g. 'meta-llama/Meta-Llama-3-8B-Instruct')
        task: Model task type ('text-generation', 'text2text-generation', 'summarization')
        backend: 'api' for HuggingFace Inference API (remote), 'local' for local pipeline
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        alias: Optional alias for referencing the model later
    """
    return await load_hf_model(repo_id, task, backend, max_new_tokens, temperature, alias)


@mcp.tool()
async def hf_load_chat_model(
    repo_id: str,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    alias: str = "",
) -> str:
    """Load a HuggingFace model as a chat model with proper chat template handling.

    Uses LangChain's ChatHuggingFace wrapper for message formatting.

    Args:
        repo_id: HuggingFace model repository ID (e.g. 'meta-llama/Meta-Llama-3-8B-Instruct')
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        alias: Optional alias for referencing the model later
    """
    return await load_hf_chat_model(repo_id, max_new_tokens, temperature, alias)


@mcp.tool()
async def hf_load_embeddings(
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    backend: str = "local",
    alias: str = "",
) -> str:
    """Load a HuggingFace embedding model via LangChain.

    Args:
        model_name: HuggingFace embedding model name
        backend: 'local' for sentence-transformers, 'api' for HF Inference API
        alias: Optional alias for referencing the model later
    """
    return await load_hf_embeddings(model_name, backend, alias)


@mcp.tool()
async def hf_text_generate(
    prompt: str,
    model: str = "",
    max_new_tokens: int = 0,
    temperature: float = 0.0,
) -> str:
    """Generate text using a loaded HuggingFace model.

    Args:
        prompt: The input prompt for text generation
        model: Model alias or repo_id. If empty, uses the first loaded model
        max_new_tokens: Override max tokens (0 = model default)
        temperature: Override temperature (0.0 = model default)
    """
    return await hf_generate(prompt, model, max_new_tokens, temperature)


@mcp.tool()
async def hf_chat_complete(
    messages_json: str,
    model: str = "",
) -> str:
    """Chat with a loaded HuggingFace model using structured messages.

    Args:
        messages_json: JSON array of message objects.
            Example: [{"role": "user", "content": "Hello!"}]
            Roles: 'system', 'user', 'assistant'
        model: Model alias or repo_id. If empty, uses the first loaded model
    """
    return await hf_chat(messages_json, model)


@mcp.tool()
async def hf_embed_texts(
    texts_json: str,
    model: str = "",
    embed_type: str = "documents",
) -> str:
    """Generate embeddings using a loaded HuggingFace embedding model.

    Args:
        texts_json: JSON array of text strings. Example: ["Hello world"]
        model: Model alias or name. If empty, uses the first loaded model
        embed_type: 'documents' or 'query' (query optimized for single text search)
    """
    return await hf_embed(texts_json, model, embed_type)


@mcp.tool()
async def hf_unload_model(model: str, model_type: str = "llm") -> str:
    """Unload a previously loaded model to free resources.

    Args:
        model: Alias or name of the model to unload
        model_type: 'llm' for language models, 'embedding' for embedding models
    """
    return await unload_model(model, model_type)


@mcp.tool()
async def hf_run_pipeline(
    task: str,
    input_text: str,
    model: str = "",
) -> str:
    """Run a HuggingFace transformers pipeline task directly.

    Supports: sentiment-analysis, summarization, translation_en_to_fr,
    text-classification, ner, question-answering, fill-mask,
    zero-shot-classification, and more.

    Args:
        task: The pipeline task name
        input_text: The input text to process
        model: Optional model ID. If empty, uses the default model for the task
    """
    return await hf_pipeline_task(task, input_text, model)


@mcp.tool()
async def hf_preview_dataset(
    dataset_name: str,
    split: str = "train",
    num_rows: int = 3,
) -> str:
    """Load and preview a HuggingFace dataset.

    Args:
        dataset_name: Dataset identifier (e.g. 'rajpurkar/squad')
        split: Dataset split ('train', 'test', 'validation')
        num_rows: Number of example rows to return
    """
    return await hf_dataset_info(dataset_name, split, num_rows)


def main():
    # Run with stdio transport (default)
    mcp.run()


if __name__ == "__main__":
    main()
