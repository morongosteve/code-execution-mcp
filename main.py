#!/usr/bin/env python3
import os
import sys

from fastmcp import FastMCP

from code_execution_tool import CodeExecutionTool
from huggingface_tools import (
    generate_image,
    gpu_status,
    hf_audio_transcribe,
    hf_batch_generate,
    hf_benchmark,
    hf_chat,
    hf_dataset_info,
    hf_embed,
    hf_finetune,
    hf_generate,
    hf_pipeline_task,
    hf_text_to_speech,
    list_models,
    load_hf_chat_model,
    load_hf_embeddings,
    load_hf_model,
    load_peft_model,
    load_vllm_model,
    model_download_status,
    rag_query,
    restore_registry_state,
    save_registry_state,
    setup_rag_pipeline,
    unload_model,
)
from security import (
    audit,
    execution_limiter,
    model_load_limiter,
    redact_secrets,
    resource_limits,
    sanitize_model_id,
    scan_output_for_secrets,
    validate_command,
)

# Helpers for debugging
version = f"v0.3.0, Python {sys.version.split(' ')[0]}, executable={sys.executable}"
os.environ["CODE_EXEC_MCP_VERSION"] = version

# Initialize FastMCP server
mcp = FastMCP(
    "code-execution-mcp",
    instructions=(
        f"Execute terminal commands, Python code, and HuggingFace model operations "
        f"on the host system. Includes LangChain-based HuggingFace integration for "
        f"text generation, chat, embeddings, pipeline tasks, LoRA/PEFT adapters, "
        f"image generation via Diffusers, and RAG pipelines. Version: {version}"
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
        # Enforce input size limit
        if len(command.encode()) > resource_limits.MAX_FILE_SIZE:
            return f"Command exceeds maximum allowed size ({resource_limits.MAX_FILE_SIZE // (1024*1024)}MB)."

        # Rate limiting
        allowed, msg = execution_limiter.check()
        if not allowed:
            audit.log_rate_limit("execution_limiter")
            return f"Rate limit: {msg}"

        # Audit log
        audit.log_command(session=session, command=command)

        # Validate (warn only, do not block)
        is_safe, warning = validate_command(command)
        if not is_safe:
            audit.log_warning(f"Dangerous command in session {session}: {warning}")

        result = await code_tool.execute_terminal_command(session=session, command=command)
        result = result or code_tool.read_prompt("fw.code.no_output.md")

        # Truncate oversized output
        if len(result) > resource_limits.MAX_OUTPUT_SIZE:
            result = result[:resource_limits.MAX_OUTPUT_SIZE] + "\n[Output truncated]"

        # Scan and redact secrets
        secret_warnings = scan_output_for_secrets(result)
        if secret_warnings:
            audit.log_warning(f"Secrets detected in output: {'; '.join(secret_warnings)}")
            result = redact_secrets(result)

        return result
    except Exception as e:
        return f"Error executing terminal command: {str(e)}"


@mcp.tool()
async def execute_python(code: str, session: int = 0) -> str:
    try:
        # Enforce input size limit
        if len(code.encode()) > resource_limits.MAX_FILE_SIZE:
            return f"Code exceeds maximum allowed size ({resource_limits.MAX_FILE_SIZE // (1024*1024)}MB)."

        # Rate limiting
        allowed, msg = execution_limiter.check()
        if not allowed:
            audit.log_rate_limit("execution_limiter")
            return f"Rate limit: {msg}"

        # Audit log
        audit.log_command(session=session, command=f"[python] {code[:200]}")

        result = await code_tool.execute_python_code(session=session, code=code)
        result = result or code_tool.read_prompt("fw.code.no_output.md")

        # Truncate oversized output
        if len(result) > resource_limits.MAX_OUTPUT_SIZE:
            result = result[:resource_limits.MAX_OUTPUT_SIZE] + "\n[Output truncated]"

        # Scan and redact secrets
        secret_warnings = scan_output_for_secrets(result)
        if secret_warnings:
            audit.log_warning(f"Secrets detected in output: {'; '.join(secret_warnings)}")
            result = redact_secrets(result)

        return result
    except Exception as e:
        return f"Error executing Python code: {str(e)}"


@mcp.tool()
async def get_output(session: int = 0) -> str:
    try:
        result = await code_tool.get_terminal_output(session=session)
        return result or code_tool.read_prompt("fw.code.no_output.md")
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
    """List all currently loaded HuggingFace models, embeddings, and RAG pipelines.

    Shows TTL remaining, backend type, age, and registry capacity limits.
    """
    return await list_models()


@mcp.tool()
async def hf_load_model(
    repo_id: str,
    task: str = "text-generation",
    backend: str = "api",
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    alias: str = "",
    ttl: int = 0,
    quantize: str = "",
) -> str:
    """Load a HuggingFace model via LangChain for text generation.

    Models are auto-evicted after TTL expires. Max concurrent models enforced.

    Args:
        repo_id: HuggingFace model repository ID (e.g. 'meta-llama/Meta-Llama-3-8B-Instruct')
        task: Model task type ('text-generation', 'text2text-generation', 'summarization')
        backend: 'api' for HuggingFace Inference API (remote), 'local' for local pipeline
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        alias: Optional alias for referencing the model later
        ttl: Time-to-live in seconds (0 = default 3600s). Model auto-evicts after this.
        quantize: Quantization mode for local backend: '4bit', '8bit', or '' for none
    """
    # Validate model ID
    valid, err = sanitize_model_id(repo_id)
    if not valid:
        return f"Invalid model ID: {err}"

    # Rate limiting
    allowed, msg = model_load_limiter.check()
    if not allowed:
        audit.log_rate_limit("model_load_limiter")
        return f"Rate limit: {msg}"

    # Audit log
    audit.log_model_load(model_id=repo_id, backend=backend)

    return await load_hf_model(repo_id, task, backend, max_new_tokens, temperature, alias, ttl, quantize)


@mcp.tool()
async def hf_load_chat_model(
    repo_id: str,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    alias: str = "",
    ttl: int = 0,
) -> str:
    """Load a HuggingFace model as a chat model with proper chat template handling.

    Uses LangChain's ChatHuggingFace wrapper for message formatting.

    Args:
        repo_id: HuggingFace model repository ID (e.g. 'meta-llama/Meta-Llama-3-8B-Instruct')
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        alias: Optional alias for referencing the model later
        ttl: Time-to-live in seconds (0 = default 3600s)
    """
    # Validate model ID
    valid, err = sanitize_model_id(repo_id)
    if not valid:
        return f"Invalid model ID: {err}"

    # Rate limiting
    allowed, msg = model_load_limiter.check()
    if not allowed:
        audit.log_rate_limit("model_load_limiter")
        return f"Rate limit: {msg}"

    # Audit log
    audit.log_model_load(model_id=repo_id, backend="chat")

    return await load_hf_chat_model(repo_id, max_new_tokens, temperature, alias, ttl)


@mcp.tool()
async def hf_load_embeddings(
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    backend: str = "local",
    alias: str = "",
    ttl: int = 0,
) -> str:
    """Load a HuggingFace embedding model via LangChain.

    Args:
        model_name: HuggingFace embedding model name
        backend: 'local' for sentence-transformers, 'api' for HF Inference API
        alias: Optional alias for referencing the model later
        ttl: Time-to-live in seconds (0 = default 3600s)
    """
    # Validate model ID
    valid, err = sanitize_model_id(model_name)
    if not valid:
        return f"Invalid model ID: {err}"

    # Rate limiting
    allowed, msg = model_load_limiter.check()
    if not allowed:
        audit.log_rate_limit("model_load_limiter")
        return f"Rate limit: {msg}"

    # Audit log
    audit.log_model_load(model_id=model_name, backend=backend)

    return await load_hf_embeddings(model_name, backend, alias, ttl)


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

    Explicitly frees GPU/CPU memory and runs garbage collection.

    Args:
        model: Alias or name of the model to unload
        model_type: 'llm' for language models, 'embedding' for embedding models, 'all' to clear everything
    """
    audit.log_model_unload(model_id=model)
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
    num_rows: int = 5,
) -> str:
    """Preview a HuggingFace dataset safely using streaming mode.

    Handles gated/auth-required datasets with clear error messages.
    Truncates large field values. Max 50 rows per preview.

    Args:
        dataset_name: Dataset identifier (e.g. 'rajpurkar/squad')
        split: Dataset split ('train', 'test', 'validation')
        num_rows: Number of example rows to return (default: 5, max: 50)
    """
    return await hf_dataset_info(dataset_name, split, num_rows)


# --- LoRA / PEFT adapter support ---

@mcp.tool()
async def hf_load_peft_model(
    base_model_id: str,
    adapter_id: str,
    task: str = "text-generation",
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    alias: str = "",
    ttl: int = 0,
    quantize: str = "",
) -> str:
    """Load a base model with a LoRA/PEFT adapter for fine-tuned inference.

    Supports 4-bit and 8-bit quantization via bitsandbytes.

    Args:
        base_model_id: Base HuggingFace model ID (e.g. 'meta-llama/Llama-2-7b-hf')
        adapter_id: PEFT adapter ID from HuggingFace Hub or local path
        task: Model task type (default: 'text-generation')
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        alias: Optional alias for the combined model
        ttl: Time-to-live in seconds (0 = default 3600s)
        quantize: '4bit', '8bit', or '' for no quantization
    """
    # Validate model IDs
    valid, err = sanitize_model_id(base_model_id)
    if not valid:
        return f"Invalid base model ID: {err}"
    valid, err = sanitize_model_id(adapter_id)
    if not valid:
        return f"Invalid adapter ID: {err}"

    # Rate limiting
    allowed, msg = model_load_limiter.check()
    if not allowed:
        audit.log_rate_limit("model_load_limiter")
        return f"Rate limit: {msg}"

    # Audit log
    audit.log_model_load(model_id=f"{base_model_id}+{adapter_id}", backend="peft")

    return await load_peft_model(
        base_model_id, adapter_id, task, max_new_tokens,
        temperature, alias, ttl, quantize
    )


# --- Diffusers image generation ---

@mcp.tool()
async def hf_generate_image(
    prompt: str,
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
    negative_prompt: str = "",
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    width: int = 1024,
    height: int = 1024,
    output_path: str = "generated_image.png",
    backend: str = "api",
) -> str:
    """Generate an image from a text prompt using Diffusers or HF Inference API.

    Supports Stable Diffusion XL, FLUX, and other text-to-image models.

    Args:
        prompt: Text description of the image to generate
        model_id: Diffusion model ID (e.g. 'stabilityai/stable-diffusion-xl-base-1.0',
            'black-forest-labs/FLUX.1-dev')
        negative_prompt: What to avoid in the generated image
        num_inference_steps: Denoising steps (default: 30). Higher = better quality
        guidance_scale: Prompt adherence strength (default: 7.5)
        width: Image width in pixels (default: 1024)
        height: Image height in pixels (default: 1024)
        output_path: Where to save the image (default: 'generated_image.png')
        backend: 'api' for HF Inference API, 'local' for local diffusers pipeline
    """
    return await generate_image(
        prompt, model_id, negative_prompt, num_inference_steps,
        guidance_scale, width, height, output_path, backend
    )


# --- RAG pipeline tools ---

@mcp.tool()
async def hf_setup_rag(
    documents_json: str,
    embedding_model: str = "",
    llm_model: str = "",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    search_type: str = "similarity",
    search_k: int = 4,
) -> str:
    """Set up a Retrieval-Augmented Generation pipeline.

    Ingests documents, chunks them, creates embeddings in FAISS vector store,
    and configures a LangChain RetrievalQA chain.

    Requires both an LLM and embedding model to be loaded first.

    Args:
        documents_json: JSON array of document strings to ingest
        embedding_model: Alias of loaded embedding model (empty = first available)
        llm_model: Alias of loaded LLM (empty = first available)
        chunk_size: Characters per text chunk (default: 500)
        chunk_overlap: Overlap between chunks (default: 50)
        search_type: 'similarity' or 'mmr' (default: 'similarity')
        search_k: Number of documents to retrieve per query (default: 4)
    """
    return await setup_rag_pipeline(
        documents_json, embedding_model, llm_model,
        chunk_size, chunk_overlap, search_type, search_k
    )


@mcp.tool()
async def hf_rag_query(
    query: str,
    pipeline: str = "",
) -> str:
    """Query a previously created RAG pipeline.

    Returns the answer along with source document snippets used.

    Args:
        query: The question to answer using retrieved context
        pipeline: RAG pipeline key (empty = first available)
    """
    return await rag_query(query, pipeline)


# --- Batch Inference ---

@mcp.tool()
async def hf_batch_text_generate(
    prompts_json: str,
    model: str = "",
    max_new_tokens: int = 0,
    temperature: float = 0.0,
) -> str:
    """Generate text for multiple prompts in a single batch call.

    More efficient than calling hf_text_generate multiple times.

    Args:
        prompts_json: JSON array of prompt strings.
            Example: '["Tell me a joke", "Write a haiku", "Explain gravity"]'
        model: Model alias or repo_id. If empty, uses the first loaded model
        max_new_tokens: Override max tokens (0 = model default)
        temperature: Override temperature (0.0 = model default)
    """
    return await hf_batch_generate(prompts_json, model, max_new_tokens, temperature)


# --- Fine-tuning ---

@mcp.tool()
async def hf_finetune_model(
    base_model_id: str,
    dataset_name: str,
    output_dir: str = "./finetuned_model",
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    batch_size: int = 4,
    max_seq_length: int = 512,
) -> str:
    """Fine-tune a model using LoRA/PEFT on a HuggingFace dataset.

    Uses SFTTrainer from trl for supervised fine-tuning with LoRA adapters.
    The fine-tuned adapter is saved to output_dir and can be loaded with hf_load_peft_model.

    Args:
        base_model_id: Base HuggingFace model ID (e.g. 'meta-llama/Llama-2-7b-hf')
        dataset_name: HuggingFace dataset name (e.g. 'timdettmers/openassistant-guanaco')
        output_dir: Directory to save the fine-tuned adapter (default: './finetuned_model')
        num_epochs: Number of training epochs (default: 3)
        learning_rate: Learning rate (default: 2e-4)
        lora_r: LoRA rank (default: 16)
        lora_alpha: LoRA alpha scaling (default: 32)
        batch_size: Training batch size per device (default: 4)
        max_seq_length: Maximum sequence length (default: 512)
    """
    # Validate model ID
    valid, err = sanitize_model_id(base_model_id)
    if not valid:
        return f"Invalid base model ID: {err}"

    # Rate limiting
    allowed, msg = model_load_limiter.check()
    if not allowed:
        audit.log_rate_limit("model_load_limiter")
        return f"Rate limit: {msg}"

    audit.log_model_load(model_id=base_model_id, backend="finetune")

    return await hf_finetune(
        base_model_id, dataset_name, output_dir, num_epochs,
        learning_rate, lora_r, lora_alpha, batch_size, max_seq_length
    )


# --- Benchmarking ---

@mcp.tool()
async def hf_benchmark_model(
    model: str = "",
    prompt: str = "The quick brown fox jumps over the lazy dog.",
    num_runs: int = 5,
    max_new_tokens: int = 50,
) -> str:
    """Benchmark a loaded model's inference performance.

    Reports: average latency, tokens/sec, p50/p95 latency across multiple runs.

    Args:
        model: Model alias or repo_id. If empty, uses the first loaded model
        prompt: Benchmark prompt text
        num_runs: Number of inference runs for averaging (default: 5)
        max_new_tokens: Max tokens per run (default: 50)
    """
    return await hf_benchmark(model, prompt, num_runs, max_new_tokens)


# --- vLLM / TGI Backend ---

@mcp.tool()
async def hf_load_vllm_model(
    repo_id: str,
    backend: str = "vllm",
    port: int = 8000,
    alias: str = "",
    ttl: int = 0,
) -> str:
    """Load a model via vLLM or TGI inference server.

    Connects to a running vLLM/TGI server that exposes an OpenAI-compatible API.

    Args:
        repo_id: HuggingFace model repository ID served by the server
        backend: 'vllm' or 'tgi' (default: 'vllm')
        port: Port where the inference server is running (default: 8000)
        alias: Optional alias for the model
        ttl: Time-to-live in seconds (0 = default 3600s)
    """
    # Validate model ID
    valid, err = sanitize_model_id(repo_id)
    if not valid:
        return f"Invalid model ID: {err}"

    # Rate limiting
    allowed, msg = model_load_limiter.check()
    if not allowed:
        audit.log_rate_limit("model_load_limiter")
        return f"Rate limit: {msg}"

    audit.log_model_load(model_id=repo_id, backend=backend)

    return await load_vllm_model(repo_id, backend, port, alias, ttl)


# --- Audio Pipeline ---

@mcp.tool()
async def hf_transcribe_audio(
    audio_path: str,
    model: str = "openai/whisper-base",
    language: str = "",
    task: str = "transcribe",
) -> str:
    """Transcribe audio using Whisper or other ASR models.

    Supports transcription and translation tasks.

    Args:
        audio_path: Path to the audio file to transcribe
        model: HuggingFace ASR model ID (default: 'openai/whisper-base')
        language: Language code (e.g. 'en', 'fr'). Empty for auto-detect
        task: 'transcribe' for same-language, 'translate' for translation to English
    """
    return await hf_audio_transcribe(audio_path, model, language, task)


@mcp.tool()
async def hf_speak_text(
    text: str,
    model: str = "microsoft/speecht5_tts",
    output_path: str = "output.wav",
) -> str:
    """Convert text to speech using a HuggingFace TTS model.

    Args:
        text: Text to convert to speech
        model: HuggingFace TTS model ID (default: 'microsoft/speecht5_tts')
        output_path: Path to save the output audio file (default: 'output.wav')
    """
    return await hf_text_to_speech(text, model, output_path)


# --- GPU & Model Status ---

@mcp.tool()
async def hf_gpu_status() -> str:
    """Report GPU memory usage, available VRAM, and loaded model info.

    Works with CUDA GPUs and falls back gracefully when no GPU is available.
    """
    return await gpu_status()


@mcp.tool()
async def hf_model_download_status(repo_id: str) -> str:
    """Check download/cache status for a HuggingFace model.

    Reports whether the model is cached locally, its size, cache path,
    and remote model info.

    Args:
        repo_id: HuggingFace model repository ID (e.g. 'meta-llama/Meta-Llama-3-8B-Instruct')
    """
    # Validate model ID
    valid, err = sanitize_model_id(repo_id)
    if not valid:
        return f"Invalid model ID: {err}"

    return await model_download_status(repo_id)


# --- Model Registry Persistence ---

@mcp.tool()
async def hf_save_registry(path: str = ".model_registry_state.json") -> str:
    """Save current model registry state to disk for later restoration.

    Saves model metadata so they can be re-loaded later with hf_restore_registry.

    Args:
        path: File path to save the state (default: '.model_registry_state.json')
    """
    return await save_registry_state(path)


@mcp.tool()
async def hf_restore_registry(path: str = ".model_registry_state.json") -> str:
    """Restore model registry state from a saved file and re-load models.

    Args:
        path: File path to load the state from (default: '.model_registry_state.json')
    """
    return await restore_registry_state(path)


def main():
    # Log startup with resource info
    mem_mb = resource_limits.get_memory_usage_mb()
    ok, disk_msg = resource_limits.check_disk_space()
    audit.log_warning(f"Server starting. Memory: {mem_mb:.1f}MB. {disk_msg}")
    # Run with stdio transport (default)
    mcp.run()


if __name__ == "__main__":
    main()
