"""HuggingFace + LangChain integration tools for code-execution-mcp.

Provides MCP tools for loading and running HuggingFace models via
LangChain's huggingface integration (langchain-huggingface) and the
transformers/diffusers libraries.

Includes:
- TTL-based model registry with max concurrent model cap
- LoRA/PEFT adapter support for fine-tuned models
- Diffusers integration for image generation (FLUX, Stable Diffusion)
- RAG pipeline setup via LangChain retriever chains
- Safe dataset preview with row limits and gated dataset handling
"""

import gc
import json
import time
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Model Registry with TTL eviction and concurrency cap
# ---------------------------------------------------------------------------

MAX_CONCURRENT_MODELS: int = 5
MAX_CONCURRENT_EMBEDDINGS: int = 3
DEFAULT_TTL_SECONDS: int = 3600  # 1 hour


class _ModelEntry:
    """Wrapper that tracks model metadata and last-access time for TTL eviction."""

    __slots__ = ("model", "loaded_at", "last_accessed", "ttl", "backend", "repo_id")

    def __init__(self, model: Any, ttl: int, backend: str, repo_id: str) -> None:
        self.model = model
        self.loaded_at = time.monotonic()
        self.last_accessed = time.monotonic()
        self.ttl = ttl
        self.backend = backend
        self.repo_id = repo_id

    def touch(self) -> None:
        self.last_accessed = time.monotonic()

    @property
    def is_expired(self) -> bool:
        return (time.monotonic() - self.last_accessed) > self.ttl

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self.loaded_at


class _ModelRegistry:
    """Thread-safe-ish model registry with TTL eviction and max cap."""

    def __init__(self, max_models: int) -> None:
        self._entries: dict[str, _ModelEntry] = {}
        self._max_models = max_models

    def _evict_expired(self) -> list[str]:
        """Remove expired entries, return list of evicted keys."""
        evicted = [k for k, e in self._entries.items() if e.is_expired]
        for k in evicted:
            self._force_unload(k)
        return evicted

    def _force_unload(self, key: str) -> None:
        entry = self._entries.pop(key, None)
        if entry is not None:
            model = entry.model
            del entry
            del model
            gc.collect()

    def _evict_oldest(self) -> Optional[str]:
        """Evict the least-recently-accessed model to make room."""
        if not self._entries:
            return None
        oldest_key = min(self._entries, key=lambda k: self._entries[k].last_accessed)
        self._force_unload(oldest_key)
        return oldest_key

    def get(self, key: str) -> Any | None:
        """Get model by key, touching its access time. Returns None if not found."""
        self._evict_expired()
        entry = self._entries.get(key)
        if entry is None:
            return None
        entry.touch()
        return entry.model

    def put(self, key: str, model: Any, ttl: int, backend: str, repo_id: str) -> list[str]:
        """Store a model. Returns list of evicted model keys (if any)."""
        self._evict_expired()
        evicted: list[str] = []
        while len(self._entries) >= self._max_models and key not in self._entries:
            evicted_key = self._evict_oldest()
            if evicted_key:
                evicted.append(evicted_key)
        self._entries[key] = _ModelEntry(model, ttl, backend, repo_id)
        return evicted

    def remove(self, key: str) -> bool:
        if key in self._entries:
            self._force_unload(key)
            return True
        return False

    def keys(self) -> list[str]:
        self._evict_expired()
        return list(self._entries.keys())

    def items_info(self) -> dict[str, dict]:
        self._evict_expired()
        return {
            k: {
                "type": type(e.model).__name__,
                "repo_id": e.repo_id,
                "backend": e.backend,
                "age_seconds": round(e.age_seconds, 1),
                "ttl_remaining": max(0, round(e.ttl - (time.monotonic() - e.last_accessed), 1)),
            }
            for k, e in self._entries.items()
        }

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)


# Global registries
_model_registry = _ModelRegistry(max_models=MAX_CONCURRENT_MODELS)
_embedding_registry = _ModelRegistry(max_models=MAX_CONCURRENT_EMBEDDINGS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_hf_token() -> Optional[str]:
    """Retrieve HuggingFace token from environment."""
    import os

    return os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")


def _truncate_value(v: Any, max_len: int = 200) -> str:
    """Safely truncate large values for JSON serialization."""
    s = str(v)
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


# ---------------------------------------------------------------------------
# 1. Core model management tools
# ---------------------------------------------------------------------------

async def list_models() -> str:
    """List all currently loaded HuggingFace models and embeddings with TTL info."""
    llm_info = _model_registry.items_info()
    emb_info = _embedding_registry.items_info()

    result = {
        "llm_models": llm_info,
        "embedding_models": emb_info,
        "limits": {
            "max_concurrent_llms": MAX_CONCURRENT_MODELS,
            "max_concurrent_embeddings": MAX_CONCURRENT_EMBEDDINGS,
            "default_ttl_seconds": DEFAULT_TTL_SECONDS,
        },
    }
    if not llm_info and not emb_info:
        return "No models currently loaded. Use load_hf_model or load_hf_embeddings to load one."
    return json.dumps(result, indent=2)


async def load_hf_model(
    repo_id: str,
    task: str = "text-generation",
    backend: str = "api",
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    alias: str = "",
    ttl: int = 0,
) -> str:
    """Load a HuggingFace model via LangChain for text generation.

    Args:
        repo_id: HuggingFace model repository ID (e.g. 'meta-llama/Meta-Llama-3-8B-Instruct')
        task: Model task type ('text-generation', 'text2text-generation', 'summarization', 'translation')
        backend: 'api' for HuggingFace Inference API (remote), 'local' for local pipeline execution
        max_new_tokens: Maximum number of tokens to generate (default: 512)
        temperature: Sampling temperature (default: 0.1)
        alias: Optional alias for the model. If empty, uses repo_id.
        ttl: Time-to-live in seconds before auto-eviction (0 = use default of 3600s).
    """
    model_key = alias or repo_id
    ttl = ttl or DEFAULT_TTL_SECONDS

    if model_key in _model_registry:
        return f"Model '{model_key}' is already loaded. Use a different alias or unload it first."

    try:
        if backend == "api":
            from langchain_huggingface import HuggingFaceEndpoint

            llm = HuggingFaceEndpoint(
                repo_id=repo_id,
                task=task,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                huggingfacehub_api_token=_get_hf_token(),
            )
            evicted = _model_registry.put(model_key, llm, ttl, "api", repo_id)
            msg = (
                f"Loaded model '{repo_id}' via HuggingFace Inference API as '{model_key}'.\n"
                f"Backend: remote (API)\n"
                f"Task: {task}\n"
                f"Max new tokens: {max_new_tokens}\n"
                f"TTL: {ttl}s"
            )
            if evicted:
                msg += f"\nEvicted to make room: {', '.join(evicted)}"
            return msg

        elif backend == "local":
            from langchain_huggingface import HuggingFacePipeline

            llm = HuggingFacePipeline.from_model_id(
                model_id=repo_id,
                task=task,
                pipeline_kwargs={
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                },
            )
            evicted = _model_registry.put(model_key, llm, ttl, "local", repo_id)
            msg = (
                f"Loaded model '{repo_id}' locally as '{model_key}'.\n"
                f"Backend: local (transformers pipeline)\n"
                f"Task: {task}\n"
                f"Max new tokens: {max_new_tokens}\n"
                f"TTL: {ttl}s"
            )
            if evicted:
                msg += f"\nEvicted to make room: {', '.join(evicted)}"
            return msg

        else:
            return f"Unknown backend '{backend}'. Use 'api' or 'local'."

    except Exception as e:
        return f"Error loading model '{repo_id}': {e}"


async def load_hf_chat_model(
    repo_id: str,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    alias: str = "",
    ttl: int = 0,
) -> str:
    """Load a HuggingFace model wrapped as a chat model via LangChain.

    This wraps HuggingFaceEndpoint with ChatHuggingFace for proper
    chat template handling and message formatting.

    Args:
        repo_id: HuggingFace model repository ID (e.g. 'meta-llama/Meta-Llama-3-8B-Instruct')
        max_new_tokens: Maximum number of tokens to generate (default: 512)
        temperature: Sampling temperature (default: 0.1)
        alias: Optional alias for the model. If empty, uses repo_id.
        ttl: Time-to-live in seconds before auto-eviction (0 = use default).
    """
    model_key = alias or repo_id
    ttl = ttl or DEFAULT_TTL_SECONDS

    if model_key in _model_registry:
        return f"Model '{model_key}' is already loaded. Use a different alias or unload it first."

    try:
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            task="text-generation",
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            huggingfacehub_api_token=_get_hf_token(),
        )
        chat_model = ChatHuggingFace(llm=llm)
        evicted = _model_registry.put(model_key, chat_model, ttl, "api-chat", repo_id)
        msg = (
            f"Loaded chat model '{repo_id}' as '{model_key}'.\n"
            f"Backend: HuggingFace Inference API + ChatHuggingFace wrapper\n"
            f"Chat templates and message formatting enabled.\n"
            f"Max new tokens: {max_new_tokens}\n"
            f"TTL: {ttl}s"
        )
        if evicted:
            msg += f"\nEvicted to make room: {', '.join(evicted)}"
        return msg
    except Exception as e:
        return f"Error loading chat model '{repo_id}': {e}"


async def load_hf_embeddings(
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    backend: str = "local",
    alias: str = "",
    ttl: int = 0,
) -> str:
    """Load a HuggingFace embedding model via LangChain.

    Args:
        model_name: HuggingFace embedding model name (default: 'sentence-transformers/all-mpnet-base-v2')
        backend: 'local' for sentence-transformers, 'api' for HF Inference API
        alias: Optional alias for the model. If empty, uses model_name.
        ttl: Time-to-live in seconds before auto-eviction (0 = use default).
    """
    model_key = alias or model_name
    ttl = ttl or DEFAULT_TTL_SECONDS

    if model_key in _embedding_registry:
        return f"Embedding model '{model_key}' is already loaded."

    try:
        if backend == "local":
            from langchain_huggingface import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(model_name=model_name)
            evicted = _embedding_registry.put(model_key, embeddings, ttl, "local", model_name)
            msg = (
                f"Loaded embedding model '{model_name}' locally as '{model_key}'.\n"
                f"Backend: sentence-transformers (local)\n"
                f"TTL: {ttl}s"
            )
            if evicted:
                msg += f"\nEvicted to make room: {', '.join(evicted)}"
            return msg

        elif backend == "api":
            from langchain_huggingface import HuggingFaceEndpointEmbeddings

            embeddings = HuggingFaceEndpointEmbeddings(
                model=model_name,
                huggingfacehub_api_token=_get_hf_token(),
            )
            evicted = _embedding_registry.put(model_key, embeddings, ttl, "api", model_name)
            msg = (
                f"Loaded embedding model '{model_name}' via API as '{model_key}'.\n"
                f"Backend: HuggingFace Inference API\n"
                f"TTL: {ttl}s"
            )
            if evicted:
                msg += f"\nEvicted to make room: {', '.join(evicted)}"
            return msg

        else:
            return f"Unknown backend '{backend}'. Use 'local' or 'api'."

    except Exception as e:
        return f"Error loading embedding model '{model_name}': {e}"


# ---------------------------------------------------------------------------
# 2. Inference tools
# ---------------------------------------------------------------------------

async def hf_generate(
    prompt: str,
    model: str = "",
    max_new_tokens: int = 0,
    temperature: float = 0.0,
) -> str:
    """Generate text using a loaded HuggingFace model.

    Args:
        prompt: The input prompt for text generation.
        model: Alias or repo_id of the loaded model. If empty, uses the first loaded model.
        max_new_tokens: Override max tokens for this call (0 = use model default).
        temperature: Override temperature for this call (0.0 = use model default).
    """
    available_keys = _model_registry.keys()
    if not available_keys:
        return "No models loaded. Use load_hf_model first."

    model_key = model or available_keys[0]
    llm = _model_registry.get(model_key)
    if llm is None:
        available = ", ".join(available_keys)
        return f"Model '{model_key}' not found. Available models: {available}"

    try:
        kwargs: dict[str, Any] = {}
        if max_new_tokens > 0:
            kwargs["max_new_tokens"] = max_new_tokens
        if temperature > 0.0:
            kwargs["temperature"] = temperature

        result = llm.invoke(prompt, **kwargs)

        if hasattr(result, "content"):
            return str(result.content)
        return str(result)

    except Exception as e:
        return f"Error generating text: {e}"


async def hf_chat(
    messages_json: str,
    model: str = "",
) -> str:
    """Chat with a loaded HuggingFace model using message format.

    Args:
        messages_json: JSON array of message objects with 'role' and 'content' keys.
            Example: [{"role": "user", "content": "Hello!"}]
            Roles: 'system', 'user', 'assistant'
        model: Alias or repo_id of the loaded chat model. If empty, uses the first loaded model.
    """
    available_keys = _model_registry.keys()
    if not available_keys:
        return "No models loaded. Use load_hf_chat_model first."

    model_key = model or available_keys[0]
    llm = _model_registry.get(model_key)
    if llm is None:
        available = ", ".join(available_keys)
        return f"Model '{model_key}' not found. Available models: {available}"

    try:
        messages = json.loads(messages_json)
    except json.JSONDecodeError as e:
        return f"Invalid JSON in messages_json: {e}"

    try:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        result = llm.invoke(lc_messages)

        if hasattr(result, "content"):
            return str(result.content)
        return str(result)

    except Exception as e:
        return f"Error in chat: {e}"


async def hf_embed(
    texts_json: str,
    model: str = "",
    embed_type: str = "documents",
) -> str:
    """Generate embeddings using a loaded HuggingFace embedding model.

    Args:
        texts_json: JSON array of text strings to embed.
            Example: ["Hello world", "How are you?"]
        model: Alias or model name of the loaded embedding model. If empty, uses the first loaded model.
        embed_type: 'documents' for document embeddings, 'query' for query embedding (single text).
    """
    available_keys = _embedding_registry.keys()
    if not available_keys:
        return "No embedding models loaded. Use load_hf_embeddings first."

    model_key = model or available_keys[0]
    embeddings_model = _embedding_registry.get(model_key)
    if embeddings_model is None:
        available = ", ".join(available_keys)
        return f"Embedding model '{model_key}' not found. Available: {available}"

    try:
        texts = json.loads(texts_json)
    except json.JSONDecodeError as e:
        return f"Invalid JSON in texts_json: {e}"

    try:
        if embed_type == "query":
            if len(texts) != 1:
                return "Query embedding expects exactly one text string."
            result = embeddings_model.embed_query(texts[0])
            return json.dumps({"embedding": result, "dimensions": len(result)})
        else:
            result = embeddings_model.embed_documents(texts)
            return json.dumps({
                "embeddings_count": len(result),
                "dimensions": len(result[0]) if result else 0,
                "embeddings": result,
            })
    except Exception as e:
        return f"Error generating embeddings: {e}"


async def unload_model(model: str, model_type: str = "llm") -> str:
    """Unload a previously loaded model to free resources.

    Explicitly releases GPU/CPU memory via garbage collection.

    Args:
        model: Alias or name of the model to unload.
        model_type: 'llm' for language models, 'embedding' for embedding models, 'all' to unload everything.
    """
    if model_type == "all":
        llm_keys = _model_registry.keys()
        emb_keys = _embedding_registry.keys()
        for k in list(llm_keys):
            _model_registry.remove(k)
        for k in list(emb_keys):
            _embedding_registry.remove(k)
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        return f"Unloaded all models. LLMs: {len(llm_keys)}, Embeddings: {len(emb_keys)}. GPU cache cleared."

    if model_type == "llm":
        if _model_registry.remove(model):
            gc.collect()
            return f"Unloaded LLM model '{model}'. Memory freed."
        available = ", ".join(_model_registry.keys()) or "(none)"
        return f"LLM model '{model}' not found. Loaded: {available}"
    elif model_type == "embedding":
        if _embedding_registry.remove(model):
            gc.collect()
            return f"Unloaded embedding model '{model}'. Memory freed."
        available = ", ".join(_embedding_registry.keys()) or "(none)"
        return f"Embedding model '{model}' not found. Loaded: {available}"
    else:
        return f"Unknown model_type '{model_type}'. Use 'llm', 'embedding', or 'all'."


# ---------------------------------------------------------------------------
# 3. Pipeline & dataset tools (with safety)
# ---------------------------------------------------------------------------

async def hf_pipeline_task(
    task: str,
    input_text: str,
    model: str = "",
) -> str:
    """Run a HuggingFace transformers pipeline task directly.

    Supports tasks like text-classification, sentiment-analysis,
    summarization, translation, question-answering, etc.

    Args:
        task: The pipeline task (e.g. 'sentiment-analysis', 'summarization',
            'translation_en_to_fr', 'text-classification', 'ner',
            'question-answering', 'fill-mask', 'zero-shot-classification')
        input_text: The input text to process.
        model: Optional HuggingFace model ID. If empty, uses the default model for the task.
    """
    try:
        from transformers import pipeline

        kwargs: dict[str, Any] = {"task": task}
        if model:
            kwargs["model"] = model

        pipe = pipeline(**kwargs)
        result = pipe(input_text)

        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return f"Error running pipeline task '{task}': {e}"


MAX_DATASET_PREVIEW_ROWS = 50
MAX_FIELD_LENGTH = 500


async def hf_dataset_info(
    dataset_name: str,
    split: str = "train",
    num_rows: int = 5,
) -> str:
    """Load and preview a HuggingFace dataset with safety limits.

    Uses streaming mode to avoid downloading entire datasets. Handles gated
    and auth-required datasets with clear error messages. Truncates large
    field values to prevent memory issues.

    Args:
        dataset_name: Dataset identifier on HuggingFace Hub (e.g. 'rajpurkar/squad')
        split: Dataset split to preview ('train', 'test', 'validation')
        num_rows: Number of example rows to return (default: 5, max: 50)
    """
    num_rows = max(1, min(num_rows, MAX_DATASET_PREVIEW_ROWS))

    try:
        from datasets import load_dataset
    except ImportError:
        return "Error: 'datasets' library not installed. Install with: pip install datasets"

    token = _get_hf_token()

    try:
        dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=True,
            token=token,
            trust_remote_code=False,
        )
    except ValueError as e:
        error_msg = str(e)
        if "gated" in error_msg.lower() or "access" in error_msg.lower():
            return (
                f"Access denied for dataset '{dataset_name}'. "
                f"This dataset is gated or requires authentication.\n"
                f"1. Set HUGGINGFACE_TOKEN or HF_TOKEN environment variable\n"
                f"2. Accept the dataset's terms at https://huggingface.co/datasets/{dataset_name}\n"
                f"Original error: {e}"
            )
        raise
    except Exception as e:
        error_msg = str(e).lower()
        if "401" in error_msg or "403" in error_msg or "unauthorized" in error_msg:
            return (
                f"Authentication required for dataset '{dataset_name}'.\n"
                f"Set HUGGINGFACE_TOKEN or HF_TOKEN environment variable.\n"
                f"Original error: {e}"
            )
        return f"Error loading dataset '{dataset_name}': {e}"

    rows: list[dict] = []
    columns: set[str] = set()
    try:
        for i, row in enumerate(dataset):
            if i >= num_rows:
                break
            truncated_row = {}
            for k, v in row.items():
                columns.add(k)
                sv = str(v)
                if len(sv) > MAX_FIELD_LENGTH:
                    truncated_row[k] = sv[:MAX_FIELD_LENGTH] + f"... [truncated, {len(sv)} chars total]"
                else:
                    truncated_row[k] = v
            rows.append(truncated_row)
    except Exception as e:
        if rows:
            pass  # return what we got
        else:
            return f"Error iterating dataset '{dataset_name}': {e}"

    info = {
        "dataset": dataset_name,
        "split": split,
        "columns": sorted(columns),
        "num_samples_shown": len(rows),
        "max_rows_allowed": MAX_DATASET_PREVIEW_ROWS,
        "sample_rows": rows,
    }
    return json.dumps(info, indent=2, default=str)


# ---------------------------------------------------------------------------
# 4. LoRA / PEFT adapter support
# ---------------------------------------------------------------------------

async def load_peft_model(
    base_model_id: str,
    adapter_id: str,
    task: str = "text-generation",
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    alias: str = "",
    ttl: int = 0,
    quantize: str = "",
) -> str:
    """Load a base model with a LoRA/PEFT adapter applied.

    Supports quantization via bitsandbytes for memory-efficient inference.

    Args:
        base_model_id: Base HuggingFace model ID (e.g. 'meta-llama/Llama-2-7b-hf')
        adapter_id: PEFT adapter ID from HuggingFace Hub or local path
        task: Model task type (default: 'text-generation')
        max_new_tokens: Maximum tokens to generate (default: 512)
        temperature: Sampling temperature (default: 0.1)
        alias: Optional alias. If empty, uses '{base_model_id}+{adapter_id}'.
        ttl: TTL in seconds (0 = default 3600s).
        quantize: Quantization mode: '4bit', '8bit', or '' for none.
    """
    model_key = alias or f"{base_model_id}+{adapter_id}"
    ttl = ttl or DEFAULT_TTL_SECONDS

    if model_key in _model_registry:
        return f"Model '{model_key}' is already loaded."

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        import torch

        # Quantization config
        quant_kwargs: dict[str, Any] = {}
        if quantize == "4bit":
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif quantize == "8bit":
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            token=_get_hf_token(),
            **quant_kwargs,
        )

        model = PeftModel.from_pretrained(base_model, adapter_id, token=_get_hf_token())
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=_get_hf_token())

        from langchain_huggingface import HuggingFacePipeline
        from transformers import pipeline as hf_pipeline

        pipe = hf_pipeline(
            task,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        llm = HuggingFacePipeline(pipeline=pipe)

        evicted = _model_registry.put(model_key, llm, ttl, "local-peft", base_model_id)
        msg = (
            f"Loaded PEFT model: base='{base_model_id}', adapter='{adapter_id}' as '{model_key}'.\n"
            f"Quantization: {quantize or 'none'}\n"
            f"Task: {task}\n"
            f"TTL: {ttl}s"
        )
        if evicted:
            msg += f"\nEvicted to make room: {', '.join(evicted)}"
        return msg

    except ImportError as e:
        missing = str(e)
        return (
            f"Missing dependency for PEFT support: {missing}\n"
            f"Install with: pip install peft bitsandbytes"
        )
    except Exception as e:
        return f"Error loading PEFT model: {e}"


# ---------------------------------------------------------------------------
# 5. Diffusers integration for image generation
# ---------------------------------------------------------------------------

async def generate_image(
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
    """Generate an image using HuggingFace Diffusers or Inference API.

    Args:
        prompt: Text description of the image to generate.
        model_id: Diffusion model ID (e.g. 'stabilityai/stable-diffusion-xl-base-1.0',
            'black-forest-labs/FLUX.1-dev')
        negative_prompt: What to avoid in the image.
        num_inference_steps: Number of denoising steps (default: 30). Higher = better quality.
        guidance_scale: Classifier-free guidance scale (default: 7.5). Higher = more prompt adherence.
        width: Image width in pixels (default: 1024).
        height: Image height in pixels (default: 1024).
        output_path: File path to save the generated image (default: 'generated_image.png').
        backend: 'api' for HuggingFace Inference API, 'local' for local diffusers pipeline.
    """
    try:
        if backend == "api":
            from huggingface_hub import InferenceClient

            client = InferenceClient(token=_get_hf_token())
            image = client.text_to_image(
                prompt,
                model=model_id,
                negative_prompt=negative_prompt or None,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
            )
            image.save(output_path)
            return (
                f"Image generated and saved to '{output_path}'.\n"
                f"Model: {model_id}\n"
                f"Size: {width}x{height}\n"
                f"Steps: {num_inference_steps}\n"
                f"Backend: HuggingFace Inference API"
            )

        elif backend == "local":
            import torch
            from diffusers import AutoPipelineForText2Image

            pipe = AutoPipelineForText2Image.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                token=_get_hf_token(),
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipe = pipe.to(device)

            gen_kwargs: dict[str, Any] = {
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "width": width,
                "height": height,
            }
            if negative_prompt:
                gen_kwargs["negative_prompt"] = negative_prompt

            image = pipe(**gen_kwargs).images[0]
            image.save(output_path)

            # Clean up to free VRAM
            del pipe
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return (
                f"Image generated and saved to '{output_path}'.\n"
                f"Model: {model_id}\n"
                f"Size: {width}x{height}\n"
                f"Steps: {num_inference_steps}\n"
                f"Backend: local (diffusers)"
            )

        else:
            return f"Unknown backend '{backend}'. Use 'api' or 'local'."

    except ImportError as e:
        return (
            f"Missing dependency: {e}\n"
            f"For API: pip install huggingface_hub\n"
            f"For local: pip install diffusers torch accelerate"
        )
    except Exception as e:
        return f"Error generating image: {e}"


# ---------------------------------------------------------------------------
# 6. RAG pipeline setup via LangChain
# ---------------------------------------------------------------------------

async def setup_rag_pipeline(
    documents_json: str,
    embedding_model: str = "",
    llm_model: str = "",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    search_type: str = "similarity",
    search_k: int = 4,
) -> str:
    """Set up a Retrieval-Augmented Generation pipeline using LangChain.

    Ingests documents, creates embeddings, stores in an in-memory vector store,
    and configures a retrieval QA chain.

    Args:
        documents_json: JSON array of document strings to ingest.
            Example: ["Document text 1...", "Document text 2..."]
        embedding_model: Alias of a loaded embedding model. If empty, uses first available.
        llm_model: Alias of a loaded LLM. If empty, uses first available.
        chunk_size: Character count per text chunk (default: 500).
        chunk_overlap: Overlap between chunks (default: 50).
        search_type: Vector search type: 'similarity' or 'mmr' (default: 'similarity').
        search_k: Number of documents to retrieve per query (default: 4).
    """
    # Validate models are loaded
    llm_keys = _model_registry.keys()
    emb_keys = _embedding_registry.keys()

    if not llm_keys:
        return "No LLM models loaded. Load one first with load_hf_model or load_hf_chat_model."
    if not emb_keys:
        return "No embedding models loaded. Load one first with load_hf_embeddings."

    llm_key = llm_model or llm_keys[0]
    emb_key = embedding_model or emb_keys[0]

    llm = _model_registry.get(llm_key)
    if llm is None:
        return f"LLM '{llm_key}' not found. Available: {', '.join(llm_keys)}"

    embeddings = _embedding_registry.get(emb_key)
    if embeddings is None:
        return f"Embedding model '{emb_key}' not found. Available: {', '.join(emb_keys)}"

    try:
        docs_raw = json.loads(documents_json)
    except json.JSONDecodeError as e:
        return f"Invalid JSON in documents_json: {e}"

    if not isinstance(docs_raw, list) or not docs_raw:
        return "documents_json must be a non-empty JSON array of strings."

    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import FAISS
        from langchain.chains import RetrievalQA
        from langchain.schema import Document

        documents = [Document(page_content=str(d)) for d in docs_raw]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": search_k},
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )

        # Store in model registry so it can be queried later
        rag_key = f"rag:{llm_key}+{emb_key}"
        _model_registry.put(rag_key, qa_chain, DEFAULT_TTL_SECONDS, "rag-chain", rag_key)

        return (
            f"RAG pipeline created as '{rag_key}'.\n"
            f"Documents ingested: {len(docs_raw)}\n"
            f"Chunks created: {len(chunks)} (size={chunk_size}, overlap={chunk_overlap})\n"
            f"Retriever: {search_type}, k={search_k}\n"
            f"LLM: {llm_key}\n"
            f"Embeddings: {emb_key}\n\n"
            f"Query it with hf_generate(prompt, model='{rag_key}')"
        )

    except ImportError as e:
        return (
            f"Missing dependency for RAG: {e}\n"
            f"Install with: pip install langchain langchain-community faiss-cpu"
        )
    except Exception as e:
        return f"Error setting up RAG pipeline: {e}"


async def rag_query(
    query: str,
    pipeline: str = "",
) -> str:
    """Query a previously created RAG pipeline.

    Args:
        query: The question to answer using the RAG pipeline.
        pipeline: RAG pipeline key (from setup_rag_pipeline output). If empty, uses first available.
    """
    available_keys = _model_registry.keys()
    rag_keys = [k for k in available_keys if k.startswith("rag:")]

    if not rag_keys:
        return "No RAG pipelines configured. Use setup_rag_pipeline first."

    pipeline_key = pipeline or rag_keys[0]
    qa_chain = _model_registry.get(pipeline_key)
    if qa_chain is None:
        return f"RAG pipeline '{pipeline_key}' not found. Available: {', '.join(rag_keys)}"

    try:
        result = qa_chain.invoke({"query": query})

        answer = result.get("result", str(result))
        sources = result.get("source_documents", [])
        source_texts = [
            _truncate_value(doc.page_content, 200) for doc in sources[:3]
        ]

        output = {
            "answer": answer,
            "sources_used": len(sources),
            "top_sources": source_texts,
        }
        return json.dumps(output, indent=2, default=str)

    except Exception as e:
        return f"Error querying RAG pipeline: {e}"
