"""HuggingFace + LangChain integration tools for code-execution-mcp.

Provides MCP tools for loading and running HuggingFace models via
LangChain's huggingface integration (langchain-huggingface) and the
transformers/diffusers libraries.
"""

import json
from typing import Any, Optional

# Global registries for loaded models
_model_registry: dict[str, Any] = {}
_embedding_registry: dict[str, Any] = {}


def _get_hf_token() -> Optional[str]:
    """Retrieve HuggingFace token from environment."""
    import os

    return os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")


async def list_models() -> str:
    """List all currently loaded HuggingFace models and embeddings."""
    models = {
        "llm_models": {
            k: {
                "type": type(v).__name__,
                "repo_id": getattr(v, "repo_id", None)
                or getattr(v, "model_id", None)
                or str(k),
            }
            for k, v in _model_registry.items()
        },
        "embedding_models": {
            k: {
                "type": type(v).__name__,
                "model_name": getattr(v, "model_name", str(k)),
            }
            for k, v in _embedding_registry.items()
        },
    }
    if not models["llm_models"] and not models["embedding_models"]:
        return "No models currently loaded. Use load_hf_model or load_hf_embeddings to load one."
    return json.dumps(models, indent=2)


async def load_hf_model(
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
        task: Model task type ('text-generation', 'text2text-generation', 'summarization', 'translation')
        backend: 'api' for HuggingFace Inference API (remote), 'local' for local pipeline execution
        max_new_tokens: Maximum number of tokens to generate (default: 512)
        temperature: Sampling temperature (default: 0.1)
        alias: Optional alias for the model. If empty, uses repo_id.
    """
    model_key = alias or repo_id

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
            _model_registry[model_key] = llm
            return (
                f"Loaded model '{repo_id}' via HuggingFace Inference API as '{model_key}'.\n"
                f"Backend: remote (API)\n"
                f"Task: {task}\n"
                f"Max new tokens: {max_new_tokens}"
            )

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
            _model_registry[model_key] = llm
            return (
                f"Loaded model '{repo_id}' locally as '{model_key}'.\n"
                f"Backend: local (transformers pipeline)\n"
                f"Task: {task}\n"
                f"Max new tokens: {max_new_tokens}"
            )

        else:
            return f"Unknown backend '{backend}'. Use 'api' or 'local'."

    except Exception as e:
        return f"Error loading model '{repo_id}': {e}"


async def load_hf_chat_model(
    repo_id: str,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    alias: str = "",
) -> str:
    """Load a HuggingFace model wrapped as a chat model via LangChain.

    This wraps HuggingFaceEndpoint with ChatHuggingFace for proper
    chat template handling and message formatting.

    Args:
        repo_id: HuggingFace model repository ID (e.g. 'meta-llama/Meta-Llama-3-8B-Instruct')
        max_new_tokens: Maximum number of tokens to generate (default: 512)
        temperature: Sampling temperature (default: 0.1)
        alias: Optional alias for the model. If empty, uses repo_id.
    """
    model_key = alias or repo_id

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
        _model_registry[model_key] = chat_model
        return (
            f"Loaded chat model '{repo_id}' as '{model_key}'.\n"
            f"Backend: HuggingFace Inference API + ChatHuggingFace wrapper\n"
            f"Chat templates and message formatting enabled.\n"
            f"Max new tokens: {max_new_tokens}"
        )
    except Exception as e:
        return f"Error loading chat model '{repo_id}': {e}"


async def load_hf_embeddings(
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    backend: str = "local",
    alias: str = "",
) -> str:
    """Load a HuggingFace embedding model via LangChain.

    Args:
        model_name: HuggingFace embedding model name (default: 'sentence-transformers/all-mpnet-base-v2')
        backend: 'local' for sentence-transformers, 'api' for HF Inference API
        alias: Optional alias for the model. If empty, uses model_name.
    """
    model_key = alias or model_name

    if model_key in _embedding_registry:
        return f"Embedding model '{model_key}' is already loaded."

    try:
        if backend == "local":
            from langchain_huggingface import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(model_name=model_name)
            _embedding_registry[model_key] = embeddings
            return (
                f"Loaded embedding model '{model_name}' locally as '{model_key}'.\n"
                f"Backend: sentence-transformers (local)"
            )

        elif backend == "api":
            from langchain_huggingface import HuggingFaceEndpointEmbeddings

            embeddings = HuggingFaceEndpointEmbeddings(
                model=model_name,
                huggingfacehub_api_token=_get_hf_token(),
            )
            _embedding_registry[model_key] = embeddings
            return (
                f"Loaded embedding model '{model_name}' via API as '{model_key}'.\n"
                f"Backend: HuggingFace Inference API"
            )

        else:
            return f"Unknown backend '{backend}'. Use 'local' or 'api'."

    except Exception as e:
        return f"Error loading embedding model '{model_name}': {e}"


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
    if not _model_registry:
        return "No models loaded. Use load_hf_model first."

    model_key = model or next(iter(_model_registry))
    llm = _model_registry.get(model_key)
    if llm is None:
        available = ", ".join(_model_registry.keys())
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
    if not _model_registry:
        return "No models loaded. Use load_hf_chat_model first."

    model_key = model or next(iter(_model_registry))
    llm = _model_registry.get(model_key)
    if llm is None:
        available = ", ".join(_model_registry.keys())
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
    if not _embedding_registry:
        return "No embedding models loaded. Use load_hf_embeddings first."

    model_key = model or next(iter(_embedding_registry))
    embeddings_model = _embedding_registry.get(model_key)
    if embeddings_model is None:
        available = ", ".join(_embedding_registry.keys())
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

    Args:
        model: Alias or name of the model to unload.
        model_type: 'llm' for language models, 'embedding' for embedding models.
    """
    if model_type == "llm":
        if model in _model_registry:
            del _model_registry[model]
            return f"Unloaded LLM model '{model}'."
        available = ", ".join(_model_registry.keys()) or "(none)"
        return f"LLM model '{model}' not found. Loaded: {available}"
    elif model_type == "embedding":
        if model in _embedding_registry:
            del _embedding_registry[model]
            return f"Unloaded embedding model '{model}'."
        available = ", ".join(_embedding_registry.keys()) or "(none)"
        return f"Embedding model '{model}' not found. Loaded: {available}"
    else:
        return f"Unknown model_type '{model_type}'. Use 'llm' or 'embedding'."


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


async def hf_dataset_info(
    dataset_name: str,
    split: str = "train",
    num_rows: int = 3,
) -> str:
    """Load and preview a HuggingFace dataset.

    Args:
        dataset_name: Dataset identifier on HuggingFace Hub (e.g. 'rajpurkar/squad')
        split: Dataset split to preview ('train', 'test', 'validation')
        num_rows: Number of example rows to return (default: 3)
    """
    try:
        from datasets import load_dataset

        dataset = load_dataset(dataset_name, split=split, streaming=True)
        rows = []
        for i, row in enumerate(dataset):
            if i >= num_rows:
                break
            rows.append(row)

        info = {
            "dataset": dataset_name,
            "split": split,
            "sample_rows": rows,
            "num_samples_shown": len(rows),
        }
        return json.dumps(info, indent=2, default=str)
    except Exception as e:
        return f"Error loading dataset '{dataset_name}': {e}"
