"""Tests for the async HuggingFace tool functions in huggingface_tools.py.

All heavy optional dependencies (torch, transformers, langchain_huggingface,
datasets, etc.) are mocked so that the test suite can run in a minimal
environment.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import huggingface_tools as ht


# ---------------------------------------------------------------------------
# Helper to run async tool functions
# ---------------------------------------------------------------------------

def run(coro):
    """Run an async coroutine to completion and return the result."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ============================================================================
# list_models()
# ============================================================================


class TestListModels:
    """Tests for the list_models() tool function."""

    def test_list_models_empty_registries(self, reset_global_registries):
        result = run(ht.list_models())
        assert "No models currently loaded" in result

    def test_list_models_with_loaded_models(self, reset_global_registries):
        model_reg, emb_reg = reset_global_registries
        model_reg.put("my-llm", "fake-llm-obj", ttl=3600, backend="api", repo_id="org/llm")
        emb_reg.put("my-emb", "fake-emb-obj", ttl=3600, backend="local", repo_id="org/emb")

        result = run(ht.list_models())
        parsed = json.loads(result)
        assert "my-llm" in parsed["llm_models"]
        assert "my-emb" in parsed["embedding_models"]
        assert parsed["limits"]["max_concurrent_llms"] == ht.MAX_CONCURRENT_MODELS
        assert parsed["limits"]["max_concurrent_embeddings"] == ht.MAX_CONCURRENT_EMBEDDINGS

    def test_list_models_only_llm(self, reset_global_registries):
        model_reg, _ = reset_global_registries
        model_reg.put("llm-only", "obj", ttl=3600, backend="api", repo_id="org/llm")

        result = run(ht.list_models())
        parsed = json.loads(result)
        assert "llm-only" in parsed["llm_models"]
        assert parsed["embedding_models"] == {}

    def test_list_models_only_embedding(self, reset_global_registries):
        _, emb_reg = reset_global_registries
        emb_reg.put("emb-only", "obj", ttl=3600, backend="local", repo_id="org/emb")

        result = run(ht.list_models())
        parsed = json.loads(result)
        assert parsed["llm_models"] == {}
        assert "emb-only" in parsed["embedding_models"]

    def test_list_models_includes_default_ttl(self, reset_global_registries):
        model_reg, _ = reset_global_registries
        model_reg.put("x", "obj", ttl=3600, backend="api", repo_id="r/x")

        result = run(ht.list_models())
        parsed = json.loads(result)
        assert parsed["limits"]["default_ttl_seconds"] == ht.DEFAULT_TTL_SECONDS


# ============================================================================
# load_hf_model()
# ============================================================================


class TestLoadHFModel:
    """Tests for the load_hf_model() tool function."""

    def test_unknown_backend(self, reset_global_registries):
        result = run(ht.load_hf_model(repo_id="org/m", backend="quantum"))
        assert "Unknown backend 'quantum'" in result

    def test_model_already_loaded(self, reset_global_registries):
        model_reg, _ = reset_global_registries
        model_reg.put("org/m", "existing-obj", ttl=3600, backend="api", repo_id="org/m")

        result = run(ht.load_hf_model(repo_id="org/m"))
        assert "already loaded" in result

    def test_model_already_loaded_with_alias(self, reset_global_registries):
        model_reg, _ = reset_global_registries
        model_reg.put("my-alias", "existing-obj", ttl=3600, backend="api", repo_id="org/m")

        result = run(ht.load_hf_model(repo_id="org/m", alias="my-alias"))
        assert "already loaded" in result

    @patch("huggingface_tools._get_hf_token", return_value="fake-token")
    def test_load_api_backend_success(self, mock_token, reset_global_registries):
        model_reg, _ = reset_global_registries
        mock_endpoint = MagicMock()

        with patch.dict("sys.modules", {"langchain_huggingface": MagicMock()}):
            import sys
            sys.modules["langchain_huggingface"].HuggingFaceEndpoint = MagicMock(
                return_value=mock_endpoint
            )
            result = run(ht.load_hf_model(repo_id="org/model", backend="api", ttl=600))

        assert "Loaded model 'org/model'" in result
        assert "Backend: remote (API)" in result
        assert "TTL: 600s" in result
        assert "org/model" in model_reg

    @patch("huggingface_tools._get_hf_token", return_value="fake-token")
    def test_load_local_backend_success(self, mock_token, reset_global_registries):
        model_reg, _ = reset_global_registries
        mock_pipeline = MagicMock()

        with patch.dict("sys.modules", {"langchain_huggingface": MagicMock()}):
            import sys
            sys.modules["langchain_huggingface"].HuggingFacePipeline = MagicMock()
            sys.modules["langchain_huggingface"].HuggingFacePipeline.from_model_id = MagicMock(
                return_value=mock_pipeline
            )
            result = run(ht.load_hf_model(repo_id="org/local-m", backend="local"))

        assert "Loaded model 'org/local-m' locally" in result
        assert "Backend: local" in result

    def test_load_default_ttl_when_zero(self, reset_global_registries):
        """When ttl=0, the function should use DEFAULT_TTL_SECONDS."""
        model_reg, _ = reset_global_registries
        mock_endpoint = MagicMock()

        with patch.dict("sys.modules", {"langchain_huggingface": MagicMock()}):
            import sys
            sys.modules["langchain_huggingface"].HuggingFaceEndpoint = MagicMock(
                return_value=mock_endpoint
            )
            with patch("huggingface_tools._get_hf_token", return_value=None):
                result = run(ht.load_hf_model(repo_id="org/m2", backend="api", ttl=0))

        assert f"TTL: {ht.DEFAULT_TTL_SECONDS}s" in result

    def test_load_handles_import_error(self, reset_global_registries):
        """If langchain_huggingface is not installed, should return an error string."""
        with patch.dict("sys.modules", {"langchain_huggingface": None}):
            # Forcing an ImportError by making the import fail
            result = run(ht.load_hf_model(repo_id="org/m3", backend="api"))
        assert "Error loading model" in result

    @patch("huggingface_tools._get_hf_token", return_value="fake-token")
    def test_load_api_with_eviction_message(self, mock_token, reset_global_registries):
        """Loading beyond capacity triggers eviction; result mentions evicted keys."""
        model_reg, _ = reset_global_registries
        # Fill to capacity (5)
        for i in range(5):
            model_reg.put(f"m{i}", f"obj{i}", ttl=3600, backend="api", repo_id=f"r/{i}")

        with patch.dict("sys.modules", {"langchain_huggingface": MagicMock()}):
            import sys
            sys.modules["langchain_huggingface"].HuggingFaceEndpoint = MagicMock(
                return_value=MagicMock()
            )
            result = run(ht.load_hf_model(repo_id="new/model", backend="api"))

        assert "Loaded model" in result
        assert "Evicted to make room" in result

    @patch("huggingface_tools._get_hf_token", return_value="fake-token")
    def test_load_with_custom_alias(self, mock_token, reset_global_registries):
        model_reg, _ = reset_global_registries
        with patch.dict("sys.modules", {"langchain_huggingface": MagicMock()}):
            import sys
            sys.modules["langchain_huggingface"].HuggingFaceEndpoint = MagicMock(
                return_value=MagicMock()
            )
            result = run(ht.load_hf_model(repo_id="org/m", alias="my-alias", backend="api"))

        assert "my-alias" in model_reg
        assert "Loaded model" in result


# ============================================================================
# load_hf_chat_model()
# ============================================================================


class TestLoadHFChatModel:
    """Tests for the load_hf_chat_model() tool function."""

    @patch("huggingface_tools._get_hf_token", return_value="fake-token")
    def test_load_chat_model_success(self, mock_token, reset_global_registries):
        model_reg, _ = reset_global_registries

        with patch.dict("sys.modules", {"langchain_huggingface": MagicMock()}):
            import sys
            sys.modules["langchain_huggingface"].HuggingFaceEndpoint = MagicMock(
                return_value=MagicMock()
            )
            sys.modules["langchain_huggingface"].ChatHuggingFace = MagicMock(
                return_value=MagicMock()
            )
            result = run(ht.load_hf_chat_model(repo_id="org/chat-model"))

        assert "Loaded chat model" in result
        assert "Chat templates and message formatting enabled" in result
        assert len(model_reg) == 1

    def test_load_chat_model_already_loaded(self, reset_global_registries):
        model_reg, _ = reset_global_registries
        model_reg.put("org/chat-m", "existing-obj", ttl=3600, backend="api", repo_id="org/chat-m")

        result = run(ht.load_hf_chat_model(repo_id="org/chat-m"))
        assert "already loaded" in result

    @patch("huggingface_tools._get_hf_token", return_value="fake-token")
    def test_load_chat_model_with_alias_and_ttl(self, mock_token, reset_global_registries):
        model_reg, _ = reset_global_registries

        with patch.dict("sys.modules", {"langchain_huggingface": MagicMock()}):
            import sys
            sys.modules["langchain_huggingface"].HuggingFaceEndpoint = MagicMock(
                return_value=MagicMock()
            )
            sys.modules["langchain_huggingface"].ChatHuggingFace = MagicMock(
                return_value=MagicMock()
            )
            result = run(ht.load_hf_chat_model(
                repo_id="org/chat-m", alias="my-chat", ttl=7200
            ))

        assert "my-chat" in model_reg
        assert "TTL: 7200s" in result

    def test_load_chat_model_error(self, reset_global_registries):
        with patch.dict("sys.modules", {"langchain_huggingface": None}):
            result = run(ht.load_hf_chat_model(repo_id="org/bad"))
        assert "Error loading chat model" in result

    @patch("huggingface_tools._get_hf_token", return_value="fake-token")
    def test_load_chat_model_default_ttl(self, mock_token, reset_global_registries):
        model_reg, _ = reset_global_registries

        with patch.dict("sys.modules", {"langchain_huggingface": MagicMock()}):
            import sys
            sys.modules["langchain_huggingface"].HuggingFaceEndpoint = MagicMock(
                return_value=MagicMock()
            )
            sys.modules["langchain_huggingface"].ChatHuggingFace = MagicMock(
                return_value=MagicMock()
            )
            result = run(ht.load_hf_chat_model(repo_id="org/chat-m2", ttl=0))

        assert f"TTL: {ht.DEFAULT_TTL_SECONDS}s" in result


# ============================================================================
# load_hf_embeddings()
# ============================================================================


class TestLoadHFEmbeddings:
    """Tests for the load_hf_embeddings() tool function."""

    def test_unknown_backend(self, reset_global_registries):
        result = run(ht.load_hf_embeddings(backend="quantum"))
        assert "Unknown backend 'quantum'" in result

    def test_embedding_already_loaded(self, reset_global_registries):
        _, emb_reg = reset_global_registries
        emb_reg.put("sentence-transformers/all-mpnet-base-v2", "obj", ttl=3600,
                     backend="local", repo_id="sentence-transformers/all-mpnet-base-v2")

        result = run(ht.load_hf_embeddings())
        assert "already loaded" in result

    def test_load_local_backend_success(self, reset_global_registries):
        _, emb_reg = reset_global_registries

        with patch.dict("sys.modules", {"langchain_huggingface": MagicMock()}):
            import sys
            sys.modules["langchain_huggingface"].HuggingFaceEmbeddings = MagicMock(
                return_value=MagicMock()
            )
            result = run(ht.load_hf_embeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                backend="local",
            ))

        assert "Loaded embedding model" in result
        assert "locally" in result
        assert len(emb_reg) == 1

    @patch("huggingface_tools._get_hf_token", return_value="fake-token")
    def test_load_api_backend_success(self, mock_token, reset_global_registries):
        _, emb_reg = reset_global_registries

        with patch.dict("sys.modules", {"langchain_huggingface": MagicMock()}):
            import sys
            sys.modules["langchain_huggingface"].HuggingFaceEndpointEmbeddings = MagicMock(
                return_value=MagicMock()
            )
            result = run(ht.load_hf_embeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                backend="api",
            ))

        assert "Loaded embedding model" in result
        assert "API" in result
        assert len(emb_reg) == 1

    def test_load_handles_import_error(self, reset_global_registries):
        with patch.dict("sys.modules", {"langchain_huggingface": None}):
            result = run(ht.load_hf_embeddings(backend="local"))
        assert "Error loading embedding model" in result

    @patch("huggingface_tools._get_hf_token", return_value="fake-token")
    def test_load_with_alias(self, mock_token, reset_global_registries):
        _, emb_reg = reset_global_registries

        with patch.dict("sys.modules", {"langchain_huggingface": MagicMock()}):
            import sys
            sys.modules["langchain_huggingface"].HuggingFaceEmbeddings = MagicMock(
                return_value=MagicMock()
            )
            result = run(ht.load_hf_embeddings(
                model_name="st/model", alias="my-emb", backend="local",
            ))

        assert "my-emb" in emb_reg
        assert "Loaded embedding model" in result

    def test_load_default_ttl(self, reset_global_registries):
        _, emb_reg = reset_global_registries

        with patch.dict("sys.modules", {"langchain_huggingface": MagicMock()}):
            import sys
            sys.modules["langchain_huggingface"].HuggingFaceEmbeddings = MagicMock(
                return_value=MagicMock()
            )
            result = run(ht.load_hf_embeddings(
                model_name="st/model2", backend="local", ttl=0,
            ))

        assert f"TTL: {ht.DEFAULT_TTL_SECONDS}s" in result


# ============================================================================
# hf_generate()
# ============================================================================


class TestHFGenerate:
    """Tests for the hf_generate() tool function."""

    def test_no_models_loaded(self, reset_global_registries):
        result = run(ht.hf_generate(prompt="Hello"))
        assert "No models loaded" in result

    def test_model_not_found(self, reset_global_registries):
        model_reg, _ = reset_global_registries
        model_reg.put("m1", MagicMock(), ttl=3600, backend="api", repo_id="r/1")

        result = run(ht.hf_generate(prompt="Hello", model="nonexistent"))
        assert "not found" in result
        assert "m1" in result  # should list available

    def test_generate_uses_first_model_when_unspecified(self, reset_global_registries):
        model_reg, _ = reset_global_registries
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Generated text output"
        model_reg.put("default-m", mock_llm, ttl=3600, backend="api", repo_id="r/d")

        result = run(ht.hf_generate(prompt="Tell me a joke"))
        assert result == "Generated text output"
        mock_llm.invoke.assert_called_once()

    def test_generate_with_content_attr(self, reset_global_registries):
        model_reg, _ = reset_global_registries
        mock_llm = MagicMock()
        response_obj = MagicMock()
        response_obj.content = "Response via .content"
        mock_llm.invoke.return_value = response_obj
        model_reg.put("chat-m", mock_llm, ttl=3600, backend="api", repo_id="r/c")

        result = run(ht.hf_generate(prompt="Hi", model="chat-m"))
        assert result == "Response via .content"

    def test_generate_handles_exception(self, reset_global_registries):
        model_reg, _ = reset_global_registries
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("GPU OOM")
        model_reg.put("broken", mock_llm, ttl=3600, backend="api", repo_id="r/b")

        result = run(ht.hf_generate(prompt="test", model="broken"))
        assert "Error generating text" in result
        assert "GPU OOM" in result

    def test_generate_with_max_new_tokens_override(self, reset_global_registries):
        """max_new_tokens > 0 should be passed as kwargs to invoke."""
        model_reg, _ = reset_global_registries
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "ok"
        model_reg.put("m", mock_llm, ttl=3600, backend="api", repo_id="r/m")

        run(ht.hf_generate(prompt="test", model="m", max_new_tokens=256))
        _, kwargs = mock_llm.invoke.call_args
        assert kwargs["max_new_tokens"] == 256

    def test_generate_with_temperature_override(self, reset_global_registries):
        model_reg, _ = reset_global_registries
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "ok"
        model_reg.put("m", mock_llm, ttl=3600, backend="api", repo_id="r/m")

        run(ht.hf_generate(prompt="test", model="m", temperature=0.8))
        _, kwargs = mock_llm.invoke.call_args
        assert kwargs["temperature"] == 0.8

    def test_generate_no_overrides_no_kwargs(self, reset_global_registries):
        """When max_new_tokens=0 and temperature=0.0, no extra kwargs."""
        model_reg, _ = reset_global_registries
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "result"
        model_reg.put("m", mock_llm, ttl=3600, backend="api", repo_id="r/m")

        run(ht.hf_generate(prompt="test", model="m", max_new_tokens=0, temperature=0.0))
        args, kwargs = mock_llm.invoke.call_args
        assert "max_new_tokens" not in kwargs
        assert "temperature" not in kwargs


# ============================================================================
# hf_chat()
# ============================================================================


class TestHFChat:
    """Tests for the hf_chat() tool function."""

    def test_no_models_loaded(self, reset_global_registries):
        result = run(ht.hf_chat(messages_json='[{"role":"user","content":"hi"}]'))
        assert "No models loaded" in result

    def test_invalid_json(self, reset_global_registries):
        model_reg, _ = reset_global_registries
        mock_llm = MagicMock()
        model_reg.put("m", mock_llm, ttl=3600, backend="api", repo_id="r/m")

        result = run(ht.hf_chat(messages_json="not-valid-json"))
        assert "Invalid JSON" in result

    def test_model_not_found(self, reset_global_registries):
        model_reg, _ = reset_global_registries
        model_reg.put("m1", MagicMock(), ttl=3600, backend="api", repo_id="r/1")

        result = run(ht.hf_chat(
            messages_json='[{"role":"user","content":"hi"}]',
            model="missing",
        ))
        assert "not found" in result

    @patch.dict("sys.modules", {
        "langchain_core": MagicMock(),
        "langchain_core.messages": MagicMock(),
    })
    def test_chat_success_with_content_attribute(self, reset_global_registries):
        model_reg, _ = reset_global_registries
        mock_llm = MagicMock()
        response_obj = MagicMock()
        response_obj.content = "Chat response"
        mock_llm.invoke.return_value = response_obj
        model_reg.put("chat", mock_llm, ttl=3600, backend="api", repo_id="r/c")

        result = run(ht.hf_chat(
            messages_json='[{"role":"user","content":"hello"}]',
            model="chat",
        ))
        assert result == "Chat response"

    @patch.dict("sys.modules", {
        "langchain_core": MagicMock(),
        "langchain_core.messages": MagicMock(),
    })
    def test_chat_with_multiple_roles(self, reset_global_registries):
        """Verify system, user, and assistant roles are accepted."""
        model_reg, _ = reset_global_registries
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "multi-role-response"
        model_reg.put("chat", mock_llm, ttl=3600, backend="api", repo_id="r/c")

        messages = json.dumps([
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ])
        result = run(ht.hf_chat(messages_json=messages, model="chat"))
        # Should invoke the model with the message objects
        mock_llm.invoke.assert_called_once()

    @patch.dict("sys.modules", {
        "langchain_core": MagicMock(),
        "langchain_core.messages": MagicMock(),
    })
    def test_chat_invoke_error(self, reset_global_registries):
        """Errors during invoke should be caught and returned."""
        model_reg, _ = reset_global_registries
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("model crashed")
        model_reg.put("chat", mock_llm, ttl=3600, backend="api", repo_id="r/c")

        result = run(ht.hf_chat(
            messages_json='[{"role":"user","content":"hello"}]',
            model="chat",
        ))
        assert "Error in chat" in result


# ============================================================================
# hf_embed()
# ============================================================================


class TestHFEmbed:
    """Tests for the hf_embed() tool function."""

    def test_no_embedding_models_loaded(self, reset_global_registries):
        result = run(ht.hf_embed(texts_json='["hello"]'))
        assert "No embedding models loaded" in result

    def test_invalid_json(self, reset_global_registries):
        _, emb_reg = reset_global_registries
        emb_reg.put("emb", MagicMock(), ttl=3600, backend="local", repo_id="r/e")

        result = run(ht.hf_embed(texts_json="{bad"))
        assert "Invalid JSON" in result

    def test_query_embed_wrong_count(self, reset_global_registries):
        _, emb_reg = reset_global_registries
        mock_emb = MagicMock()
        emb_reg.put("emb", mock_emb, ttl=3600, backend="local", repo_id="r/e")

        result = run(ht.hf_embed(
            texts_json='["a", "b"]',
            embed_type="query",
        ))
        assert "exactly one text" in result

    def test_model_not_found(self, reset_global_registries):
        _, emb_reg = reset_global_registries
        emb_reg.put("emb-a", MagicMock(), ttl=3600, backend="local", repo_id="r/a")

        result = run(ht.hf_embed(texts_json='["hello"]', model="emb-missing"))
        assert "not found" in result

    def test_document_embedding_success(self, reset_global_registries):
        _, emb_reg = reset_global_registries
        mock_emb = MagicMock()
        mock_emb.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        emb_reg.put("emb", mock_emb, ttl=3600, backend="local", repo_id="r/e")

        result = run(ht.hf_embed(texts_json='["hello", "world"]', model="emb"))
        parsed = json.loads(result)
        assert parsed["embeddings_count"] == 2
        assert parsed["dimensions"] == 3

    def test_query_embedding_success(self, reset_global_registries):
        _, emb_reg = reset_global_registries
        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        emb_reg.put("emb", mock_emb, ttl=3600, backend="local", repo_id="r/e")

        result = run(ht.hf_embed(
            texts_json='["single query"]',
            embed_type="query",
            model="emb",
        ))
        parsed = json.loads(result)
        assert parsed["dimensions"] == 4
        assert parsed["embedding"] == [0.1, 0.2, 0.3, 0.4]

    def test_embed_handles_exception(self, reset_global_registries):
        _, emb_reg = reset_global_registries
        mock_emb = MagicMock()
        mock_emb.embed_documents.side_effect = RuntimeError("CUDA error")
        emb_reg.put("emb", mock_emb, ttl=3600, backend="local", repo_id="r/e")

        result = run(ht.hf_embed(texts_json='["text"]', model="emb"))
        assert "Error generating embeddings" in result

    def test_embed_uses_first_model_by_default(self, reset_global_registries):
        """When model is empty, uses the first available."""
        _, emb_reg = reset_global_registries
        mock_emb = MagicMock()
        mock_emb.embed_documents.return_value = [[0.1]]
        emb_reg.put("first-emb", mock_emb, ttl=3600, backend="local", repo_id="r/e")

        result = run(ht.hf_embed(texts_json='["hello"]'))
        parsed = json.loads(result)
        assert parsed["embeddings_count"] == 1

    def test_document_embedding_empty_result(self, reset_global_registries):
        """Empty embedding result should have dimensions=0."""
        _, emb_reg = reset_global_registries
        mock_emb = MagicMock()
        mock_emb.embed_documents.return_value = []
        emb_reg.put("emb", mock_emb, ttl=3600, backend="local", repo_id="r/e")

        result = run(ht.hf_embed(texts_json='["hello"]', model="emb"))
        parsed = json.loads(result)
        assert parsed["embeddings_count"] == 0
        assert parsed["dimensions"] == 0


# ============================================================================
# unload_model()
# ============================================================================


class TestUnloadModel:
    """Tests for the unload_model() tool function."""

    def test_unload_all(self, reset_global_registries):
        model_reg, emb_reg = reset_global_registries
        model_reg.put("llm1", "obj1", ttl=3600, backend="api", repo_id="r/1")
        model_reg.put("llm2", "obj2", ttl=3600, backend="api", repo_id="r/2")
        emb_reg.put("emb1", "eobj1", ttl=3600, backend="local", repo_id="r/e1")

        result = run(ht.unload_model(model="ignored", model_type="all"))
        assert "Unloaded all models" in result
        assert "LLMs: 2" in result
        assert "Embeddings: 1" in result
        assert len(model_reg) == 0
        assert len(emb_reg) == 0

    def test_unload_specific_llm(self, reset_global_registries):
        model_reg, _ = reset_global_registries
        model_reg.put("my-llm", "obj", ttl=3600, backend="api", repo_id="r/m")

        result = run(ht.unload_model(model="my-llm", model_type="llm"))
        assert "Unloaded LLM model 'my-llm'" in result
        assert len(model_reg) == 0

    def test_unload_unknown_llm(self, reset_global_registries):
        result = run(ht.unload_model(model="ghost", model_type="llm"))
        assert "not found" in result

    def test_unload_unknown_embedding(self, reset_global_registries):
        result = run(ht.unload_model(model="ghost", model_type="embedding"))
        assert "not found" in result

    def test_unload_specific_embedding(self, reset_global_registries):
        _, emb_reg = reset_global_registries
        emb_reg.put("my-emb", "eobj", ttl=3600, backend="local", repo_id="r/e")

        result = run(ht.unload_model(model="my-emb", model_type="embedding"))
        assert "Unloaded embedding model 'my-emb'" in result
        assert len(emb_reg) == 0

    def test_unload_unknown_type(self, reset_global_registries):
        result = run(ht.unload_model(model="x", model_type="quantum"))
        assert "Unknown model_type" in result

    def test_unload_all_empty_registries(self, reset_global_registries):
        """Unloading all when nothing loaded should still succeed."""
        result = run(ht.unload_model(model="", model_type="all"))
        assert "Unloaded all models" in result
        assert "LLMs: 0" in result
        assert "Embeddings: 0" in result


# ============================================================================
# hf_pipeline_task()
# ============================================================================


class TestHFPipelineTask:
    """Tests for the hf_pipeline_task() tool function."""

    def test_pipeline_success(self):
        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"label": "POSITIVE", "score": 0.99}]
        mock_pipeline_fn = MagicMock(return_value=mock_pipe)

        with patch.dict("sys.modules", {"transformers": MagicMock(pipeline=mock_pipeline_fn)}):
            result = run(ht.hf_pipeline_task(
                task="sentiment-analysis",
                input_text="I love this!",
            ))

        parsed = json.loads(result)
        assert parsed[0]["label"] == "POSITIVE"

    def test_pipeline_with_model_arg(self):
        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"summary_text": "short summary"}]
        mock_pipeline_fn = MagicMock(return_value=mock_pipe)

        with patch.dict("sys.modules", {"transformers": MagicMock(pipeline=mock_pipeline_fn)}):
            result = run(ht.hf_pipeline_task(
                task="summarization",
                input_text="Long text...",
                model="facebook/bart-large-cnn",
            ))

        parsed = json.loads(result)
        assert "summary_text" in parsed[0]
        # Verify model was passed to pipeline constructor
        mock_pipeline_fn.assert_called_once()
        call_kwargs = mock_pipeline_fn.call_args[1]
        assert call_kwargs["model"] == "facebook/bart-large-cnn"

    def test_pipeline_import_error(self):
        """When transformers is not installed, should return an error."""
        with patch.dict("sys.modules", {"transformers": None}):
            result = run(ht.hf_pipeline_task(task="sentiment-analysis", input_text="test"))
        assert "Error running pipeline" in result

    def test_pipeline_runtime_error(self):
        mock_pipeline_fn = MagicMock(side_effect=RuntimeError("model not found"))

        with patch.dict("sys.modules", {"transformers": MagicMock(pipeline=mock_pipeline_fn)}):
            result = run(ht.hf_pipeline_task(task="ner", input_text="test"))
        assert "Error running pipeline" in result

    def test_pipeline_no_model_arg(self):
        """When model is empty, it should not be passed as a kwarg."""
        mock_pipe = MagicMock()
        mock_pipe.return_value = []
        mock_pipeline_fn = MagicMock(return_value=mock_pipe)

        with patch.dict("sys.modules", {"transformers": MagicMock(pipeline=mock_pipeline_fn)}):
            run(ht.hf_pipeline_task(task="ner", input_text="test", model=""))

        call_kwargs = mock_pipeline_fn.call_args[1]
        assert "model" not in call_kwargs


# ============================================================================
# hf_dataset_info() -- safety clamping & error handling
# ============================================================================


class TestHFDatasetInfo:
    """Tests for hf_dataset_info() safety clamping."""

    def test_num_rows_clamped_to_max(self, reset_global_registries):
        """num_rows > MAX_DATASET_PREVIEW_ROWS should be clamped."""
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter([]))

        with patch.dict("sys.modules", {"datasets": MagicMock()}) as mods:
            import sys
            sys.modules["datasets"].load_dataset.return_value = mock_ds

            result = run(ht.hf_dataset_info(
                dataset_name="test/ds",
                num_rows=999,
            ))
            parsed = json.loads(result)
            assert parsed["max_rows_allowed"] == ht.MAX_DATASET_PREVIEW_ROWS

    def test_num_rows_clamped_to_min(self, reset_global_registries):
        """num_rows < 1 should be clamped to 1."""
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter([{"col": "val"}]))

        with patch.dict("sys.modules", {"datasets": MagicMock()}) as mods:
            import sys
            sys.modules["datasets"].load_dataset.return_value = mock_ds

            result = run(ht.hf_dataset_info(
                dataset_name="test/ds",
                num_rows=-5,
            ))
            parsed = json.loads(result)
            assert parsed["num_samples_shown"] <= 1

    def test_datasets_import_error(self, reset_global_registries):
        """If 'datasets' is not installed, should return a helpful message."""
        with patch.dict("sys.modules", {"datasets": None}):
            result = run(ht.hf_dataset_info(dataset_name="test/ds"))
        assert "datasets" in result.lower()
        assert "not installed" in result.lower() or "Error" in result

    def test_field_truncation(self, reset_global_registries):
        """Fields longer than MAX_FIELD_LENGTH should be truncated."""
        long_text = "x" * 1000
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter([{"text": long_text}]))

        with patch.dict("sys.modules", {"datasets": MagicMock()}) as mods:
            import sys
            sys.modules["datasets"].load_dataset.return_value = mock_ds
            result = run(ht.hf_dataset_info(dataset_name="test/ds", num_rows=1))

        parsed = json.loads(result)
        row = parsed["sample_rows"][0]
        assert "truncated" in row["text"]
        assert len(row["text"]) <= ht.MAX_FIELD_LENGTH + 50  # allow for suffix

    def test_gated_access_error(self, reset_global_registries):
        """Gated dataset should produce a clear error message."""
        mock_datasets = MagicMock()
        mock_datasets.load_dataset.side_effect = ValueError(
            "This dataset is gated and requires access"
        )

        with patch.dict("sys.modules", {"datasets": mock_datasets}):
            result = run(ht.hf_dataset_info(dataset_name="gated/dataset"))

        assert "Access denied" in result or "gated" in result.lower()

    def test_auth_error_401(self, reset_global_registries):
        """401 errors produce auth-required message."""
        mock_datasets = MagicMock()
        mock_datasets.load_dataset.side_effect = Exception("401 Unauthorized")

        with patch.dict("sys.modules", {"datasets": mock_datasets}):
            result = run(ht.hf_dataset_info(dataset_name="private/dataset"))

        assert "Authentication required" in result

    def test_normal_preview_structure(self, reset_global_registries):
        """Normal dataset preview returns proper structure."""
        rows = [
            {"id": 1, "text": "Hello world"},
            {"id": 2, "text": "Second row"},
        ]
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(rows))

        with patch.dict("sys.modules", {"datasets": MagicMock()}) as mods:
            import sys
            sys.modules["datasets"].load_dataset.return_value = mock_ds
            result = run(ht.hf_dataset_info(dataset_name="test/ds", num_rows=5))

        parsed = json.loads(result)
        assert parsed["dataset"] == "test/ds"
        assert parsed["split"] == "train"
        assert parsed["num_samples_shown"] == 2
        assert sorted(parsed["columns"]) == ["id", "text"]


# ============================================================================
# load_peft_model()
# ============================================================================


class TestLoadPeftModel:
    """Tests for the load_peft_model() tool function."""

    def test_peft_already_loaded(self, reset_global_registries):
        model_reg, _ = reset_global_registries
        key = "base/model+adapter/id"
        model_reg.put(key, "obj", ttl=3600, backend="local-peft", repo_id="base/model")

        result = run(ht.load_peft_model(base_model_id="base/model", adapter_id="adapter/id"))
        assert "already loaded" in result

    def test_peft_import_error(self, reset_global_registries):
        """Missing peft/torch dependency gives clear error."""
        with patch.dict("sys.modules", {
            "transformers": None,
            "peft": None,
            "torch": None,
        }):
            result = run(ht.load_peft_model(base_model_id="base/m", adapter_id="adapter"))
        assert "Missing dependency" in result or "Error" in result

    def test_peft_with_custom_alias(self, reset_global_registries):
        model_reg, _ = reset_global_registries
        model_reg.put("my-peft", "obj", ttl=3600, backend="local-peft", repo_id="base/m")

        result = run(ht.load_peft_model(
            base_model_id="base/m", adapter_id="adapter", alias="my-peft",
        ))
        assert "already loaded" in result


# ============================================================================
# generate_image()
# ============================================================================


class TestGenerateImage:
    """Tests for the generate_image() tool function."""

    @patch("huggingface_tools._get_hf_token", return_value="fake-token")
    def test_api_backend_success(self, mock_token):
        mock_image = MagicMock()
        mock_client = MagicMock()
        mock_client.text_to_image.return_value = mock_image
        mock_hf_hub = MagicMock()
        mock_hf_hub.InferenceClient.return_value = mock_client

        with patch.dict("sys.modules", {"huggingface_hub": mock_hf_hub}):
            result = run(ht.generate_image(
                prompt="A cat in space",
                backend="api",
                output_path="/tmp/test_img.png",
            ))

        assert "Image generated" in result
        assert "API" in result
        mock_image.save.assert_called_once_with("/tmp/test_img.png")

    def test_unknown_backend(self):
        result = run(ht.generate_image(prompt="test", backend="tpu"))
        assert "Unknown backend" in result

    def test_import_error(self):
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            result = run(ht.generate_image(prompt="test", backend="api"))
        assert "Missing dependency" in result or "Error" in result

    @patch("huggingface_tools._get_hf_token", return_value="fake-token")
    def test_api_backend_runtime_error(self, mock_token):
        mock_client = MagicMock()
        mock_client.text_to_image.side_effect = RuntimeError("API timeout")
        mock_hf_hub = MagicMock()
        mock_hf_hub.InferenceClient.return_value = mock_client

        with patch.dict("sys.modules", {"huggingface_hub": mock_hf_hub}):
            result = run(ht.generate_image(prompt="test", backend="api"))

        assert "Error generating image" in result

    def test_local_backend_import_error(self):
        with patch.dict("sys.modules", {"torch": None, "diffusers": None}):
            result = run(ht.generate_image(prompt="test", backend="local"))
        assert "Missing dependency" in result or "Error" in result


# ============================================================================
# setup_rag_pipeline()
# ============================================================================


class TestSetupRagPipeline:
    """Tests for setup_rag_pipeline() tool function."""

    def test_no_llm_loaded(self, reset_global_registries):
        _, emb_reg = reset_global_registries
        emb_reg.put("emb", MagicMock(), ttl=3600, backend="local", repo_id="r/e")

        result = run(ht.setup_rag_pipeline('["doc1"]'))
        assert "No LLM models loaded" in result

    def test_no_embedding_loaded(self, reset_global_registries):
        model_reg, _ = reset_global_registries
        model_reg.put("llm", MagicMock(), ttl=3600, backend="api", repo_id="r/l")

        result = run(ht.setup_rag_pipeline('["doc1"]'))
        assert "No embedding models loaded" in result

    def test_invalid_json(self, reset_global_registries):
        model_reg, emb_reg = reset_global_registries
        model_reg.put("llm", MagicMock(), ttl=3600, backend="api", repo_id="r/l")
        emb_reg.put("emb", MagicMock(), ttl=3600, backend="local", repo_id="r/e")

        result = run(ht.setup_rag_pipeline("not json"))
        assert "Invalid JSON" in result

    def test_empty_documents_list(self, reset_global_registries):
        model_reg, emb_reg = reset_global_registries
        model_reg.put("llm", MagicMock(), ttl=3600, backend="api", repo_id="r/l")
        emb_reg.put("emb", MagicMock(), ttl=3600, backend="local", repo_id="r/e")

        result = run(ht.setup_rag_pipeline("[]"))
        assert "non-empty" in result

    def test_documents_not_a_list(self, reset_global_registries):
        model_reg, emb_reg = reset_global_registries
        model_reg.put("llm", MagicMock(), ttl=3600, backend="api", repo_id="r/l")
        emb_reg.put("emb", MagicMock(), ttl=3600, backend="local", repo_id="r/e")

        result = run(ht.setup_rag_pipeline('"just a string"'))
        assert "non-empty JSON array" in result

    def test_llm_not_found(self, reset_global_registries):
        model_reg, emb_reg = reset_global_registries
        model_reg.put("llm", MagicMock(), ttl=3600, backend="api", repo_id="r/l")
        emb_reg.put("emb", MagicMock(), ttl=3600, backend="local", repo_id="r/e")

        result = run(ht.setup_rag_pipeline('["doc"]', llm_model="ghost"))
        assert "not found" in result

    def test_embedding_not_found(self, reset_global_registries):
        model_reg, emb_reg = reset_global_registries
        model_reg.put("llm", MagicMock(), ttl=3600, backend="api", repo_id="r/l")
        emb_reg.put("emb", MagicMock(), ttl=3600, backend="local", repo_id="r/e")

        result = run(ht.setup_rag_pipeline('["doc"]', embedding_model="ghost"))
        assert "not found" in result

    def test_rag_import_error(self, reset_global_registries):
        model_reg, emb_reg = reset_global_registries
        model_reg.put("llm", MagicMock(), ttl=3600, backend="api", repo_id="r/l")
        emb_reg.put("emb", MagicMock(), ttl=3600, backend="local", repo_id="r/e")

        with patch.dict("sys.modules", {
            "langchain": None,
            "langchain.text_splitter": None,
        }):
            result = run(ht.setup_rag_pipeline('["doc text"]'))

        assert "Missing dependency" in result or "Error" in result

    def test_rag_success_with_mock_chain(self, reset_global_registries):
        model_reg, emb_reg = reset_global_registries
        model_reg.put("llm", MagicMock(), ttl=3600, backend="api", repo_id="r/l")
        emb_reg.put("emb", MagicMock(), ttl=3600, backend="local", repo_id="r/e")

        mock_splitter = MagicMock()
        mock_splitter.split_documents.return_value = [MagicMock(), MagicMock(), MagicMock()]

        mock_vectorstore = MagicMock()
        mock_retriever = MagicMock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_faiss = MagicMock()
        mock_faiss.from_documents.return_value = mock_vectorstore

        mock_qa_chain = MagicMock()
        mock_retrieval_qa = MagicMock()
        mock_retrieval_qa.from_chain_type.return_value = mock_qa_chain

        mock_lc_text_splitter = MagicMock()
        mock_lc_text_splitter.RecursiveCharacterTextSplitter.return_value = mock_splitter

        mock_lc_community_vs = MagicMock()
        mock_lc_community_vs.FAISS = mock_faiss

        mock_lc_chains = MagicMock()
        mock_lc_chains.RetrievalQA = mock_retrieval_qa

        mock_lc_schema = MagicMock()

        with patch.dict("sys.modules", {
            "langchain": MagicMock(),
            "langchain.text_splitter": mock_lc_text_splitter,
            "langchain_community": MagicMock(),
            "langchain_community.vectorstores": mock_lc_community_vs,
            "langchain.chains": mock_lc_chains,
            "langchain.schema": mock_lc_schema,
        }):
            result = run(ht.setup_rag_pipeline(
                documents_json='["Doc 1", "Doc 2"]',
            ))

        assert "RAG pipeline created" in result
        assert "Documents ingested: 2" in result
        assert "Chunks created: 3" in result


# ============================================================================
# rag_query()
# ============================================================================


class TestRagQuery:
    """Tests for the rag_query() tool function."""

    def test_no_rag_pipelines(self, reset_global_registries):
        result = run(ht.rag_query(query="What is X?"))
        assert "No RAG pipelines configured" in result

    def test_rag_pipeline_not_found(self, reset_global_registries):
        model_reg, _ = reset_global_registries
        # Add a rag: entry so rag_keys is non-empty, but query for a different one
        model_reg.put("rag:llm+emb", MagicMock(), ttl=3600, backend="rag-chain", repo_id="rag:llm+emb")

        result = run(ht.rag_query(query="What?", pipeline="rag:other"))
        assert "not found" in result

    def test_rag_query_success(self, reset_global_registries):
        model_reg, _ = reset_global_registries
        mock_chain = MagicMock()
        mock_source_doc = MagicMock()
        mock_source_doc.page_content = "Source content here"
        mock_chain.invoke.return_value = {
            "result": "The answer is 42",
            "source_documents": [mock_source_doc],
        }
        model_reg.put("rag:m+e", mock_chain, ttl=3600, backend="rag-chain", repo_id="rag:m+e")

        result = run(ht.rag_query(query="What is the answer?"))
        parsed = json.loads(result)
        assert parsed["answer"] == "The answer is 42"
        assert parsed["sources_used"] == 1

    def test_rag_query_invoke_error(self, reset_global_registries):
        model_reg, _ = reset_global_registries
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = RuntimeError("chain error")
        model_reg.put("rag:m+e", mock_chain, ttl=3600, backend="rag-chain", repo_id="rag:m+e")

        result = run(ht.rag_query(query="question"))
        assert "Error querying RAG pipeline" in result

    def test_rag_query_uses_first_pipeline(self, reset_global_registries):
        """When pipeline is empty, first rag: key should be used."""
        model_reg, _ = reset_global_registries
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "result": "answer",
            "source_documents": [],
        }
        model_reg.put("rag:first+emb", mock_chain, ttl=3600, backend="rag-chain", repo_id="rag:first+emb")

        result = run(ht.rag_query(query="question"))
        parsed = json.loads(result)
        assert parsed["answer"] == "answer"

    def test_rag_query_only_selects_rag_keys(self, reset_global_registries):
        """Non-rag keys should not be considered as rag pipelines."""
        model_reg, _ = reset_global_registries
        model_reg.put("normal-llm", MagicMock(), ttl=3600, backend="api", repo_id="r/l")

        result = run(ht.rag_query(query="question"))
        assert "No RAG pipelines configured" in result


# ============================================================================
# _truncate_value() helper
# ============================================================================


class TestTruncateValue:
    """Tests for the _truncate_value helper."""

    def test_short_value_unchanged(self):
        assert ht._truncate_value("short", max_len=200) == "short"

    def test_long_value_truncated(self):
        long_str = "x" * 300
        result = ht._truncate_value(long_str, max_len=200)
        assert len(result) == 203  # 200 + len("...")
        assert result.endswith("...")

    def test_non_string_converted(self):
        assert ht._truncate_value(12345) == "12345"

    def test_exact_length_not_truncated(self):
        s = "a" * 200
        assert ht._truncate_value(s, max_len=200) == s

    def test_one_over_truncated(self):
        s = "a" * 201
        result = ht._truncate_value(s, max_len=200)
        assert result.endswith("...")
        assert len(result) == 203


# ============================================================================
# _get_hf_token() helper
# ============================================================================


class TestGetHfToken:
    """Tests for the _get_hf_token helper."""

    @patch.dict("os.environ", {"HUGGINGFACE_TOKEN": "hf_test_token"}, clear=False)
    def test_prefers_huggingface_token(self):
        assert ht._get_hf_token() == "hf_test_token"

    @patch.dict("os.environ", {"HF_TOKEN": "hf_alt_token"}, clear=False)
    def test_falls_back_to_hf_token(self):
        import os
        orig = os.environ.pop("HUGGINGFACE_TOKEN", None)
        try:
            assert ht._get_hf_token() == "hf_alt_token"
        finally:
            if orig is not None:
                os.environ["HUGGINGFACE_TOKEN"] = orig

    @patch.dict("os.environ", {}, clear=True)
    def test_returns_none_when_no_env_vars(self):
        assert ht._get_hf_token() is None
