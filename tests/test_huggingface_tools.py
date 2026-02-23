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


# ============================================================================
# hf_dataset_info() -- num_rows clamping
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
                num_rows=999,  # way above MAX_DATASET_PREVIEW_ROWS (50)
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
