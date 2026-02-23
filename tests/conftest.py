"""Common fixtures for code-execution-mcp tests."""

import io
import sys
import os
import pytest

# Ensure the project root is on sys.path so we can import modules directly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Patch sys.stdin/stdout for tty_session import compatibility
# ---------------------------------------------------------------------------
# pytest replaces sys.stdin with DontReadFromInput which lacks reconfigure().
# tty_session.py calls sys.stdin.reconfigure() at module level.  We provide
# real TextIOWrapper objects during the import, then restore originals.

_original_stdin = sys.stdin
_original_stdout = sys.stdout

if not hasattr(sys.stdin, "reconfigure"):
    sys.stdin = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
if not hasattr(sys.stdout, "reconfigure"):
    sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")

# Pre-import code_execution_tool so the tty_session reconfigure happens now
try:
    import code_execution_tool as _cet  # noqa: F401
except Exception:
    pass

# Restore original streams for pytest output
sys.stdin = _original_stdin
sys.stdout = _original_stdout


# ---------------------------------------------------------------------------
# _ModelRegistry helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_registry():
    """Return a factory that creates a fresh _ModelRegistry with a given capacity."""
    from huggingface_tools import _ModelRegistry

    def _factory(max_models: int = 5):
        return _ModelRegistry(max_models=max_models)

    return _factory


@pytest.fixture
def populated_registry(fresh_registry):
    """Return a registry pre-loaded with 3 dummy models (capacity=5)."""
    reg = fresh_registry(max_models=5)
    for i in range(3):
        reg.put(f"model-{i}", f"dummy-obj-{i}", ttl=3600, backend="test", repo_id=f"repo/{i}")
    return reg


# ---------------------------------------------------------------------------
# Global-registry reset fixture (used by HF tool tests)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=False)
def reset_global_registries():
    """Reset the module-level _model_registry and _embedding_registry before
    and after each test that opts in to this fixture."""
    import huggingface_tools as ht

    # Save originals
    orig_model = ht._model_registry
    orig_embed = ht._embedding_registry

    # Replace with fresh instances so tests are isolated
    ht._model_registry = ht._ModelRegistry(max_models=ht.MAX_CONCURRENT_MODELS)
    ht._embedding_registry = ht._ModelRegistry(max_models=ht.MAX_CONCURRENT_EMBEDDINGS)

    yield ht._model_registry, ht._embedding_registry

    # Restore originals
    ht._model_registry = orig_model
    ht._embedding_registry = orig_embed


# ---------------------------------------------------------------------------
# CodeExecutionTool fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def code_exec_tool():
    """Return a CodeExecutionTool instance pointed at the real prompts dir."""
    from code_execution_tool import CodeExecutionTool

    return CodeExecutionTool()


# ---------------------------------------------------------------------------
# Mock model helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm_model():
    """Return a MagicMock that behaves like a LangChain LLM."""
    from unittest.mock import MagicMock
    mock = MagicMock(name="MockLLM")
    mock.invoke.return_value = "mock llm output"
    return mock


@pytest.fixture
def mock_embedding_model():
    """Return a MagicMock that behaves like a LangChain embedding model."""
    from unittest.mock import MagicMock
    mock = MagicMock(name="MockEmbedding")
    mock.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    mock.embed_query.return_value = [0.1, 0.2, 0.3]
    return mock
