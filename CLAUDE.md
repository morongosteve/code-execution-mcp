# CLAUDE.md

## Project Overview

An MCP (Model Context Protocol) server that wraps Agent Zero's battle-tested code execution engine. Exposes terminal command execution, Python/IPython execution, and optional HuggingFace model management to any MCP-compatible client (Claude, Cursor, Windsurf, etc.).

## Runtime

- Python ≥ 3.10
- MCP framework: **FastMCP** ≥ 2.0
- Entry point: `main.py` → `main()` (also `code-execution-mcp` script via pip)

## Development Commands

```bash
# Install base + all optional deps
pip install -e ".[all]"

# Run tests
pytest

# Lint / format
ruff check .
ruff format .

# Type checking
mypy .

# Run server directly
python main.py
```

## Project Structure

```
main.py                  # FastMCP server setup and tool registration
code_execution_tool.py   # Core execution logic (Agent Zero, unchanged)
huggingface_tools.py     # 24 HuggingFace integration tools
security.py              # Input validation, rate limiting, audit logging
helpers/
  tty_session.py         # TTY session management (Agent Zero)
  shell_local.py         # Local shell interface (Agent Zero)
  print_style.py         # Output styling (Agent Zero)
  strings.py             # String utilities (Agent Zero)
prompts/                 # System message prompts
tests/                   # pytest test suite
docs/                    # Extended documentation
examples/                # Usage examples
```

## Architecture

The server is a **thin wrapper** — Agent Zero's `code_execution_tool.py` and `helpers/` are used unmodified. `main.py` registers tools with FastMCP; `huggingface_tools.py` adds ML-specific tools on top.

Session model: numbered sessions (0..N) each backed by a persistent TTY. Multiple sessions can run in parallel for task isolation.

## Configuration (Environment Variables)

| Variable | Default | Description |
|---|---|---|
| `CODE_EXEC_EXECUTABLE` | `/bin/bash` | Shell to use |
| `CODE_EXEC_INIT_COMMANDS` | `` | Semicolon-separated init commands per session |
| `CODE_EXEC_FIRST_OUTPUT_TIMEOUT` | `30` | Seconds to wait for first output |
| `CODE_EXEC_BETWEEN_OUTPUT_TIMEOUT` | `15` | Seconds between output chunks |
| `CODE_EXEC_MAX_EXEC_TIMEOUT` | `180` | Maximum total execution time |
| `CODE_EXEC_LOG_DIR` | `` | Audit log directory (empty = disabled) |

## Optional Dependency Groups

Install only what you need:

```bash
pip install code-execution-mcp[huggingface]   # LangChain + Transformers
pip install code-execution-mcp[peft]          # LoRA / bitsandbytes quantization
pip install code-execution-mcp[diffusers]     # Image generation
pip install code-execution-mcp[rag]           # FAISS RAG pipeline
pip install code-execution-mcp[audio]         # Whisper transcription + TTS
pip install code-execution-mcp[training]      # SFTTrainer fine-tuning
pip install code-execution-mcp[vllm]          # vLLM/TGI server integration
pip install code-execution-mcp[all]           # Everything above
```

## Key Conventions

- Do **not** rewrite Agent Zero's helper files (`tty_session.py`, `shell_local.py`, etc.) — they are intentionally preserved as-is
- All tool handlers must return a **JSON string**
- HuggingFace tools use a model registry (`huggingface_tools.py`) with LRU/TTL eviction — always call `hf_unload_model` when done to free VRAM
- Security validation in `security.py` runs before execution; do not bypass it for convenience
- Virtual environments are not automatically inherited by shell sessions — use `CODE_EXEC_INIT_COMMANDS` to activate them
