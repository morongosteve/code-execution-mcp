# Code Execution MCP Server

A Model Context Protocol (MCP) server that exposes [Agent Zero's](https://github.com/agent0ai/agent-zero) battle-tested code execution capabilities.

This MCP server allows any AI agent (Claude, Cursor, Windsurf, etc.) to execute terminal commands and Python code on the host system using Agent Zero's proven implementation.

## Features

- **Execute Terminal Commands**: Run shell commands with full session persistence
- **Execute Python Code**: Run Python code via IPython with session management
- **Multiple Sessions**: Maintain separate execution contexts
- **Smart Output Handling**: Automatic prompt detection, timeout management, and dialog detection
- **Cross-Platform**: Works on Linux, macOS, and Windows (experimental)
- **HuggingFace Integration**: Load and run models, embeddings, RAG pipelines, image generation, and more
- **Security**: Input validation, rate limiting, audit logging, and output secret scanning

## MCP Client Configuration (no installation needed)

Add to your application MCP config:

- simple case using uvx

```json
{
  "mcpServers": {
    "code-execution": {
      "command": "uvx",
      "args": ["code-execution-mcp"]
    }
  }
}
```

- or pipx if uvx is not installed

```json
{
  "mcpServers": {
    "code-execution": {
      "command": "pipx",
      "args": ["run", "code-execution-mcp"]
    }
  }
}
```

## Additional configuration

The MCP server can be configured via environment variables:

```bash
# Shell executable (default: /bin/bash on Unix, powershell.exe on Windows)
export CODE_EXEC_EXECUTABLE=/bin/bash

# Init commands (semicolon-separated, run when creating new sessions, empty by default)
export CODE_EXEC_INIT_COMMANDS="source /path/to/venv/bin/activate;export PATH=\$PATH:/custom/bin"

# Timeout configuration (in seconds)
export CODE_EXEC_FIRST_OUTPUT_TIMEOUT=30      # Wait for first output
export CODE_EXEC_BETWEEN_OUTPUT_TIMEOUT=15    # Wait between output chunks
export CODE_EXEC_DIALOG_TIMEOUT=5             # Detect dialog prompts
export CODE_EXEC_MAX_EXEC_TIMEOUT=180         # Maximum execution time

# Log directory (default empty = logging disabled)
export CODE_EXEC_LOG_DIR=/path/to/logs
```

### Additional examples

- start sessions with custom shell and python environment + logging

```json
{
  "mcpServers": {
    "code-execution-mcp": {
      "command": "uvx",
      "args": ["code-execution-mcp"],
      "env": {
        "CODE_EXEC_EXECUTABLE": "/bin/zsh",
        "CODE_EXEC_INIT_COMMANDS": "source /Users/lazy/Projects/code-execution-mcp/.venv/bin/activate",
        "CODE_EXEC_LOG_DIR": "/Users/lazy/Projects/code-execution-mcp/logs"
      }
    }
  }
}
```

- override timeouts

```json
{
  "mcpServers": {
    "code-execution-mcp": {
      "command": "uvx",
      "args": ["code-execution-mcp"],
      "env": {
        "CODE_EXEC_FIRST_OUTPUT_TIMEOUT": "60",
        "CODE_EXEC_MAX_EXEC_TIMEOUT": "300"
      }
    }
  }
}
```

## Manual installation

```bash
# Clone or download this package, then navigate to the directory
git clone https://github.com/agent0ai/code-execution-mcp.git
cd </path/to>/code-execution-mcp

# Install dependencies
pip install -e .
```

and run with config:

```json
{
  "mcpServers": {
    "code-execution-mcp": {
      "command": "python",
      "args": ["</path/to/code-execution-mcp>/main.py"]
    }
  }
}
```

## Available Tools

### execute_terminal

Execute a terminal command in the specified session.

**Parameters:**

- `command` (string, required): The shell command to execute
- `session` (integer, optional): Session (terminal window) number (default: 0)

**Output:**

- (string) The accumulated terminal output from the session

### execute_python

Execute Python code via IPython in the specified session.

**Parameters:**

- `code` (string, required): The Python code to execute
- `session` (integer, optional): Session (terminal window) number (default: 0)

**Output:**

- (string) The accumulated IPython output from the session

### get_output

Get accumulated output from a terminal session.

**Parameters:**

- `session` (integer, optional): Session (terminal window) number (default: 0)

**Output:**

- (string) The accumulated terminal output from the session

### reset_terminal

Reset a terminal session, closing and reopening it.

**Parameters:**

- `session` (integer, optional): Session (terminal window) number (default: 0)
- `reason` (string, optional): Reason for the reset

**Output:**

- (string) Text confirmation for the agent

## HuggingFace Integration Tools

The MCP server includes 24 tools for working with HuggingFace models, embeddings, datasets, and pipelines via LangChain.

### Setup

```bash
# Install with HuggingFace support
pip install code-execution-mcp[huggingface]

# Install with all features
pip install code-execution-mcp[all]

# Set your HuggingFace token (optional, for gated models)
export HUGGINGFACE_TOKEN=hf_...
```

### hf_list_models

List all currently loaded models and their metadata in the registry.

**Parameters:** None

**Example:**
```
hf_list_models
```

### hf_load_model

Load a HuggingFace model for text generation via API or locally.

**Parameters:**

- `repo_id` (string, required): HuggingFace model repository ID (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`)
- `backend` (string, optional): `"api"` for HuggingFace Inference API, `"local"` for local loading (default: `"api"`)
- `max_new_tokens` (integer, optional): Maximum new tokens to generate (default: 256)
- `temperature` (float, optional): Sampling temperature (default: 0.7)

**Example:**
```
hf_load_model repo_id="meta-llama/Meta-Llama-3-8B-Instruct" backend="api" max_new_tokens=512 temperature=0.8
```

### hf_load_chat_model

Load a HuggingFace model optimized for chat/conversational use.

**Parameters:**

- `repo_id` (string, required): HuggingFace model repository ID
- `backend` (string, optional): `"api"` or `"local"` (default: `"api"`)
- `max_new_tokens` (integer, optional): Maximum new tokens (default: 256)
- `temperature` (float, optional): Sampling temperature (default: 0.7)

**Example:**
```
hf_load_chat_model repo_id="mistralai/Mistral-7B-Instruct-v0.2" backend="api"
```

### hf_load_embeddings

Load an embedding model for text embedding tasks.

**Parameters:**

- `model_name` (string, required): Model name (e.g., `sentence-transformers/all-mpnet-base-v2`)
- `backend` (string, optional): `"local"` for local inference, `"api"` for Inference API (default: `"local"`)

**Example (local):**
```
hf_load_embeddings model_name="sentence-transformers/all-mpnet-base-v2" backend="local"
```

**Example (API):**
```
hf_load_embeddings model_name="sentence-transformers/all-mpnet-base-v2" backend="api"
```

### hf_text_generate

Generate text using a loaded model.

**Parameters:**

- `prompt` (string, required): The text prompt
- `model` (string, optional): Model name to use (uses default if not specified)

**Example:**
```
hf_text_generate prompt="Explain quantum computing in simple terms:" model="meta-llama/Meta-Llama-3-8B-Instruct"
```

### hf_chat_complete

Send a chat conversation to a loaded chat model.

**Parameters:**

- `messages_json` (string, required): JSON array of message objects with `role` and `content` fields
- `model` (string, optional): Model name to use

**Example:**
```json
hf_chat_complete messages_json='[
  {"role": "system", "content": "You are a helpful coding assistant."},
  {"role": "user", "content": "Write a Python function to sort a list."}
]'
```

### hf_embed_texts

Generate embeddings for a list of texts.

**Parameters:**

- `texts_json` (string, required): JSON array of text strings
- `embed_type` (string, optional): `"documents"` or `"queries"` (default: `"documents"`)

**Example:**
```json
hf_embed_texts texts_json='["Hello world", "How are you?", "Machine learning is great"]' embed_type="documents"
```

### hf_unload_model

Unload a model from the registry to free resources.

**Parameters:**

- `model_name` (string, required): Name of the model to unload

**Example:**
```
hf_unload_model model_name="meta-llama/Meta-Llama-3-8B-Instruct"
```

### hf_run_pipeline

Run a HuggingFace pipeline for common NLP tasks.

**Parameters:**

- `task` (string, required): Pipeline task name
- `input_text` (string, required): Input text to process
- `model` (string, optional): Model to use (uses task default if not specified)

**Supported tasks:**

- `sentiment-analysis`
- `text-classification`
- `ner` (Named Entity Recognition)
- `question-answering`
- `summarization`
- `translation`
- `zero-shot-classification`
- `fill-mask`
- `text2text-generation`

**Example:**
```
hf_run_pipeline task="sentiment-analysis" input_text="I love this product!"
```

### hf_preview_dataset

Preview rows from a HuggingFace dataset with streaming support.

**Parameters:**

- `dataset_name` (string, required): Dataset name on HuggingFace Hub
- `split` (string, optional): Dataset split (default: `"train"`)
- `num_rows` (integer, optional): Number of rows to preview (default: 5)

**Example:**
```
hf_preview_dataset dataset_name="squad" split="train" num_rows=3
```

### hf_load_peft_model

Load a PEFT/LoRA adapter on top of a base model with optional quantization.

**Parameters:**

- `base_model` (string, required): Base model repository ID
- `adapter_id` (string, required): PEFT adapter repository ID
- `quantize` (string, optional): Quantization level: `"4bit"`, `"8bit"`, or `null` (default: `null`)
- `max_new_tokens` (integer, optional): Maximum new tokens (default: 256)

**Example:**
```
hf_load_peft_model base_model="meta-llama/Meta-Llama-3-8B" adapter_id="my-org/llama3-lora-adapter" quantize="4bit"
```

### hf_generate_image

Generate an image from a text prompt using diffusion models.

**Parameters:**

- `prompt` (string, required): Text description of the image
- `backend` (string, optional): `"api"` or `"local"` (default: `"api"`)
- `model` (string, optional): Model to use (default: `"stabilityai/stable-diffusion-xl-base-1.0"`)
- `output_path` (string, optional): File path to save the image

**Example (API):**
```
hf_generate_image prompt="A futuristic city at sunset, digital art" backend="api"
```

**Example (local):**
```
hf_generate_image prompt="A cat wearing a top hat" backend="local" model="stabilityai/stable-diffusion-xl-base-1.0" output_path="output.png"
```

### hf_setup_rag

Set up a Retrieval-Augmented Generation (RAG) pipeline with documents.

**Parameters:**

- `documents_json` (string, required): JSON array of document strings
- `chunk_size` (integer, optional): Text chunk size for splitting (default: 500)
- `chunk_overlap` (integer, optional): Overlap between chunks (default: 50)
- `search_k` (integer, optional): Number of results to retrieve (default: 4)

**Example:**
```json
hf_setup_rag documents_json='[
  "Python is a high-level programming language known for readability.",
  "JavaScript is the language of the web, running in browsers.",
  "Rust focuses on safety and performance with zero-cost abstractions."
]' chunk_size=200 chunk_overlap=20 search_k=2
```

### hf_rag_query

Query the configured RAG pipeline.

**Parameters:**

- `query` (string, required): The question to ask

**Example:**
```
hf_rag_query query="Which language is best for web development?"
```

### hf_batch_generate

Generate text for multiple prompts in batch.

**Parameters:**

- `prompts_json` (string, required): JSON array of prompt strings
- `model` (string, optional): Model to use

**Example:**
```json
hf_batch_generate prompts_json='["Explain gravity", "What is photosynthesis?", "Define entropy"]'
```

### hf_finetune

Fine-tune a model using SFTTrainer with LoRA.

**Parameters:**

- `model_name` (string, required): Base model to fine-tune
- `dataset_name` (string, required): HuggingFace dataset for training
- `output_dir` (string, optional): Directory to save the fine-tuned model
- `num_epochs` (integer, optional): Number of training epochs (default: 3)
- `learning_rate` (float, optional): Learning rate (default: 2e-4)

**Example:**
```
hf_finetune model_name="meta-llama/Meta-Llama-3-8B" dataset_name="tatsu-lab/alpaca" output_dir="./finetuned-model" num_epochs=1
```

### hf_benchmark

Benchmark a loaded model for latency and throughput.

**Parameters:**

- `model` (string, optional): Model to benchmark
- `num_runs` (integer, optional): Number of benchmark iterations (default: 5)

**Example:**
```
hf_benchmark model="meta-llama/Meta-Llama-3-8B-Instruct" num_runs=10
```

### hf_load_vllm_model

Connect to a vLLM or TGI inference server.

**Parameters:**

- `model_name` (string, required): Model name on the server
- `base_url` (string, required): Server URL (e.g., `http://localhost:8000/v1`)
- `max_new_tokens` (integer, optional): Maximum new tokens (default: 256)

**Example:**
```
hf_load_vllm_model model_name="meta-llama/Meta-Llama-3-8B-Instruct" base_url="http://localhost:8000/v1"
```

### hf_audio_transcribe

Transcribe audio using Whisper.

**Parameters:**

- `audio_path` (string, required): Path to the audio file
- `model` (string, optional): Whisper model to use (default: `"openai/whisper-base"`)

**Example:**
```
hf_audio_transcribe audio_path="/path/to/audio.wav" model="openai/whisper-large-v3"
```

### hf_text_to_speech

Convert text to speech audio.

**Parameters:**

- `text` (string, required): Text to convert to speech
- `output_path` (string, optional): Path to save the audio file
- `model` (string, optional): TTS model to use

**Example:**
```
hf_text_to_speech text="Hello, this is a test of text to speech." output_path="output.wav"
```

### hf_gpu_status

Check GPU memory usage and availability.

**Parameters:** None

**Example:**
```
hf_gpu_status
```

### hf_model_download_status

Check model download and cache status.

**Parameters:**

- `repo_id` (string, required): Model repository ID to check

**Example:**
```
hf_model_download_status repo_id="meta-llama/Meta-Llama-3-8B-Instruct"
```

### hf_save_registry

Save the current model registry state to disk for persistence.

**Parameters:**

- `path` (string, optional): File path to save the registry state

**Example:**
```
hf_save_registry path="./registry_state.json"
```

### hf_restore_registry

Restore a previously saved model registry state.

**Parameters:**

- `path` (string, optional): File path to restore from

**Example:**
```
hf_restore_registry path="./registry_state.json"
```

## Model Registry

The model registry manages loaded models with automatic resource management:

- **TTL Eviction**: Models are automatically unloaded after a configurable time-to-live period of inactivity, freeing GPU/CPU memory.
- **LRU (Least Recently Used)**: When the maximum number of concurrent models is reached, the least recently used model is evicted to make room for new ones.
- **Max Concurrent Limits**: Configure the maximum number of models that can be loaded simultaneously to prevent out-of-memory errors.
- **Listing**: Use `hf_list_models` to see all currently loaded models, their types, backends, and last-used timestamps.
- **Manual Unloading**: Use `hf_unload_model` to explicitly free a model's resources.
- **Save/Restore**: Use `hf_save_registry` and `hf_restore_registry` to persist and recover registry state across server restarts.

## Optional Dependencies

The package uses optional dependency groups to keep the base installation lightweight:

| Group | Install Command | Description |
|-------|----------------|-------------|
| `huggingface` | `pip install code-execution-mcp[huggingface]` | Core HuggingFace + LangChain integration |
| `peft` | `pip install code-execution-mcp[peft]` | LoRA/PEFT adapter support with bitsandbytes quantization |
| `diffusers` | `pip install code-execution-mcp[diffusers]` | Image generation (Stable Diffusion, SDXL, FLUX) |
| `rag` | `pip install code-execution-mcp[rag]` | RAG pipeline with FAISS vector store |
| `audio` | `pip install code-execution-mcp[audio]` | Audio transcription (Whisper) and text-to-speech |
| `training` | `pip install code-execution-mcp[training]` | Fine-tuning with SFTTrainer and LoRA |
| `vllm` | `pip install code-execution-mcp[vllm]` | vLLM/TGI inference server integration |
| `all` | `pip install code-execution-mcp[all]` | All optional dependencies |

## Session Management

- Sessions (terminal instances) allow maintaining separate execution contexts for multitasking, persistence or context isolation
- Each session can be used and reset individually
- Sessions persist until reset
- Session 0 is default
- Any session number can be used


## Virtual Environment Considerations

**Important:** When the MCP server is launched from a virtual environment, shell sessions may NOT automatically inherit the venv activation.

**Solution:** Use init commands to explicitly activate your virtual environment:

```json
{
  "env": {
    "CODE_EXEC_INIT_COMMANDS": "source /path/to/venv/bin/activate"
  }
}
```

## Docker

Build and run the MCP server in a Docker container:

```bash
docker build -t code-execution-mcp .
docker run -it code-execution-mcp
```

To pass environment variables:

```bash
docker run -it \
  -e HUGGINGFACE_TOKEN=hf_... \
  -e CODE_EXEC_LOG_DIR=/logs \
  code-execution-mcp
```

## Platform Support

- **Linux**: Fully tested and supported
- **macOS**: Fully tested and supported
- **Windows**: Experimental support via pywinpty
  - Some features may behave differently

## Architecture

This MCP server is a **minimal wrapper** around Agent Zero's code execution tool:

- Preserves Agent Zero's battle-tested logic
- No rewrites or reimplementations
- Uses Agent Zero's helper modules unchanged:
  - `tty_session.py` - TTY session management
  - `shell_local.py` - Local shell interface
  - `print_style.py` - Output styling and logging
  - `strings.py` - String manipulation utilities

## Security

**WARNING:** This MCP server allows full code execution on the host system. Security is the responsibility of the MCP client.

The server includes several security layers for the HuggingFace integration:

- **Input Validation**: Dangerous commands and code patterns are detected and blocked
- **Rate Limiting**: Configurable rate limits for code execution and model loading operations
- **Audit Logging**: Structured logging of all tool invocations and security events
- **Resource Monitoring**: Memory and disk space checks before loading models
- **Output Scanning**: Automatic detection and redaction of secrets (API keys, tokens) in output

Only use with trusted AI agents and in controlled environments.

## Development

```bash
# Clone the repository
git clone https://github.com/agent0ai/code-execution-mcp.git
cd code-execution-mcp

# Install with all development dependencies
pip install -e ".[all]"

# Run tests
pytest

# Lint
ruff check .

# Format
ruff format .
```

## License

MIT License

This project wraps and reuses code from [Agent Zero](https://github.com/agent0ai/agent-zero) (Copyright (c) 2025 Agent Zero, s.r.o), which is licensed under the MIT License.

See the LICENSE file for full license text and attribution details.

## Credits

**Built on Agent Zero's proven code execution implementation.**

This MCP server preserves and reuses Agent Zero's battle-tested code:

- Core execution logic from `code_execution_tool.py`
- Helper modules: `tty_session.py`, `shell_local.py`, `print_style.py`, `strings.py`
- All system message prompts

All credit for the robust code execution implementation goes to the [Agent Zero](https://github.com/agent0ai/agent-zero) team.

Uses [FastMCP](https://gofastmcp.com/) for MCP protocol handling.
