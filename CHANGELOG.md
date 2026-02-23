# Changelog

## [0.3.0] - 2026-02-23

### Added
- HuggingFace + LangChain integration with 24 new MCP tools
- TTL-based model registry with LRU eviction and concurrency caps
- LoRA/PEFT adapter loading with 4-bit/8-bit quantization
- Diffusers integration for image generation (SDXL, FLUX)
- RAG pipeline setup and querying via LangChain + FAISS
- Safe dataset preview with streaming and gated dataset handling
- Batch inference for multiple prompts
- Fine-tuning support via SFTTrainer/LoRA
- Model benchmarking (latency, throughput)
- vLLM/TGI inference server integration
- Audio pipeline (Whisper transcription, TTS)
- GPU memory monitoring
- Model download/cache status checking
- Registry state save/restore for persistence
- Input validation and dangerous command detection
- Rate limiting (execution and model loading)
- Structured audit logging
- Resource limits monitoring (memory, disk space)
- Output secret scanning and redaction
- GitHub Actions CI/CD (lint, test, build, publish)
- Pre-commit hooks (ruff, yaml/toml checks)
- Docker support
- Comprehensive unit test suite

### Changed
- Updated pyproject.toml with optional dependency groups
- Version bumped to 0.3.0

## [0.2.0] - 2026-02-21

### Added
- Initial HuggingFace model management tools
- LangChain integration for text generation and chat
- Embedding model support

## [0.1.0] - 2026-02-20

### Added
- Initial release
- Terminal command execution with session persistence
- Python code execution via IPython
- Smart output handling with prompt/dialog detection
- Cross-platform support (Linux, macOS, Windows experimental)
