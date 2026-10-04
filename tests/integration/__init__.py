"""Opt-in integration tests that talk to a real local Ollama server.

These are skipped by default so ``pytest`` stays offline and fast. To run them:

    ollama serve
    ollama pull qwen2.5-coder:7b
    set AHMED_RAG_LIVE_TESTS=1
    pytest tests/integration -m integration
"""
