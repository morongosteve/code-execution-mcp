"""Example: Setting up a RAG (Retrieval-Augmented Generation) pipeline.

Shows the sequence of MCP tool calls for creating and querying a RAG pipeline.
"""

# Step 1: Load an LLM
step1 = {
    "tool": "hf_load_model",
    "args": {
        "repo_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "backend": "api",
    }
}

# Step 2: Load embeddings
step2 = {
    "tool": "hf_load_embeddings",
    "args": {
        "model_name": "sentence-transformers/all-mpnet-base-v2",
    }
}

# Step 3: Set up RAG pipeline with documents
step3 = {
    "tool": "hf_setup_rag",
    "args": {
        "documents_json": '['
            '"Python is a high-level programming language known for readability.", '
            '"JavaScript is the language of the web, running in browsers.", '
            '"Rust focuses on safety and performance with zero-cost abstractions.", '
            '"Go was designed at Google for simplicity and concurrency."'
        ']',
        "chunk_size": 200,
        "chunk_overlap": 20,
        "search_k": 2,
    }
}

# Step 4: Query the RAG pipeline
step4 = {
    "tool": "hf_rag_query",
    "args": {
        "query": "Which language is best for web development?",
    }
}

print("RAG Pipeline Example")
print("=" * 50)
print("1. Load LLM -> 2. Load Embeddings -> 3. Setup RAG -> 4. Query")
print("See README.md for full documentation.")
