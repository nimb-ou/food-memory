I rebuilt my "Digit Memory" idea at Food-101 scale.

The original project classified tiny handwritten digits by remembering every training sample and doing hash-first, nearest-neighbor fallback. No training. Just memory plus measurement.

Food Memory keeps the same fundamentals, but the problem is larger and messier:

- 101 food categories
- CLIP embeddings instead of raw pixels
- exact image hashes for repeat inputs
- FAISS flat + HNSW vector search
- k-NN voting over nearest examples
- baselines against CLIP zero-shot and logistic regression on frozen embeddings
- SQLite metadata, reproducible notebooks, benchmarks, and portfolio assets

The question I wanted to test: how strong is retrieval before we reach for fine-tuning?

That is the lesson I keep coming back to. The fastest model is the one you do not need, but you only earn that sentence by measuring it.
