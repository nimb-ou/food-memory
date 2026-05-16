# Food Memory

Retrieval-first image classification system for Food-101.

I extended a small "Digit Memory" project into a larger Food-101 image retrieval/classification engine. The system stores CLIP embeddings for training images, checks an exact image hash first, and falls back to FAISS nearest-neighbor search with k-NN voting.

Built pieces:

- Hugging Face Food-101 loader
- CLIP image and text embeddings
- SQLite metadata store
- FAISS flat and HNSW indexes
- exact-image cache
- query CLI returning label, confidence, neighbors, exact-hit flag, and latency
- evaluation against CLIP zero-shot and logistic regression over frozen embeddings
- benchmark scripts, reproducible notebook, and portfolio asset generator

The point of the project is simple: before training a bigger model, measure how far a well-engineered memory gets you.
