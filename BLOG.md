# Building Food Memory: Retrieval Before Training, At Food-101 Scale

Digit Memory asked a deliberately simple question: what happens if a classifier remembers every example and looks up the closest one? Food Memory asks the same question on a problem where raw pixels are no longer enough.

Food-101 has 101 food categories and roughly 101,000 images. A direct raw-pixel nearest-neighbor system would mostly compare crops, lighting, backgrounds, and plate geometry. Food Memory moves the memory into CLIP embedding space, where semantically similar images are closer, then keeps the original two-tier lookup structure:

1. Hash the canonical RGB image. If this exact image is known, return the stored label immediately.
2. Otherwise, embed the image with `openai/clip-vit-base-patch32`.
3. Search a FAISS index over normalized training embeddings.
4. Vote across the nearest neighbors and return the label, confidence, latency, and examples.

The system is intentionally retrieval-first. No fine-tuning, no epochs, no GPU requirement beyond faster embedding generation. The baselines are there to keep us honest: CLIP zero-shot classification and logistic regression over frozen embeddings.

The interesting object is not a single accuracy number. It is the tradeoff surface:

- exact hash latency versus vector-search latency
- FAISS flat search versus HNSW approximate search
- top-1 versus top-5 accuracy
- retrieval accuracy versus CLIP zero-shot and a lightweight classifier
- robustness under recompression, crop, brightness, and noise

This is the practical lesson carried over from Digit Memory: benchmark the simple memory before assuming the trained model is necessary. On small datasets the memory can win outright. On larger datasets it becomes the baseline every learned system has to beat.

