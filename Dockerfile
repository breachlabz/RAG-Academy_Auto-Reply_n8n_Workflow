FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# No embedder is baked in. Embedding is bge-m3 over HTTP (the `embedder`
# container), and rag.core builds every collection with embedding_function=None,
# so Chroma never reconstructs its default MiniLM and never downloads the 80MB
# ONNX blob this image used to carry.

COPY classifier ./classifier
COPY mail ./mail
COPY rag ./rag
COPY threads ./threads
COPY api.py .

# data/ is NOT copied -- it is bind-mounted from the host by compose. The
# thread database lives there, and a copy baked into the image would be
# shadowed by the mount at runtime and silently discarded on every rebuild.

EXPOSE 8100

# EC_* / RAG_* come from compose. No .env is copied in, so the container never
# falls back to the localhost defaults in core.py -- those point at published
# ports that do not exist from inside the network.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8100"]
