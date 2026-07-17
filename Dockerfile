FROM python:3.11-slim

ARG EMBEDDER_MODEL=sentence-transformers/all-mpnet-base-v2
ARG RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
ARG GENERATOR_MODEL=google/flan-t5-large
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RAG_DATA_DIR=/app/pdfs \
    RAG_EMBED_MODEL=$EMBEDDER_MODEL RAG_RERANK_MODEL=$RERANKER_MODEL RAG_GENERATOR_MODEL=$GENERATOR_MODEL \
    HF_HOME=/models
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from sentence_transformers import SentenceTransformer; from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer; SentenceTransformer('$EMBEDDER_MODEL'); AutoTokenizer.from_pretrained('$RERANKER_MODEL'); AutoModelForSequenceClassification.from_pretrained('$RERANKER_MODEL'); AutoTokenizer.from_pretrained('$GENERATOR_MODEL'); AutoModelForSeq2SeqLM.from_pretrained('$GENERATOR_MODEL')"
RUN useradd --create-home rag && chown -R rag:rag /app /models
COPY --chown=rag:rag modelcard_rag.py api.py ./
COPY --chown=rag:rag pdfs ./pdfs

ENV RAG_OFFLINE=1
USER rag
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
