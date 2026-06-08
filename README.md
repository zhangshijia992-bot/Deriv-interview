# Simple RAG API Service

This project is a small local API service that answers questions from a `docs/` folder using retrieval-augmented generation principles.

It does not call external LLM APIs. Retrieval uses TF-IDF and cosine similarity. Answering is extractive: the service returns relevant sentences from the retrieved document chunks.

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── README.md
└── docs/
    ├── product_overview.txt
    ├── privacy_policy.txt
    ├── refund_policy.txt
    └── shipping_policy.txt
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+ is recommended. If you are using a very new Python version and `scikit-learn` does not have a wheel for it yet, use Python 3.10, 3.11, or 3.12.

## Run

```bash
python -m uvicorn App:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

## Endpoints

### Health Check

```bash
curl http://127.0.0.1:8000/
```

### Index Documents

Reads all `.txt` files from the local `docs/` folder, splits them into overlapping chunks, and builds an in-memory TF-IDF index.

```bash
curl -X POST http://127.0.0.1:8000/index
```

Example response:

```json
{
  "status": "success",
  "documents_indexed": 4,
  "chunks_indexed": 15,
  "unreadable_files": []
}
```

### Ask a Question

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What is the refund policy?\"}"
```

Example response:

```json
{
  "answer": "Based on the documents: Customers may request a refund within 14 days of the first payment for a new Pro or Enterprise subscription.",
  "confidence": "answered_from_docs",
  "top_chunks": [
    {
      "source": "refund_policy.txt",
      "chunk_id": 0,
      "score": 0.48,
      "text": "..."
    }
  ]
}
```

If the documents do not contain strong evidence, the API returns:

```json
{
  "answer": "I could not find enough evidence in the indexed documents to answer this question.",
  "confidence": "insufficient_context",
  "top_chunks": []
}
```

## Error Handling

The implementation handles:

- Asking before indexing.
- Empty questions.
- Missing `docs/` folder.
- Empty `docs/` folder.
- Unreadable `.txt` files.
- Questions without strong supporting evidence.

## Tradeoffs and Future Improvements

This solution uses TF-IDF because it is simple, local, fast, and easy to explain in an interview. It is less semantically powerful than embedding models, so it may miss answers when the question uses very different wording from the documents.

With more time, I would improve the project by adding better sentence boundary detection, tests, configurable chunk size and threshold, persistent indexes, richer metadata, and optional local embedding models such as `sentence-transformers`.
