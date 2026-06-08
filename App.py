from pathlib import Path
import re
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DOCS_DIR = Path("docs")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3
MIN_SCORE = 0.15
GENERIC_QUERY_TERMS = {
    "about",
    "derivmate",
    "does",
    "from",
    "have",
    "policy",
    "product",
    "service",
    "support",
    "tell",
    "what",
    "when",
    "where",
    "which",
    "with",
}


app = FastAPI(
    title="Simple RAG API",
    description="A small local RAG-style API using TF-IDF retrieval and extractive answers.",
)


class AskRequest(BaseModel):
    question: str


class RetrievalIndex:
    def __init__(self) -> None:
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix: Any = None
        self.chunks: list[dict[str, Any]] = []
        self.document_count = 0

    @property
    def is_ready(self) -> bool:
        return self.vectorizer is not None and self.tfidf_matrix is not None and bool(self.chunks)


index = RetrievalIndex()


def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        if end < len(cleaned):
            boundary = max(
                cleaned.rfind(". ", start, end),
                cleaned.rfind("? ", start, end),
                cleaned.rfind("! ", start, end),
            )
            if boundary > start + chunk_size // 2:
                end = boundary + 1
            else:
                space = cleaned.rfind(" ", start, end)
                if space > start + chunk_size // 2:
                    end = space

        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break

        start = max(0, end - overlap)
        while start < len(cleaned) and cleaned[start] != " ":
            start += 1
        start = min(start + 1, len(cleaned))

    return chunks


def sentence_split(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def meaningful_question_terms(question: str) -> set[str]:
    terms = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", question.lower()))
    return {term for term in terms if len(term) > 3 and term not in GENERIC_QUERY_TERMS}


def has_meaningful_evidence(question: str, chunk_text: str) -> bool:
    terms = meaningful_question_terms(question)
    if not terms:
        return True

    chunk_words = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", chunk_text.lower()))
    return bool(terms & chunk_words)


def build_extractive_answer(question: str, retrieved_chunks: list[dict[str, Any]]) -> str:
    candidate_sentences: list[dict[str, Any]] = []

    for chunk in retrieved_chunks:
        for sentence in sentence_split(chunk["text"]):
            candidate_sentences.append(
                {
                    "sentence": sentence,
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"],
                }
            )

    if not candidate_sentences:
        combined = " ".join(chunk["text"] for chunk in retrieved_chunks)
        return f"Based on the documents: {combined[:700].strip()}"

    sentence_texts = [item["sentence"] for item in candidate_sentences]
    local_vectorizer = TfidfVectorizer(stop_words="english")
    sentence_matrix = local_vectorizer.fit_transform(sentence_texts + [question])
    question_vector = sentence_matrix[-1]
    sentence_vectors = sentence_matrix[:-1]
    scores = cosine_similarity(question_vector, sentence_vectors).flatten()

    best_indices = scores.argsort()[::-1][:3]
    selected_sentences: list[str] = []
    seen: set[str] = set()
    for sentence_index in best_indices:
        sentence = candidate_sentences[int(sentence_index)]["sentence"]
        if sentence not in seen:
            selected_sentences.append(sentence)
            seen.add(sentence)

    answer = " ".join(selected_sentences)
    return f"Based on the documents: {answer}"


@app.get("/")
def health_check() -> dict[str, str]:
    return {"status": "ok", "message": "Simple RAG API is running."}


@app.post("/index")
def index_documents() -> dict[str, Any]:
    if not DOCS_DIR.exists() or not DOCS_DIR.is_dir():
        return {
            "status": "error",
            "message": "docs/ folder does not exist.",
            "documents_indexed": 0,
            "chunks_indexed": 0,
        }

    txt_files = sorted(DOCS_DIR.glob("*.txt"))
    if not txt_files:
        return {
            "status": "error",
            "message": "No .txt documents found.",
            "documents_indexed": 0,
            "chunks_indexed": 0,
        }

    chunks: list[dict[str, Any]] = []
    unreadable_files: list[str] = []

    for file_path in txt_files:
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = file_path.read_text(encoding="latin-1")
            except Exception:
                unreadable_files.append(file_path.name)
                continue
        except Exception:
            unreadable_files.append(file_path.name)
            continue

        for chunk_id, chunk_text in enumerate(split_into_chunks(text)):
            chunks.append(
                {
                    "text": chunk_text,
                    "source": file_path.name,
                    "chunk_id": chunk_id,
                }
            )

    if not chunks:
        return {
            "status": "error",
            "message": "No readable text content found in docs/.",
            "documents_indexed": 0,
            "chunks_indexed": 0,
            "unreadable_files": unreadable_files,
        }

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(chunk["text"] for chunk in chunks)

    index.vectorizer = vectorizer
    index.tfidf_matrix = tfidf_matrix
    index.chunks = chunks
    index.document_count = len(txt_files) - len(unreadable_files)

    return {
        "status": "success",
        "documents_indexed": index.document_count,
        "chunks_indexed": len(chunks),
        "unreadable_files": unreadable_files,
    }


@app.post("/ask")
def ask_question(request: AskRequest) -> dict[str, Any]:
    question = request.question.strip()
    if not question:
        return {
            "answer": "Question cannot be empty.",
            "confidence": "insufficient_context",
            "top_chunks": [],
        }

    if not index.is_ready:
        return {
            "answer": "Index has not been built yet. Please call POST /index first.",
            "confidence": "insufficient_context",
            "top_chunks": [],
        }

    question_vector = index.vectorizer.transform([question])
    scores = cosine_similarity(question_vector, index.tfidf_matrix).flatten()
    ranked_indices = scores.argsort()[::-1][:TOP_K]

    top_chunks: list[dict[str, Any]] = []
    for chunk_index in ranked_indices:
        score = float(scores[int(chunk_index)])
        chunk = index.chunks[int(chunk_index)]
        top_chunks.append(
            {
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "score": round(score, 4),
                "text": chunk["text"],
            }
        )

    best_chunk = top_chunks[0] if top_chunks else None
    best_score = best_chunk["score"] if best_chunk else 0.0
    has_evidence = bool(best_chunk) and has_meaningful_evidence(question, best_chunk["text"])

    if best_score < MIN_SCORE or not has_evidence:
        return {
            "answer": "I could not find enough evidence in the indexed documents to answer this question.",
            "confidence": "insufficient_context",
            "top_chunks": top_chunks,
        }

    useful_chunks = [
        chunk
        for chunk in top_chunks
        if chunk["score"] >= MIN_SCORE and has_meaningful_evidence(question, chunk["text"])
    ]
    answer = build_extractive_answer(question, useful_chunks)

    return {
        "answer": answer,
        "confidence": "answered_from_docs",
        "top_chunks": top_chunks,
    }
