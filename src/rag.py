from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import faiss
import numpy as np


@dataclass(frozen=True)
class ChunkMetadata:
    doc_id: str
    text: str


class RAGService:
    def __init__(self, embedder, chunk_size: int = 500, chunk_overlap: int = 50):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be between 0 and chunk_size - 1")
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index = None
        self.metadata: list[ChunkMetadata] = []

    @classmethod
    def from_local_model(cls, model_path: Path | str):
        from sentence_transformers import SentenceTransformer

        model_path = Path(model_path)
        if not model_path.is_dir():
            raise RuntimeError(
                f"Embedding model not found at {model_path}. "
                "Run: uv run python scripts/download_embedding_model.py"
            )
        return cls(SentenceTransformer(str(model_path)))

    @property
    def document_count(self) -> int:
        return len(self.metadata)

    def chunk_text(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        step = self.chunk_size - self.chunk_overlap
        chunks = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size].strip()
            if chunk:
                chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks

    def upload(self, doc_id: str | None, text: str) -> tuple[str, int]:
        doc_id = doc_id or str(uuid4())
        chunks = self.chunk_text(text)
        vectors = self._encode(chunks)
        if self.index is None:
            self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.metadata.extend(ChunkMetadata(doc_id=doc_id, text=chunk) for chunk in chunks)
        return doc_id, len(chunks)

    def search(self, question: str, top_k: int = 3) -> tuple[list[str], list[str]]:
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("No documents have been uploaded.")
        limit = min(top_k, self.index.ntotal)
        _, indices = self.index.search(self._encode([question]), limit)
        matches = [self.metadata[index] for index in indices[0] if index >= 0]
        sources = list(dict.fromkeys(match.doc_id for match in matches))
        return [match.text for match in matches], sources

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.embedder.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype="float32")
