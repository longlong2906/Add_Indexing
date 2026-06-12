from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from uuid import uuid4

import faiss
import numpy as np


@dataclass(frozen=True)
class ChunkMetadata:
    doc_id: str
    chunk_id: str
    text: str
    position: int


class RAGService:
    SENTENCE_PATTERN = re.compile(r"[^.!?\n]+(?:[.!?]+|$)", re.UNICODE)

    def __init__(
        self,
        embedder,
        chunk_size: int = 900,
        chunk_overlap: int = 150,
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 96,
        storage_path: Path | str | None = None,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be between 0 and chunk_size - 1")
        if hnsw_m <= 0:
            raise ValueError("hnsw_m must be positive")
        if hnsw_ef_construction <= 0:
            raise ValueError("hnsw_ef_construction must be positive")
        if hnsw_ef_search <= 0:
            raise ValueError("hnsw_ef_search must be positive")
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        self.storage_path = Path(storage_path) if storage_path is not None else None
        self.index = None
        self.metadata: list[ChunkMetadata] = []
        self._load_storage()

    @classmethod
    def from_local_model(cls, model_path: Path | str, storage_path: Path | str | None = None):
        from sentence_transformers import SentenceTransformer

        model_path = Path(model_path)
        if not model_path.is_dir():
            raise RuntimeError(
                f"Embedding model not found at {model_path}. "
                "Run: uv run python scripts/download_embedding_model.py"
            )
        return cls(SentenceTransformer(str(model_path)), storage_path=storage_path)

    @property
    def document_count(self) -> int:
        return len(self.metadata)

    def chunk_text(self, text: str) -> list[str]:
        text = self._normalize_text(text)
        if not text:
            return []

        chunks: list[str] = []
        current_words: list[str] = []
        for sentence in self._split_sentences(text):
            sentence_words = sentence.split()
            if not sentence_words:
                continue
            if len(sentence_words) > self.chunk_size:
                chunks.extend(self._flush_chunk(current_words))
                current_words = []
                chunks.extend(self._split_long_words(sentence_words))
                continue
            if current_words and len(current_words) + len(sentence_words) > self.chunk_size:
                chunks.extend(self._flush_chunk(current_words))
                current_words = self._overlap_words(current_words)
            current_words.extend(sentence_words)

        chunks.extend(self._flush_chunk(current_words))
        return chunks

    def upload(self, doc_id: str | None, text: str) -> tuple[str, int]:
        doc_id = doc_id or str(uuid4())
        chunks = self.chunk_text(text)
        if not chunks:
            return doc_id, 0
        vectors = self._encode_passages(chunks)
        if self.index is None:
            self.index = self._create_index(vectors.shape[1])
        self.index.add(vectors)
        self.metadata.extend(
            ChunkMetadata(
                doc_id=doc_id,
                chunk_id=f"{doc_id}:{position}",
                text=chunk,
                position=position,
            )
            for position, chunk in enumerate(chunks)
        )
        self._save_storage()
        return doc_id, len(chunks)

    def search(self, question: str, top_k: int = 3) -> tuple[list[str], list[str]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("No documents have been uploaded.")
        limit = min(max(top_k * 4, top_k), self.index.ntotal)
        self.index.hnsw.efSearch = max(self.hnsw_ef_search, limit)
        _, indices = self.index.search(self._encode_query(question), limit)
        matches = [self.metadata[index] for index in indices[0] if index >= 0][:top_k]
        sources = list(dict.fromkeys(match.doc_id for match in matches))
        return [match.text for match in matches], sources

    def _create_index(self, dimension: int):
        index = faiss.IndexHNSWFlat(dimension, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.hnsw_ef_construction
        index.hnsw.efSearch = self.hnsw_ef_search
        return index

    @property
    def _index_path(self) -> Path | None:
        if self.storage_path is None:
            return None
        return self.storage_path / "index.faiss"

    @property
    def _metadata_path(self) -> Path | None:
        if self.storage_path is None:
            return None
        return self.storage_path / "metadata.json"

    def _load_storage(self) -> None:
        index_path = self._index_path
        metadata_path = self._metadata_path
        if index_path is None or metadata_path is None:
            return
        if not index_path.is_file() or not metadata_path.is_file():
            return
        self.index = faiss.read_index(str(index_path))
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        self.metadata = [ChunkMetadata(**item) for item in metadata]

    def _save_storage(self) -> None:
        index_path = self._index_path
        metadata_path = self._metadata_path
        if index_path is None or metadata_path is None or self.index is None:
            return
        self.storage_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump([asdict(item) for item in self.metadata], file, ensure_ascii=False, indent=2)

    def _encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode([f"passage: {text}" for text in texts])

    def _encode_query(self, text: str) -> np.ndarray:
        return self._encode([f"query: {text}"])

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.embedder.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype="float32")

    def _split_sentences(self, text: str) -> list[str]:
        sentences: list[str] = []
        for paragraph in text.split("\n"):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            matches = [match.group(0).strip() for match in self.SENTENCE_PATTERN.finditer(paragraph)]
            sentences.extend(match for match in matches if match)
        return sentences

    def _split_long_words(self, words: list[str]) -> list[str]:
        chunks: list[str] = []
        step = self.chunk_size - self.chunk_overlap
        for start in range(0, len(words), step):
            chunk_words = words[start : start + self.chunk_size]
            chunks.extend(self._flush_chunk(chunk_words))
            if start + self.chunk_size >= len(words):
                break
        return chunks

    def _overlap_words(self, words: list[str]) -> list[str]:
        if self.chunk_overlap == 0:
            return []
        return words[-self.chunk_overlap :]

    def _flush_chunk(self, words: list[str]) -> list[str]:
        chunk = " ".join(words).strip()
        return [chunk] if chunk else []

    def _normalize_text(self, text: str) -> str:
        lines = [" ".join(line.split()) for line in text.strip().splitlines()]
        return "\n".join(line for line in lines if line)
