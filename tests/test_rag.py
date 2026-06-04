import numpy as np

from rag import RAGService


class FakeEmbedder:
    def encode(self, texts, normalize_embeddings=True):
        vectors = {
            "alpha": [1.0, 0.0],
            "beta": [0.0, 1.0],
            "alpha question": [1.0, 0.0],
        }
        return np.asarray([vectors[text] for text in texts], dtype="float32")


def test_chunk_text_uses_overlap_and_drops_blank_chunks():
    service = RAGService(FakeEmbedder(), chunk_size=5, chunk_overlap=2)

    assert service.chunk_text("abcdefghij   ") == ["abcde", "defgh", "ghij"]


def test_upload_generates_doc_id_and_accumulates_duplicate_ids():
    service = RAGService(FakeEmbedder())

    generated_id, generated_chunks = service.upload(None, "alpha")
    _, duplicate_chunks = service.upload("shared", "alpha")
    _, more_duplicate_chunks = service.upload("shared", "beta")

    assert generated_id
    assert generated_chunks == 1
    assert duplicate_chunks == 1
    assert more_duplicate_chunks == 1
    assert service.document_count == 3


def test_search_returns_ranked_chunks_and_unique_source_ids():
    service = RAGService(FakeEmbedder())
    service.upload("doc-alpha", "alpha")
    service.upload("doc-beta", "beta")
    service.upload("doc-alpha", "alpha")

    chunks, sources = service.search("alpha question", top_k=3)

    assert chunks[:2] == ["alpha", "alpha"]
    assert sources == ["doc-alpha", "doc-beta"]


def test_search_rejects_empty_index():
    service = RAGService(FakeEmbedder())

    try:
        service.search("alpha question")
    except ValueError as exc:
        assert str(exc) == "No documents have been uploaded."
    else:
        raise AssertionError("search should reject an empty index")
