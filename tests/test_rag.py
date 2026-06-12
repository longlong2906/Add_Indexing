import faiss
import numpy as np

from rag import RAGService


class FakeEmbedder:
    def __init__(self):
        self.encoded_texts = []

    def encode(self, texts, normalize_embeddings=True):
        self.encoded_texts.extend(texts)
        vectors = {
            "alpha": [1.0, 0.0],
            "beta": [0.0, 1.0],
            "gamma": [0.8, 0.2],
            "delta": [0.7, 0.3],
            "alpha question": [1.0, 0.0],
        }
        clean_texts = [text.removeprefix("passage: ").removeprefix("query: ") for text in texts]
        return np.asarray([vectors[text] for text in clean_texts], dtype="float32")


def test_chunk_text_prefers_vietnamese_sentence_boundaries():
    service = RAGService(FakeEmbedder(), chunk_size=12, chunk_overlap=2)

    text = "FastAPI la framework Python. No phu hop de tao API nhanh! Ban co the upload tai lieu."

    assert service.chunk_text(text) == [
        "FastAPI la framework Python. No phu hop de tao API nhanh!",
        "API nhanh! Ban co the upload tai lieu.",
    ]


def test_chunk_text_splits_long_sentence_with_word_overlap():
    service = RAGService(FakeEmbedder(), chunk_size=4, chunk_overlap=1)

    assert service.chunk_text("mot hai ba bon nam sau bay") == [
        "mot hai ba bon",
        "bon nam sau bay",
    ]


def test_chunk_text_drops_blank_chunks():
    service = RAGService(FakeEmbedder())

    assert service.chunk_text(" \n\t ") == []


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


def test_upload_and_search_use_e5_passage_and_query_prefixes():
    embedder = FakeEmbedder()
    service = RAGService(embedder)
    service.upload("doc-alpha", "alpha")

    chunks, sources = service.search("alpha question", top_k=1)

    assert chunks == ["alpha"]
    assert sources == ["doc-alpha"]
    assert embedder.encoded_texts == ["passage: alpha", "query: alpha question"]


def test_persistent_storage_survives_service_recreation(tmp_path):
    storage_path = tmp_path / "rag"
    first_service = RAGService(FakeEmbedder(), storage_path=storage_path)
    first_service.upload("doc-alpha", "alpha")

    second_service = RAGService(FakeEmbedder(), storage_path=storage_path)

    chunks, sources = second_service.search("alpha question", top_k=1)

    assert chunks == ["alpha"]
    assert sources == ["doc-alpha"]
    assert second_service.document_count == 1
    assert (storage_path / "index.faiss").is_file()
    assert (storage_path / "metadata.json").is_file()


def test_upload_creates_hnsw_inner_product_index_with_configured_construction():
    service = RAGService(FakeEmbedder(), hnsw_m=16, hnsw_ef_construction=80)

    service.upload("doc-alpha", "alpha")

    assert isinstance(service.index, faiss.IndexHNSWFlat)
    assert service.index.metric_type == faiss.METRIC_INNER_PRODUCT
    assert service.index.hnsw.efConstruction == 80


def test_search_fetches_extra_candidates_before_returning_top_k():
    service = RAGService(FakeEmbedder())
    service.upload("doc-alpha", "alpha")
    service.upload("doc-gamma", "gamma")
    service.upload("doc-delta", "delta")
    service.upload("doc-beta", "beta")

    chunks, sources = service.search("alpha question", top_k=2)

    assert chunks == ["alpha", "gamma"]
    assert sources == ["doc-alpha", "doc-gamma"]


def test_search_raises_for_non_positive_top_k():
    service = RAGService(FakeEmbedder())
    service.upload("doc-alpha", "alpha")

    try:
        service.search("alpha question", top_k=0)
    except ValueError as exc:
        assert str(exc) == "top_k must be positive"
    else:
        raise AssertionError("search should reject non-positive top_k")


def test_search_raises_ef_search_to_candidate_count():
    service = RAGService(FakeEmbedder(), hnsw_ef_search=2)
    service.upload("doc-alpha", "alpha")
    service.upload("doc-gamma", "gamma")
    service.upload("doc-delta", "delta")
    service.upload("doc-beta", "beta")

    service.search("alpha question", top_k=2)

    assert service.index.hnsw.efSearch == 4


def test_upload_ignores_text_without_chunks():
    service = RAGService(FakeEmbedder())

    doc_id, chunk_count = service.upload("blank", " \n\t ")

    assert doc_id == "blank"
    assert chunk_count == 0
    assert service.document_count == 0
    assert service.index is None


def test_search_rejects_empty_index():
    service = RAGService(FakeEmbedder())

    try:
        service.search("alpha question")
    except ValueError as exc:
        assert str(exc) == "No documents have been uploaded."
    else:
        raise AssertionError("search should reject an empty index")
