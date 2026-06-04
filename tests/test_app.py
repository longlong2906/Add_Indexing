from fastapi.testclient import TestClient

from app import create_app
from llm_proxy_client import LLMProxyError, LLMProxyNotConfiguredError


class FakeRAG:
    def __init__(self):
        self.uploads = []

    def upload(self, doc_id, text):
        self.uploads.append((doc_id, text))
        return doc_id or "generated-id", 2

    def search(self, question, top_k=3):
        if question == "empty":
            raise ValueError("No documents have been uploaded.")
        return ["relevant context"], ["doc-1"]


class FakeLLM:
    def generate_answer(self, question, contexts):
        return "B"


def test_upload_returns_doc_id_and_chunk_count():
    rag = FakeRAG()
    client = TestClient(create_app(rag_service=rag, llm_client=FakeLLM()))

    response = client.post("/upload", json={"text": "document"})

    assert response.status_code == 200
    assert response.json() == {"status": "success", "doc_id": "generated-id", "chunks": 2}
    assert rag.uploads == [(None, "document")]


def test_ask_returns_answer_and_sources():
    client = TestClient(create_app(rag_service=FakeRAG(), llm_client=FakeLLM()))

    response = client.post("/ask", json={"question": "question"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "B",
        "sources": ["doc-1"],
    }


def test_ask_maps_empty_index_to_409():
    client = TestClient(create_app(rag_service=FakeRAG(), llm_client=FakeLLM()))

    response = client.post("/ask", json={"question": "empty"})

    assert response.status_code == 409


def test_ask_maps_missing_proxy_configuration_to_503():
    class MissingProxyConfiguration:
        def generate_answer(self, question, contexts):
            raise LLMProxyNotConfiguredError("missing")

    client = TestClient(create_app(rag_service=FakeRAG(), llm_client=MissingProxyConfiguration()))

    response = client.post("/ask", json={"question": "question"})

    assert response.status_code == 503


def test_ask_maps_llm_proxy_error_to_502():
    class BrokenProxy:
        def generate_answer(self, question, contexts):
            raise LLMProxyError("upstream failed")

    client = TestClient(create_app(rag_service=FakeRAG(), llm_client=BrokenProxy()))

    response = client.post("/ask", json={"question": "question"})

    assert response.status_code == 502


def test_upload_rejects_blank_text():
    client = TestClient(create_app(rag_service=FakeRAG(), llm_client=FakeLLM()))

    response = client.post("/upload", json={"text": "   "})

    assert response.status_code == 422
