from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from llm_proxy_client import LLMProxyClient, LLMProxyError, LLMProxyNotConfiguredError
from rag import RAGService
from schemas import AskRequest, AskResponse, UploadRequest, UploadResponse


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "multilingual-e5-small"


def create_app(rag_service=None, llm_client=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.rag_service is None:
            app.state.rag_service = RAGService.from_local_model(MODEL_PATH)
        yield

    app = FastAPI(title="Vietnamese RAG API", lifespan=lifespan)
    app.state.rag_service = rag_service
    app.state.llm_client = llm_client or LLMProxyClient()

    @app.post("/upload", response_model=UploadResponse)
    def upload_document(request: UploadRequest):
        doc_id, chunk_count = app.state.rag_service.upload(request.doc_id, request.text)
        return UploadResponse(status="success", doc_id=doc_id, chunks=chunk_count)

    @app.post("/ask", response_model=AskResponse)
    def ask_question(request: AskRequest):
        try:
            contexts, sources = app.state.rag_service.search(request.question, top_k=3)
            answer = app.state.llm_client.generate_answer(request.question, contexts)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except LLMProxyNotConfiguredError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except LLMProxyError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return AskResponse(answer=answer, sources=sources)

    return app


app = create_app()
