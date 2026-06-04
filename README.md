# Vietnamese RAG API

FastAPI service for uploading Vietnamese text and answering questions with:

- Local embeddings from `keepitreal/vietnamese-sbert`
- In-memory FAISS similarity search
- OpenAI-compatible Teacher Proxy answer generation over LAN

## Requirements

- Python `3.12`
- [`uv`](https://docs.astral.sh/uv/)

## Setup

```powershell
uv sync
uv run python scripts/download_embedding_model.py
uv run uvicorn app:app --app-dir src --reload
```

Copy `.env.example` to `.env` and fill in your student registration settings:

```dotenv
STUDENT_ID=B22DCCN501
TEACHER_PROXY_BASE_URL=http://192.168.50.218:8000/api/v1
STUDENT_SERVER_URL=http://192.168.1.15:8000
LLM_PROXY_MODEL=gpt-4o-mini
```

The embedding model is loaded exclusively from `models/vietnamese-sbert/`.
The server exits during startup with a download instruction if the model is
missing. FAISS vectors and document metadata are stored in RAM and reset when
the server restarts.

The `models/` directory is not committed to Git. After cloning the repository,
run `uv run python scripts/download_embedding_model.py` before starting the
server.

`LLM_PROXY_MODEL` optionally overrides the default `gpt-4o-mini` model. The
Student Server sends LLM requests to `{TEACHER_PROXY_BASE_URL}/proxy` with a
`120` second timeout and does not require Internet access.

## API

Upload text:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/upload `
  -ContentType "application/json" `
  -Body '{"doc_id":"intro","text":"FastAPI la mot web framework cho Python."}'
```

Ask a question:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/ask `
  -ContentType "application/json" `
  -Body '{"question":"FastAPI la gi?"}'
```

## Competition Registration

Start the Student Server first:

```powershell
uv run uvicorn app:app --app-dir src --reload
```

In another terminal, register the Student Server URL with the Teacher Proxy.
After registration succeeds, the script automatically starts the evaluation:

```powershell
uv run python scripts/register_competition.py
```

The registration client sends:

```http
POST {TEACHER_PROXY_BASE_URL}/competition/register
X-Student-ID: {STUDENT_ID}
Content-Type: application/json

{"server_url": "{STUDENT_SERVER_URL}"}
```

The registration request timeout is `100` seconds. Evaluation may take up to
`20` minutes because the Teacher Proxy uploads documents and asks multiple
questions before returning the score.

To clear the previous score and competition state before trying again:

```powershell
uv run python scripts/reset_competition.py
```

To view the current score and evaluation progress in another terminal:

```powershell
uv run python scripts/result_competition.py
```

The reset timeout is `30` seconds. A network error, an invalid Teacher Proxy
response, or a response that does not match the request causes a script to exit
with a non-zero status. Evaluation failures are reported without automatically
resetting the Teacher Proxy state.

The result request also has a `30` second timeout and can be used while an
evaluation is running or after it completes.

## Test

Tests use a fake embedder and fake LLM clients, so they do not download the
model or call external services.

```powershell
uv run pytest
```
