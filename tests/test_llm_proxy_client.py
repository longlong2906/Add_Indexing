from types import SimpleNamespace

import pytest
from openai import OpenAIError

from llm_proxy_client import LLMProxyClient, LLMProxyError, LLMProxyNotConfiguredError


class FakeCompletions:
    def __init__(self, content="A", error=None):
        self.content = content
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


class FakeOpenAI:
    def __init__(self, completions):
        self.chat = SimpleNamespace(completions=completions)


def test_client_requires_teacher_proxy_base_url(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TEACHER_PROXY_BASE_URL", raising=False)
    monkeypatch.setenv("STUDENT_ID", "B22DCCN501")

    with pytest.raises(LLMProxyNotConfiguredError, match="TEACHER_PROXY_BASE_URL"):
        LLMProxyClient()


def test_client_requires_uppercase_student_id(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEACHER_PROXY_BASE_URL", "http://teacher/api/v1")
    monkeypatch.setenv("STUDENT_ID", "b22dccn501")

    with pytest.raises(LLMProxyNotConfiguredError, match="uppercase"):
        LLMProxyClient()


def test_client_loads_configuration_from_dotenv(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TEACHER_PROXY_BASE_URL", raising=False)
    monkeypatch.delenv("STUDENT_ID", raising=False)
    monkeypatch.delenv("LLM_PROXY_MODEL", raising=False)
    (tmp_path / ".env").write_text(
        "TEACHER_PROXY_BASE_URL=http://teacher/api/v1\n"
        "STUDENT_ID=B22DCCN501\n"
        "LLM_PROXY_MODEL=custom-model\n",
        encoding="utf-8",
    )
    captured = {}

    def fake_factory(**kwargs):
        captured.update(kwargs)
        return FakeOpenAI(FakeCompletions())

    client = LLMProxyClient(client_factory=fake_factory)

    assert client.model == "custom-model"
    assert captured == {
        "base_url": "http://teacher/api/v1/proxy",
        "api_key": "B22DCCN501",
        "timeout": 120.0,
        "max_retries": 0,
    }


def test_generate_answer_calls_chat_completions_with_rag_prompt(monkeypatch):
    monkeypatch.delenv("LLM_PROXY_MODEL", raising=False)
    completions = FakeCompletions(content="B")
    client = LLMProxyClient(
        teacher_proxy_base_url="http://teacher/api/v1",
        student_id="B22DCCN501",
        client_factory=lambda **kwargs: FakeOpenAI(completions),
    )

    assert client.generate_answer("question", ["context"]) == "B"
    assert completions.calls[0]["model"] == "gpt-4o-mini"
    assert completions.calls[0]["temperature"] == 0.0
    assert completions.calls[0]["messages"][0]["role"] == "user"
    prompt = completions.calls[0]["messages"][0]["content"]
    assert "context" in prompt
    assert "question" in prompt
    assert "A, B, C hoặc D" in prompt
    assert "Không giải thích" in prompt


@pytest.mark.parametrize("content", [None, ""])
def test_generate_answer_rejects_empty_response(content):
    client = LLMProxyClient(
        teacher_proxy_base_url="http://teacher/api/v1",
        student_id="B22DCCN501",
        client_factory=lambda **kwargs: FakeOpenAI(FakeCompletions(content=content)),
    )

    with pytest.raises(LLMProxyError, match="empty response"):
        client.generate_answer("question", ["context"])


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("A", "A"),
        ("b", "B"),
        ("C.", "C"),
        ("Dap an: D", "D"),
        ("Answer: b", "B"),
    ],
)
def test_generate_answer_normalizes_single_multiple_choice_answer(content, expected):
    client = LLMProxyClient(
        teacher_proxy_base_url="http://teacher/api/v1",
        student_id="B22DCCN501",
        client_factory=lambda **kwargs: FakeOpenAI(FakeCompletions(content=content)),
    )

    assert client.generate_answer("question", ["context"]) == expected


@pytest.mark.parametrize(
    "content",
    [
        "Khong xac dinh",
        "A hoac B",
        "DATABASE",
    ],
)
def test_generate_answer_rejects_invalid_multiple_choice_answer(content):
    client = LLMProxyClient(
        teacher_proxy_base_url="http://teacher/api/v1",
        student_id="B22DCCN501",
        client_factory=lambda **kwargs: FakeOpenAI(FakeCompletions(content=content)),
    )

    with pytest.raises(LLMProxyError, match="valid multiple-choice answer"):
        client.generate_answer("question", ["context"])


def test_generate_answer_maps_openai_errors():
    client = LLMProxyClient(
        teacher_proxy_base_url="http://teacher/api/v1",
        student_id="B22DCCN501",
        client_factory=lambda **kwargs: FakeOpenAI(
            FakeCompletions(error=OpenAIError("upstream failed"))
        ),
    )

    with pytest.raises(LLMProxyError, match="LLM proxy request failed"):
        client.generate_answer("question", ["context"])
