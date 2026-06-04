import os
import re
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError


class LLMProxyError(RuntimeError):
    pass


class LLMProxyNotConfiguredError(LLMProxyError):
    pass


class LLMProxyClient:
    def __init__(
        self,
        teacher_proxy_base_url: str | None = None,
        student_id: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
        client_factory=OpenAI,
    ):
        load_dotenv(Path.cwd() / ".env")
        teacher_proxy_base_url = teacher_proxy_base_url or os.getenv("TEACHER_PROXY_BASE_URL")
        student_id = student_id or os.getenv("STUDENT_ID")
        self.teacher_proxy_base_url = self._validate_url(teacher_proxy_base_url)
        self.student_id = self._validate_student_id(student_id)
        self.model = model or os.getenv("LLM_PROXY_MODEL", "gpt-4o-mini")
        self.client = client_factory(
            base_url=f"{self.teacher_proxy_base_url}/proxy",
            api_key=self.student_id,
            timeout=timeout,
            max_retries=0,
        )

    def generate_answer(self, question: str, contexts: list[str]) -> str:
        prompt = (
            "Bạn đang trả lời một câu hỏi trắc nghiệm.\n\n"
            "Yêu cầu:\n"
            "1. Đọc kỹ ngữ cảnh và câu hỏi.\n"
            "2. Tự suy luận để chọn một đáp án đúng nhất.\n"
            "3. Nếu ngữ cảnh chưa đủ, hãy chọn phương án hợp lý nhất dựa trên kiến thức của bạn.\n"
            "4. Chỉ trả về DUY NHẤT một ký tự viết hoa: A, B, C hoặc D.\n"
            "5. Không giải thích. Không thêm dấu câu. Không thêm bất kỳ nội dung nào khác.\n\n"
            f"Ngữ cảnh:\n{'\n\n'.join(contexts)}\n\n"
            f"Câu hỏi:\n{question}\n\n"
            "Đáp án:"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            answer = response.choices[0].message.content
        except (OpenAIError, AttributeError, IndexError, TypeError) as exc:
            raise LLMProxyError("LLM proxy request failed.") from exc
        if not answer:
            raise LLMProxyError("LLM proxy returned an empty response.")
        return self._normalize_multiple_choice_answer(answer)

    @staticmethod
    def _normalize_multiple_choice_answer(content: str) -> str:
        answers = set(re.findall(r"(?<![A-Z0-9_])[A-D](?![A-Z0-9_])", content.upper()))
        if len(answers) != 1:
            raise LLMProxyError("LLM proxy did not return a valid multiple-choice answer.")
        return answers.pop()

    @staticmethod
    def _validate_url(value: str | None) -> str:
        if not value:
            raise LLMProxyNotConfiguredError("TEACHER_PROXY_BASE_URL is not configured.")
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise LLMProxyNotConfiguredError("TEACHER_PROXY_BASE_URL must use http or https.")
        return value.rstrip("/")

    @staticmethod
    def _validate_student_id(value: str | None) -> str:
        if not value:
            raise LLMProxyNotConfiguredError("STUDENT_ID is not configured.")
        if value != value.upper():
            raise LLMProxyNotConfiguredError("STUDENT_ID must be uppercase.")
        return value
