from urllib.parse import urlparse

import requests
from pydantic import ValidationError

from schemas import EvaluateResponse, RegisterPayload, RegisterResponse, ResetResponse, ResultResponse


class CompetitionRegistrationError(RuntimeError):
    pass


class CompetitionClient:
    EVALUATE_TIMEOUT = 1200.0
    RESET_TIMEOUT = 30.0
    RESULT_TIMEOUT = 30.0

    def __init__(self, teacher_proxy_base_url: str, timeout: float = 100.0):
        self.teacher_proxy_base_url = self._validate_url(
            teacher_proxy_base_url,
            "TEACHER_PROXY_BASE_URL",
        ).rstrip("/")
        self.timeout = timeout

    def register(self, student_id: str, student_server_url: str) -> RegisterResponse:
        self._validate_student_id(student_id)
        student_server_url = self._validate_url(student_server_url, "STUDENT_SERVER_URL")
        payload = RegisterPayload(server_url=student_server_url)
        try:
            response = requests.post(
                f"{self.teacher_proxy_base_url}/competition/register",
                headers={"X-Student-ID": student_id},
                json=payload.model_dump(),
                timeout=self.timeout,
            )
            response.raise_for_status()
            registration = RegisterResponse.model_validate(response.json())
        except ValidationError as exc:
            raise CompetitionRegistrationError("Teacher Proxy returned an invalid response.") from exc
        except (requests.RequestException, TypeError, ValueError) as exc:
            raise CompetitionRegistrationError("Competition registration failed.") from exc
        if registration.student_id != student_id or registration.server_url != student_server_url:
            raise CompetitionRegistrationError("Teacher Proxy response does not match the request.")
        return registration

    def evaluate(self, student_id: str) -> EvaluateResponse:
        evaluation = self._post_action(
            "evaluate",
            student_id,
            EvaluateResponse,
            self.EVALUATE_TIMEOUT,
        )
        if evaluation.student_id != student_id:
            raise CompetitionRegistrationError("Teacher Proxy response does not match the request.")
        return evaluation

    def reset(self, student_id: str) -> ResetResponse:
        return self._post_action(
            "reset",
            student_id,
            ResetResponse,
            self.RESET_TIMEOUT,
        )

    def result(self, student_id: str) -> ResultResponse:
        self._validate_student_id(student_id)
        try:
            response = requests.get(
                f"{self.teacher_proxy_base_url}/competition/result",
                headers={"X-Student-ID": student_id},
                timeout=self.RESULT_TIMEOUT,
            )
            response.raise_for_status()
            result = ResultResponse.model_validate(response.json())
        except ValidationError as exc:
            raise CompetitionRegistrationError("Teacher Proxy returned an invalid response.") from exc
        except (requests.RequestException, TypeError, ValueError) as exc:
            raise CompetitionRegistrationError("Competition request failed.") from exc
        if result.student_id != student_id:
            raise CompetitionRegistrationError("Teacher Proxy response does not match the request.")
        return result

    def _post_action(self, action: str, student_id: str, response_model, timeout: float):
        self._validate_student_id(student_id)
        try:
            response = requests.post(
                f"{self.teacher_proxy_base_url}/competition/{action}",
                headers={"X-Student-ID": student_id},
                timeout=timeout,
            )
            response.raise_for_status()
            return response_model.model_validate(response.json())
        except ValidationError as exc:
            raise CompetitionRegistrationError("Teacher Proxy returned an invalid response.") from exc
        except (requests.RequestException, TypeError, ValueError) as exc:
            raise CompetitionRegistrationError("Competition request failed.") from exc

    @staticmethod
    def _validate_student_id(student_id: str) -> None:
        if not student_id:
            raise CompetitionRegistrationError("STUDENT_ID must not be blank.")
        if student_id != student_id.upper():
            raise CompetitionRegistrationError("STUDENT_ID must be uppercase.")

    @staticmethod
    def _validate_url(value: str, name: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise CompetitionRegistrationError(f"{name} must use http or https.")
        return value.rstrip("/")
