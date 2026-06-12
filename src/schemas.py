from pydantic import BaseModel, Field, field_validator


class UploadRequest(BaseModel):
    doc_id: str | None = None
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class UploadResponse(BaseModel):
    status: str
    doc_id: str | None = None
    chunks: int


class AskRequest(BaseModel):
    question: str = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


class RegisterPayload(BaseModel):
    server_url: str


class RegisterResponse(BaseModel):
    message: str
    student_id: str
    server_url: str


class EvaluateRequest(BaseModel):
    document_received: bool | None = False


class EvaluateResponse(BaseModel):
    message: str
    final_score: float


class ResetResponse(BaseModel):
    status: str
    message: str
    score: float


class ResultResponse(BaseModel):
    student_id: str
    score: float
    status: str
    current_question: int
