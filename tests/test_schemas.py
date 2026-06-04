import pytest
from pydantic import ValidationError

from schemas import AskRequest, UploadRequest


@pytest.mark.parametrize(
    ("schema", "field"),
    [
        (UploadRequest, "text"),
        (AskRequest, "question"),
    ],
)
def test_request_rejects_blank_text(schema, field):
    with pytest.raises(ValidationError):
        schema(**{field: "   "})
