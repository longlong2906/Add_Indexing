import pytest
import requests

from competition_client import CompetitionClient, CompetitionRegistrationError


class FakeResponse:
    def __init__(self, payload=None, status_error=None):
        self.payload = payload
        self.status_error = status_error

    def raise_for_status(self):
        if self.status_error:
            raise self.status_error

    def json(self):
        return self.payload


def test_register_sends_expected_request(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured.update(url=url, headers=headers, json=json, timeout=timeout)
        return FakeResponse(
            {
                "message": "Dang ky thanh cong!",
                "student_id": "B21DCCN629",
                "server_url": "http://192.168.1.15:5000",
            }
        )

    monkeypatch.setattr(requests, "post", fake_post)
    client = CompetitionClient("http://192.168.50.218:8000/api/v1/")

    response = client.register("B21DCCN629", "http://192.168.1.15:5000")

    assert response.message == "Dang ky thanh cong!"
    assert captured == {
        "url": "http://192.168.50.218:8000/api/v1/competition/register",
        "headers": {"X-Student-ID": "B21DCCN629"},
        "json": {"server_url": "http://192.168.1.15:5000"},
        "timeout": 100.0,
    }


@pytest.mark.parametrize(
    ("student_id", "server_url", "message"),
    [
        ("", "http://192.168.1.15:5000", "STUDENT_ID must not be blank."),
        ("b21dccn629", "http://192.168.1.15:5000", "STUDENT_ID must be uppercase."),
        ("B21DCCN629", "ftp://192.168.1.15:5000", "STUDENT_SERVER_URL must use http or https."),
    ],
)
def test_register_validates_student_configuration(student_id, server_url, message):
    client = CompetitionClient("http://192.168.50.218:8000/api/v1")

    with pytest.raises(CompetitionRegistrationError, match=message):
        client.register(student_id, server_url)


def test_client_validates_teacher_proxy_url():
    with pytest.raises(
        CompetitionRegistrationError,
        match="TEACHER_PROXY_BASE_URL must use http or https.",
    ):
        CompetitionClient("ftp://192.168.50.218:8000/api/v1")


@pytest.mark.parametrize(
    "failure",
    [
        requests.Timeout("timeout"),
        requests.HTTPError("bad request"),
    ],
)
def test_register_maps_request_errors(monkeypatch, failure):
    def fake_post(*args, **kwargs):
        if isinstance(failure, requests.HTTPError):
            return FakeResponse(status_error=failure)
        raise failure

    monkeypatch.setattr(requests, "post", fake_post)
    client = CompetitionClient("http://192.168.50.218:8000/api/v1")

    with pytest.raises(CompetitionRegistrationError, match="Competition registration failed."):
        client.register("B21DCCN629", "http://192.168.1.15:5000")


def test_register_rejects_invalid_response_schema(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: FakeResponse({"message": "ok"}))
    client = CompetitionClient("http://192.168.50.218:8000/api/v1")

    with pytest.raises(CompetitionRegistrationError, match="invalid response"):
        client.register("B21DCCN629", "http://192.168.1.15:5000")


def test_register_rejects_mismatched_response(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: FakeResponse(
            {
                "message": "ok",
                "student_id": "OTHER",
                "server_url": "http://192.168.1.15:5000",
            }
        ),
    )
    client = CompetitionClient("http://192.168.50.218:8000/api/v1")

    with pytest.raises(CompetitionRegistrationError, match="does not match"):
        client.register("B21DCCN629", "http://192.168.1.15:5000")


def test_evaluate_sends_expected_request(monkeypatch):
    captured = {}

    def fake_post(url, headers, timeout):
        captured.update(url=url, headers=headers, timeout=timeout)
        return FakeResponse(
            {
                "student_id": "B21DCCN629",
                "score": 8.0,
                "status": "completed",
                "detail": [{"question": 1, "score": 1.0}],
            }
        )

    monkeypatch.setattr(requests, "post", fake_post)
    client = CompetitionClient("http://192.168.50.218:8000/api/v1")

    response = client.evaluate("B21DCCN629")

    assert response.score == 8.0
    assert response.detail == [{"question": 1, "score": 1.0}]
    assert captured == {
        "url": "http://192.168.50.218:8000/api/v1/competition/evaluate",
        "headers": {"X-Student-ID": "B21DCCN629"},
        "timeout": 1200.0,
    }


def test_evaluate_rejects_mismatched_student_id(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: FakeResponse(
            {
                "student_id": "OTHER",
                "score": 8.0,
                "status": "completed",
                "detail": [],
            }
        ),
    )
    client = CompetitionClient("http://192.168.50.218:8000/api/v1")

    with pytest.raises(CompetitionRegistrationError, match="does not match"):
        client.evaluate("B21DCCN629")


def test_evaluate_rejects_invalid_response_schema(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: FakeResponse({"score": 8.0}))
    client = CompetitionClient("http://192.168.50.218:8000/api/v1")

    with pytest.raises(CompetitionRegistrationError, match="invalid response"):
        client.evaluate("B21DCCN629")


def test_reset_sends_expected_request(monkeypatch):
    captured = {}

    def fake_post(url, headers, timeout):
        captured.update(url=url, headers=headers, timeout=timeout)
        return FakeResponse({"status": "success", "message": "Da reset trang thai."})

    monkeypatch.setattr(requests, "post", fake_post)
    client = CompetitionClient("http://192.168.50.218:8000/api/v1")

    response = client.reset("B21DCCN629")

    assert response.message == "Da reset trang thai."
    assert captured == {
        "url": "http://192.168.50.218:8000/api/v1/competition/reset",
        "headers": {"X-Student-ID": "B21DCCN629"},
        "timeout": 30.0,
    }


@pytest.mark.parametrize("method", ["evaluate", "reset"])
@pytest.mark.parametrize("failure", [requests.Timeout(), requests.HTTPError()])
def test_competition_action_maps_request_errors(monkeypatch, method, failure):
    def fake_post(*args, **kwargs):
        if isinstance(failure, requests.HTTPError):
            return FakeResponse(status_error=failure)
        raise failure

    monkeypatch.setattr(requests, "post", fake_post)
    client = CompetitionClient("http://192.168.50.218:8000/api/v1")

    with pytest.raises(CompetitionRegistrationError, match="Competition request failed."):
        getattr(client, method)("B21DCCN629")


def test_reset_rejects_invalid_response_schema(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: FakeResponse({"status": "success"}))
    client = CompetitionClient("http://192.168.50.218:8000/api/v1")

    with pytest.raises(CompetitionRegistrationError, match="invalid response"):
        client.reset("B21DCCN629")


def test_result_sends_expected_request(monkeypatch):
    captured = {}

    def fake_get(url, headers, timeout):
        captured.update(url=url, headers=headers, timeout=timeout)
        return FakeResponse(
            {
                "student_id": "B21DCCN629",
                "score": 5.0,
                "status": "evaluating",
                "current_question": 6,
            }
        )

    monkeypatch.setattr(requests, "get", fake_get)
    client = CompetitionClient("http://192.168.50.218:8000/api/v1")

    response = client.result("B21DCCN629")

    assert response.current_question == 6
    assert captured == {
        "url": "http://192.168.50.218:8000/api/v1/competition/result",
        "headers": {"X-Student-ID": "B21DCCN629"},
        "timeout": 30.0,
    }


def test_result_rejects_mismatched_student_id(monkeypatch):
    monkeypatch.setattr(
        requests,
        "get",
        lambda *args, **kwargs: FakeResponse(
            {
                "student_id": "OTHER",
                "score": 5.0,
                "status": "evaluating",
                "current_question": 6,
            }
        ),
    )
    client = CompetitionClient("http://192.168.50.218:8000/api/v1")

    with pytest.raises(CompetitionRegistrationError, match="does not match"):
        client.result("B21DCCN629")


def test_result_rejects_invalid_response_schema(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: FakeResponse({"score": 5.0}))
    client = CompetitionClient("http://192.168.50.218:8000/api/v1")

    with pytest.raises(CompetitionRegistrationError, match="invalid response"):
        client.result("B21DCCN629")


@pytest.mark.parametrize("failure", [requests.Timeout(), requests.HTTPError()])
def test_result_maps_request_errors(monkeypatch, failure):
    def fake_get(*args, **kwargs):
        if isinstance(failure, requests.HTTPError):
            return FakeResponse(status_error=failure)
        raise failure

    monkeypatch.setattr(requests, "get", fake_get)
    client = CompetitionClient("http://192.168.50.218:8000/api/v1")

    with pytest.raises(CompetitionRegistrationError, match="Competition request failed."):
        client.result("B21DCCN629")
