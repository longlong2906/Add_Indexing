import importlib.util
import json
from pathlib import Path

from competition_client import CompetitionRegistrationError


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "result_competition.py"
REGISTRATION_ENV = ("STUDENT_ID", "TEACHER_PROXY_BASE_URL")


def load_script():
    spec = importlib.util.spec_from_file_location("result_competition", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def clear_registration_env(monkeypatch):
    for name in REGISTRATION_ENV:
        monkeypatch.delenv(name, raising=False)


def test_main_loads_dotenv_and_prints_result(monkeypatch, tmp_path, capsys):
    clear_registration_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "STUDENT_ID=B21DCCN629\n"
        "TEACHER_PROXY_BASE_URL=http://teacher/api/v1\n",
        encoding="utf-8",
    )
    script = load_script()

    class FakeClient:
        def __init__(self, base_url):
            assert base_url == "http://teacher/api/v1"

        def result(self, student_id):
            assert student_id == "B21DCCN629"
            return type(
                "Response",
                (),
                {
                    "model_dump": lambda self: {
                        "student_id": "B21DCCN629",
                        "score": 5.0,
                        "status": "evaluating",
                        "current_question": 6,
                    }
                },
            )()

    monkeypatch.setattr(script, "CompetitionClient", FakeClient)

    assert script.main() == 0
    assert json.loads(capsys.readouterr().out) == {
        "student_id": "B21DCCN629",
        "score": 5.0,
        "status": "evaluating",
        "current_question": 6,
    }


def test_main_returns_nonzero_when_configuration_is_missing(monkeypatch, tmp_path, capsys):
    clear_registration_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    script = load_script()

    assert script.main() == 1
    assert "TEACHER_PROXY_BASE_URL is not configured." in capsys.readouterr().err


def test_main_returns_nonzero_when_result_fails(monkeypatch, tmp_path, capsys):
    clear_registration_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "STUDENT_ID=B21DCCN629\n"
        "TEACHER_PROXY_BASE_URL=http://teacher/api/v1\n",
        encoding="utf-8",
    )
    script = load_script()

    class FakeClient:
        def __init__(self, base_url):
            pass

        def result(self, student_id):
            raise CompetitionRegistrationError("result failed")

    monkeypatch.setattr(script, "CompetitionClient", FakeClient)

    assert script.main() == 1
    assert capsys.readouterr().err.strip() == "result failed"
