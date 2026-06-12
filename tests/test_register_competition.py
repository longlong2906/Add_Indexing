import importlib.util
from pathlib import Path

from competition_client import CompetitionRegistrationError


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "register_competition.py"
REGISTRATION_ENV = ("STUDENT_ID", "TEACHER_PROXY_BASE_URL", "STUDENT_SERVER_URL")


def load_script():
    spec = importlib.util.spec_from_file_location("register_competition", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def clear_registration_env(monkeypatch):
    for name in REGISTRATION_ENV:
        monkeypatch.delenv(name, raising=False)


def test_main_loads_dotenv_and_prints_success(monkeypatch, tmp_path, capsys):
    clear_registration_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "STUDENT_ID=B21DCCN629\n"
        "TEACHER_PROXY_BASE_URL=http://teacher/api/v1\n"
        "STUDENT_SERVER_URL=http://student:5000\n",
        encoding="utf-8",
    )
    script = load_script()

    class FakeClient:
        def __init__(self, base_url):
            assert base_url == "http://teacher/api/v1"

        def register(self, student_id, server_url):
            assert student_id == "B21DCCN629"
            assert server_url == "http://student:5000"
            return type("Response", (), {"message": "Dang ky thanh cong!"})()

        def evaluate(self, student_id):
            raise AssertionError("evaluate should not be called")

    monkeypatch.setattr(script, "CompetitionClient", FakeClient)

    assert script.main() == 0
    assert capsys.readouterr().out.strip() == "Dang ky thanh cong!"


def test_main_returns_nonzero_when_configuration_is_missing(monkeypatch, tmp_path, capsys):
    clear_registration_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    script = load_script()

    assert script.main() == 1
    assert "TEACHER_PROXY_BASE_URL is not configured." in capsys.readouterr().err


def test_main_does_not_evaluate_when_registration_fails(monkeypatch, tmp_path, capsys):
    clear_registration_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "STUDENT_ID=B21DCCN629\n"
        "TEACHER_PROXY_BASE_URL=http://teacher/api/v1\n"
        "STUDENT_SERVER_URL=http://student:5000\n",
        encoding="utf-8",
    )
    script = load_script()

    class FakeClient:
        def __init__(self, base_url):
            pass

        def register(self, student_id, server_url):
            raise CompetitionRegistrationError("register failed")

        def evaluate(self, student_id):
            raise AssertionError("evaluate should not be called")

    monkeypatch.setattr(script, "CompetitionClient", FakeClient)

    assert script.main() == 1
    assert capsys.readouterr().err.strip() == "register failed"


def test_main_returns_success_without_running_evaluation(monkeypatch, tmp_path, capsys):
    clear_registration_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "STUDENT_ID=B21DCCN629\n"
        "TEACHER_PROXY_BASE_URL=http://teacher/api/v1\n"
        "STUDENT_SERVER_URL=http://student:5000\n",
        encoding="utf-8",
    )
    script = load_script()

    class FakeClient:
        def __init__(self, base_url):
            pass

        def register(self, student_id, server_url):
            return type("Response", (), {"message": "Dang ky thanh cong!"})()

        def evaluate(self, student_id):
            raise AssertionError("evaluate should not be called")

    monkeypatch.setattr(script, "CompetitionClient", FakeClient)

    assert script.main() == 0
    assert capsys.readouterr().out.strip() == "Dang ky thanh cong!"
