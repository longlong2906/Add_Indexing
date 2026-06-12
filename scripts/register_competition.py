import os
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from competition_client import CompetitionClient, CompetitionRegistrationError


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise CompetitionRegistrationError(f"{name} is not configured.")
    return value


def main() -> int:
    load_dotenv(Path.cwd() / ".env")
    try:
        client = CompetitionClient(required_env("TEACHER_PROXY_BASE_URL"))
        response = client.register(
            required_env("STUDENT_ID"),
            required_env("STUDENT_SERVER_URL"),
        )
    except CompetitionRegistrationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(response.message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
