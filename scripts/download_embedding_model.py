from pathlib import Path

from sentence_transformers import SentenceTransformer


MODEL_NAME = "keepitreal/vietnamese-sbert"
TARGET_PATH = Path(__file__).resolve().parents[1] / "models" / "vietnamese-sbert"


def main() -> None:
    TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    model.save_pretrained(str(TARGET_PATH))
    print(f"Saved embedding model to {TARGET_PATH}")


if __name__ == "__main__":
    main()
