import os
import sys

from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit(
        "ERROR: ANTHROPIC_API_KEY is not set.\n"
        "Copy .env.example to .env and fill in your Anthropic API key."
    )

from generate import generate_dataset

if __name__ == "__main__":
    generate_dataset("data/documents.jsonl")
