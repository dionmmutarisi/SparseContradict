import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()

VALID_MODELS = ["gpt-4o-mini", "llama-3.1-8b", "mistral-7b", "phi-3.5-mini"]
API_MODELS = {"gpt-4o-mini": "OPENAI_API_KEY"}

parser = argparse.ArgumentParser(description="Evaluate a model on SparseContradict.")
parser.add_argument(
    "--model",
    required=True,
    choices=VALID_MODELS,
    help="Model to evaluate.",
)
args = parser.parse_args()

if args.model in API_MODELS:
    key_name = API_MODELS[args.model]
    if not os.environ.get(key_name):
        sys.exit(
            f"ERROR: {key_name} is not set.\n"
            "Copy .env.example to .env and fill in your API key."
        )

from evaluate import evaluate_model

if __name__ == "__main__":
    evaluate_model(
        args.model,
        "data/documents.jsonl",
        f"results/{args.model}.jsonl",
    )
