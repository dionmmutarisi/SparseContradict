DOCUMENT_LENGTH_MIN = 52
DOCUMENT_LENGTH_MAX = 65

DISTANCE_LEVELS = [5, 10, 20, 30, 40, 48]
DISTRACTOR_COUNT_LEVELS = [0, 2, 4, 6, 8]
DOCUMENTS_PER_CELL = 25

MODELS = [
    "gpt-4o-mini",
    "llama-3.1-8b",
    "mistral-7b",
    "phi-3.5-mini",
]

HF_MODEL_IDS = {
    "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "phi-3.5-mini": "microsoft/Phi-3.5-mini-instruct",
}

EXPERIMENTAL_GRID = [
    (distance, distractor_count)
    for distance in DISTANCE_LEVELS
    for distractor_count in DISTRACTOR_COUNT_LEVELS
]
