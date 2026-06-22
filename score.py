import re
import json
import math
from collections import defaultdict


def chance_accuracy(n_sentences: int) -> float:
    n = n_sentences
    return 2 / (n * (n - 1))


def score_prediction(
    prediction: tuple[int, int],
    ground_truth: tuple[int, int],
) -> bool:
    return set(prediction) == set(ground_truth)


def binomial_se(accuracy: float, n: int) -> float:
    return math.sqrt(accuracy * (1 - accuracy) / n)


def confidence_interval(
    accuracy: float, n: int, z: float = 1.96
) -> tuple[float, float]:
    se = binomial_se(accuracy, n)
    return (max(0.0, accuracy - z * se), min(1.0, accuracy + z * se))


def parse_response(text: str) -> tuple[int, int] | None:
    """
    Extract the LAST <answer>...</answer> block and parse its 'pair' field.
    Returns None if no valid block exists or parsing fails.
    """
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if not matches:
        return None
    try:
        data = json.loads(matches[-1].strip())
        pair = data["pair"]
        if len(pair) != 2:
            return None
        return (int(pair[0]), int(pair[1]))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, IndexError):
        return None


def aggregate(rows: list[dict]) -> list[dict]:
    """
    Group rows by (model, distance, distractor_count) and compute
    n_documents, accuracy, ci_low, ci_high, chance_accuracy.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["model"], row["distance"], row["distractor_count"])
        groups[key].append(row)

    results = []
    for (model, distance, distractor_count), grp in groups.items():
        n_docs = len(grp)
        n_correct = sum(1 for r in grp if r["correct"])
        acc = n_correct / n_docs if n_docs > 0 else 0.0
        mean_n = sum(r["n_sentences"] for r in grp) / n_docs
        ci_lo, ci_hi = confidence_interval(acc, n_docs)
        results.append(
            {
                "model": model,
                "distance": distance,
                "distractor_count": distractor_count,
                "n_documents": n_docs,
                "accuracy": acc,
                "ci_low": ci_lo,
                "ci_high": ci_hi,
                "chance_accuracy": chance_accuracy(mean_n),
            }
        )
    return results
