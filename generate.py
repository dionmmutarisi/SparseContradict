import json
import os
import random
import re

import anthropic

from config import (
    DISTANCE_LEVELS,
    DISTRACTOR_COUNT_LEVELS,
    DOCUMENT_LENGTH_MAX,
    DOCUMENT_LENGTH_MIN,
    DOCUMENTS_PER_CELL,
    EXPERIMENTAL_GRID,
)
from prompts import build_generation_prompt
from validate import DocConfig, validate_document

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.match(text.strip())
    return m.group(1).strip() if m else text.strip()


def _count_existing(output_path: str) -> dict[tuple[int, int], int]:
    """Count already-accepted documents per (distance, distractor_count) cell."""
    counts: dict[tuple[int, int], int] = {}
    if not os.path.exists(output_path):
        return counts
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = (rec["distance"], rec["distractor_count"])
                counts[key] = counts.get(key, 0) + 1
            except (json.JSONDecodeError, KeyError):
                continue
    return counts


def generate_dataset(
    output_path: str,
    *,
    max_cells: int | None = None,
    max_per_cell: int | None = None,
) -> None:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    docs_per_cell = max_per_cell if max_per_cell is not None else DOCUMENTS_PER_CELL
    grid = EXPERIMENTAL_GRID[:max_cells] if max_cells is not None else EXPERIMENTAL_GRID

    existing = _count_existing(output_path)

    with open(output_path, "a", encoding="utf-8") as out_f:
        for cell_idx, (distance, distractor_count) in enumerate(grid):
            already_done = existing.get((distance, distractor_count), 0)

            if already_done >= docs_per_cell:
                print(
                    f"\n[cell {cell_idx + 1}/{len(grid)}] "
                    f"distance={distance}  distractors={distractor_count}  "
                    f"SKIPPING ({already_done}/{docs_per_cell} already done)"
                )
                continue

            print(
                f"\n[cell {cell_idx + 1}/{len(grid)}] "
                f"distance={distance}  distractors={distractor_count}  "
                f"(resuming from {already_done}/{docs_per_cell})"
            )

            accepted = already_done
            consecutive_skips = 0

            while accepted < docs_per_cell:
                n = random.randint(DOCUMENT_LENGTH_MIN, DOCUMENT_LENGTH_MAX)
                max_i = n - distance
                if max_i < 1:
                    continue
                i = random.randint(1, max_i)
                j = i + distance
                config = DocConfig(n=n, distance=distance, distractor_count=distractor_count)

                doc_data = None
                last_error = ""

                for attempt in range(3):
                    prompt = build_generation_prompt(n, i, j, distractor_count)
                    if attempt > 0:
                        prompt += (
                            f"\n\nYour previous response failed validation: "
                            f"{last_error}. Please try again."
                        )

                    print(
                        f"  doc {accepted + 1}/{docs_per_cell}  "
                        f"(n={n}, i={i}, j={j}, attempt={attempt + 1}) ... ",
                        end="",
                        flush=True,
                    )

                    try:
                        response = client.messages.create(
                            model="claude-sonnet-4-6",
                            max_tokens=8192,
                            messages=[{"role": "user", "content": prompt}],
                        )
                        raw = response.content[0].text
                        text = _strip_fences(raw)

                        try:
                            data = json.loads(text)
                        except json.JSONDecodeError as exc:
                            last_error = f"JSON parse error: {exc}"
                            print("RETRY (json error)")
                            continue

                        valid, reason = validate_document(data, config)
                        if valid:
                            doc_data = data
                            print("OK")
                            break
                        else:
                            last_error = reason
                            print(f"RETRY ({reason})")

                    except Exception as exc:
                        last_error = str(exc)
                        print(f"RETRY (api error: {exc})")

                if doc_data is None:
                    print(
                        f"  WARNING: skipping doc {accepted + 1} after 3 failed "
                        f"attempts. Last error: {last_error}"
                    )
                    consecutive_skips += 1
                    if consecutive_skips >= 10:
                        print(
                            f"  ERROR: 10 consecutive failures for cell "
                            f"(distance={distance}, k={distractor_count}); "
                            "aborting this cell."
                        )
                        break
                    continue

                doc_id = f"d{distance}_k{distractor_count}_{accepted:03d}"
                record = {
                    "doc_id": doc_id,
                    "sentences": doc_data["sentences"],
                    "contradiction_pair": doc_data["contradiction_pair"],
                    "distractor_pairs": doc_data["distractor_pairs"],
                    "distance": distance,
                    "distractor_count": distractor_count,
                    "n_sentences": n,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                accepted += 1
                consecutive_skips = 0