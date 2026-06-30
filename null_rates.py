"""
Compute per-model null prediction rates and accuracy breakdown.
Usage: python null_rates.py path/to/results/
       python null_rates.py results/gpt-4o-mini.jsonl results/llama-3.1-8b.jsonl ...
"""

import json
import sys
import os
from collections import defaultdict, Counter


def load_results(paths):
    results = []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    return results


def resolve_paths(args):
    paths = []
    for arg in args:
        if os.path.isdir(arg):
            for fname in sorted(os.listdir(arg)):
                if fname.endswith(".jsonl"):
                    paths.append(os.path.join(arg, fname))
        elif os.path.isfile(arg):
            paths.append(arg)
    return paths


def analyse(results):
    total   = defaultdict(int)
    nulls   = defaultdict(int)
    correct = defaultdict(int)
    wrong   = defaultdict(int)

    for row in results:
        m = row["model"]
        total[m] += 1
        if row["prediction"] is None:
            nulls[m] += 1
        elif row["correct"]:
            correct[m] += 1
        else:
            wrong[m] += 1

    models = sorted(total.keys())

    # ── Table 1: overall breakdown ────────────────────────────────────
    print(f"\n{'Model':<30} {'N':>5} {'Correct':>8} {'Wrong':>8} "
          f"{'Null':>7} {'Acc%':>7} {'Null%':>7} {'Wrong%':>8}")
    print("─" * 82)
    for m in models:
        n = total[m]
        print(f"{m:<30} {n:>5} {correct[m]:>8} {wrong[m]:>8} "
              f"{nulls[m]:>7} {correct[m]/n*100:>6.1f}% "
              f"{nulls[m]/n*100:>6.1f}% {wrong[m]/n*100:>7.1f}%")

    # ── Table 2: null rate by distance ────────────────────────────────
    null_d  = defaultdict(Counter)
    total_d = defaultdict(Counter)
    null_k  = defaultdict(Counter)
    total_k = defaultdict(Counter)

    for row in results:
        m = row["model"]
        d = row["distance"]
        k = row["distractor_count"]
        total_d[m][d] += 1
        total_k[m][k] += 1
        if row["prediction"] is None:
            null_d[m][d] += 1
            null_k[m][k] += 1

    print(f"\n{'Null rate by model × distance'}")
    print("─" * 70)
    for m in models:
        ds = sorted(total_d[m])
        row_str = "  ".join(
            f"d={d}:{null_d[m][d]/total_d[m][d]*100:4.0f}%" for d in ds
        )
        print(f"{m:<30}  {row_str}")

    # ── Table 3: null rate by distractor count ────────────────────────
    print(f"\n{'Null rate by model × distractors'}")
    print("─" * 70)
    for m in models:
        ks = sorted(total_k[m])
        row_str = "  ".join(
            f"k={k}:{null_k[m][k]/total_k[m][k]*100:4.0f}%" for k in ks
        )
        print(f"{m:<30}  {row_str}")

    print()


if __name__ == "__main__":
    args = sys.argv[1:] if len(sys.argv) > 1 else ["results"]
    paths = resolve_paths(args)
    if not paths:
        print("No .jsonl files found. Pass a directory or file paths.")
        sys.exit(1)
    print(f"Loaded files: {[os.path.basename(p) for p in paths]}")
    results = load_results(paths)
    print(f"Total rows: {len(results)}")
    analyse(results)
