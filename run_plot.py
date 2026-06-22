import glob
import sys

from dotenv import load_dotenv

load_dotenv()

result_files = sorted(glob.glob("results/*.jsonl"))

if not result_files:
    sys.exit(
        "ERROR: No result files found in results/.\n"
        "Run run_evaluate.py for at least one model first."
    )

print(f"Found {len(result_files)} result file(s): {result_files}")

from plot import make_all_figures

if __name__ == "__main__":
    make_all_figures(result_files)
