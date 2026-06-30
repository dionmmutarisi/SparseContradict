# SparseContradict

SparseContradict is the benchmark implementation used in my bachelor thesis on contradiction localisation in long, financial-style documents. The project evaluates whether instruction-tuned language models can identify the exact two numbered sentences that form a planted contradiction when the evidence is sparse, separated in context, and surrounded by near-miss distractors.

The thesis does not propose a new model. It uses a controlled synthetic benchmark to test a narrower question: whether contradiction localisation performance is systematically affected by evidence distance and distractor density, and whether model failures are caused by retrieval of the wrong evidence pair, inability to compare the correct pair, or failure to produce a parseable structured answer.

## Thesis framing

Long-context access alone does not guarantee reliable contradiction localisation. A model may technically receive the full document but still fail to select the two relevant claims, compare them correctly, or return the answer in the required format. SparseContradict therefore treats contradiction detection as an end-to-end retrieval-and-selection problem rather than as isolated sentence-pair classification.

Each document contains numbered financial-style statements. Exactly two sentence indices form the gold contradiction pair. The model receives the document and a one-shot instruction prompt, then must return only the contradictory sentence IDs in a structured answer. A prediction is correct only when the extracted unordered pair exactly matches the gold pair.

## Research question

> How does exact-pair contradiction-localisation accuracy change as a function of sentence-level contradiction distance and distractor-pair density in synthetic financial-style documents?

The original hypothesis was that greater contradiction distance and higher distractor density would reduce accuracy. The thesis results are more nuanced: the strongest model remains comparatively stable, while weaker local models are limited mainly by wrong-pair selection and structured-output failures rather than by a clean distance-degradation curve.

## Benchmark design

| Component | Design choice |
|---|---|
| Task | Extractive contradiction localisation |
| Domain | Synthetic fictional financial-style documents |
| Document length | 52-65 numbered sentences |
| Gold answer | Exactly two sentence IDs forming one contradiction |
| Contradiction type | Same entity, metric, and reporting period; conflicting value |
| Distance levels | 5, 10, 20, 30, 40, 48 sentences |
| Distractor levels | 0, 2, 4, 6, 8 near-miss pairs |
| Documents per cell | 50 |
| Total documents | 1,500 |
| Prompting setup | One-shot prompting; no training or fine-tuning |
| Main metric | Exact-pair accuracy, order invariant |

A distractor pair is designed to look relevant but remain non-contradictory. These pairs share much of the surface structure of the true contradiction, forcing the model to distinguish the actual inconsistent evidence from plausible alternatives.

## Models evaluated

| Model | Evaluation setting | Purpose in the thesis |
|---|---|---|
| GPT-4o-mini | API model | Strong instruction-following reference model |
| Mistral 7B Instruct v0.3 | Local, 4-bit quantised | Local decoder-only baseline |
| Llama 3.1 8B Instruct | Local, 4-bit quantised | Local decoder-only baseline with different failure profile |
| Phi-3.5 Mini Instruct | Local, 4-bit quantised | Smaller local instruction-tuned baseline |

All models are evaluated on the same documents, with the same one-shot prompt structure and the same parser-based scoring procedure. The comparison is therefore an end-to-end benchmark comparison, not a perfectly controlled architectural ablation.

## Evaluation protocol

The model is shown a numbered document and asked to return the two contradictory sentence indices in a structured `<answer>` block. The parser extracts the predicted pair. Scoring is strict:

- `correct`: the parsed unordered pair exactly matches the gold contradiction pair;
- `wrong`: a parseable pair is returned, but it does not match the gold pair;
- `null`: no valid structured answer can be parsed.

This distinction is important because not all errors mean the same thing. A wrong but parseable answer suggests evidence-selection failure. A null answer suggests instruction-following or structured-output failure. Conditional accuracy can then be computed over non-null predictions to separate answer-format reliability from evidence-selection quality.

## Main thesis findings

The thesis reports that GPT-4o-mini performs best overall, while the three local 4-bit models remain substantially weaker under the same setup. The local models also fail differently: Mistral mostly returns valid but incorrect pairs, whereas Llama 3.1 8B and Phi-3.5 Mini more often fail by producing no parseable answer.

The expected clean degradation with contradiction distance is not strongly observed. This suggests that, in this benchmark, performance is shaped less by context length alone and more by the interaction between model capacity, attention selectivity, instruction following, and structured-output control.

## Repository structure

```text
SparseContradict/
├── .env                  # API keys; not committed
├── .gitignore
├── README.md
├── requirements.txt
├── config.py             # Locked benchmark parameters and experimental grid
├── prompts.py            # Generation and inference prompt builders
├── generate.py           # Dataset generation via Anthropic API
├── validate.py           # Dataset validation checks
├── evaluate.py           # One-shot model evaluation
├── score.py              # Exact-pair scoring and aggregation utilities
├── plot.py               # Thesis figures
├── run_generate.py       # Generate the dataset
├── run_evaluate.py       # Evaluate one model
├── run_plot.py           # Produce all figures
├── data/                 # Generated documents.jsonl; gitignored
├── results/              # Per-model result files; gitignored
├── figures/              # PDF and PNG figures; gitignored
└── tests/
    ├── test_validate.py
    └── test_score.py
```

## Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd SparseContradict

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
cp .env.example .env
```

Edit `.env` and provide the required keys:

```text
ANTHROPIC_API_KEY=<your-key>   # Required for dataset generation
OPENAI_API_KEY=<your-key>      # Required for GPT-4o-mini evaluation
```

Local HuggingFace models require suitable GPU memory and the dependencies listed in `requirements.txt`. The thesis experiments use 4-bit quantised local models, so absolute performance should be interpreted in that context.

## Running the benchmark

```bash
# Step 1: Generate the 1,500-document dataset
python run_generate.py

# Step 2: Evaluate each model
python run_evaluate.py --model gpt-4o-mini
python run_evaluate.py --model mistral-7b
python run_evaluate.py --model llama-3.1-8b
python run_evaluate.py --model phi-3.5-mini

# Step 3: Produce thesis figures
python run_plot.py
```

Evaluation is resumable. Re-running `run_evaluate.py` with the same `--model` flag skips documents that already have completed results.

## Output files

### `data/documents.jsonl`

One JSON object per document.

| Field | Type | Description |
|---|---|---|
| `doc_id` | string | Unique document identifier |
| `sentences` | list[string] | Numbered document sentences |
| `contradiction_pair` | [int, int] | Gold 1-based sentence IDs |
| `distractor_pairs` | list[[int, int]] | Non-contradictory near-miss pairs |
| `distance` | int | Sentence-level distance condition |
| `distractor_count` | int | Number of distractor pairs |
| `n_sentences` | int | Total sentence count |

### `results/{model}.jsonl`

One JSON object per evaluated document.

| Field | Type | Description |
|---|---|---|
| `model` | string | Model name |
| `doc_id` | string | Document identifier |
| `distance` | int | Distance condition |
| `distractor_count` | int | Distractor condition |
| `n_sentences` | int | Document length |
| `raw_response` | string | Unmodified model output |
| `prediction` | [int, int] or null | Parsed predicted pair |
| `correct` | bool | Exact-pair match with gold answer |

### `figures/`

| Figure | Thesis role |
|---|---|
| `figure1_accuracy_vs_distance` | Accuracy trend across distance levels |
| `figure2_heatmap_grid` | Accuracy across distance and distractor conditions |
| `figure3_degradation_slopes` | Estimated distance effect per model |

Figures are saved as both `.pdf` for thesis inclusion and `.png` for quick inspection.

## Tests

```bash
python3 -m pytest tests/ -v
```

The tests cover validation and scoring utilities. They are intended to protect the benchmark contract: exactly one gold contradiction pair, valid sentence indices, and strict order-invariant exact-pair scoring.

## Limitations

SparseContradict is deliberately controlled, which improves interpretability but limits ecological validity. The documents are synthetic and financial-style rather than natural filings. Validation checks enforce structural consistency, but they do not fully replace independent human semantic validation of contradiction quality. The one-shot prompt may make the task easier than a zero-shot setting because it anchors the expected answer format. Finally, model comparisons are not matched by parameter count, training data, architecture, or quantisation sensitivity, so the results should be interpreted as benchmark outcomes rather than isolated causal estimates of model design choices.


