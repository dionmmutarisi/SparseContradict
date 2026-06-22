def build_generation_prompt(n: int, i: int, j: int, k: int) -> str:
    """
    n : total sentence count (52–65)
    i : 1-based index of first sentence in the true contradiction pair
    j : 1-based index of second sentence  (j = i + distance)
    k : number of distractor pairs
    """
    distractor_instruction = (
        f"- Include exactly {k} distractor pair(s): pairs of sentences that are "
        "topically related to the true contradiction but are genuinely consistent "
        "under careful reading. They should look superficially suspicious but must "
        "NOT actually contradict each other."
        if k > 0
        else "- distractor_pairs must be an empty list []."
    )

    return f"""You are constructing a synthetic financial document for an NLP research benchmark.

TASK
Write a coherent financial report excerpt for a FICTIONAL company of EXACTLY {n} sentences.
Invent a plausible company name, sector, financial figures, and executive names. Every call to this prompt uses a DIFFERENT company.

PLANTED CONTRADICTION
- Sentence {i} and sentence {j} must logically contradict each other IN MEANING.
- Do NOT use direct negation or repeat the same vocabulary. Use different wording, different metrics, or different framings that are nonetheless logically incompatible.
- The contradiction may concern any financial or operational claim: revenue, costs, headcount, product launches, market share, guidance, debt, dividends, inventory, etc.
- Example pattern (do not copy literally): sentence {i} could state the company ended the year with zero debt, while sentence {j} states it took on a €200 million credit facility during the same period.

DISTRACTORS
{distractor_instruction}

CONSTRAINTS (all mandatory)
- No sentence index may appear in more than one pair (true pair or distractor).
- Every pair (true and distractor) must satisfy |a − b| >= 3.
- Every sentence NOT involved in any pair must be mutually consistent with all other such sentences.
- The document must contain EXACTLY ONE logical contradiction: the pair at positions {i} and {j}.
- The "sentences" array must have EXACTLY {n} elements (no more, no less).
- All sentence indices in the output are 1-based.

OUTPUT FORMAT
Return ONLY valid JSON with no preamble and no markdown fences. Use this exact schema:
{{
  "sentences": ["<sentence 1>", "<sentence 2>", ..., "<sentence {n}>"],
  "contradiction_pair": [{i}, {j}],
  "distractor_pairs": [<list of [a, b] pairs, or empty list if k=0>]
}}

Pairs are unordered. Do not include any text outside the JSON object."""


def build_inference_prompt(sentences: list[str]) -> str:
    """
    sentences : list of strings (the document to analyse)
    """
    numbered_doc = "\n".join(f"[{idx + 1}] {s}" for idx, s in enumerate(sentences))

    example_doc = """\
[1] The village of Mossford was founded by settlers in the early 1800s.
[2] Its main industry for over a century was wool production from local sheep farms.
[3] The town hall, built in 1887, still serves as the community's administrative centre.
[4] A severe drought in 1923 destroyed most of the wheat harvest that summer.
[5] The Mossford railway station opened in 1901, linking the village to the regional network.
[6] The 1923 growing season was the most productive on record for local farmers, with wheat yields exceeding all prior benchmarks.
[7] Population peaked at around 4,000 residents in the 1950s before a gradual decline.
[8] The old mill on River Street was converted into a heritage museum in 2003.
[9] Local elections are held every four years, with the next scheduled for 2026."""

    example_reasoning = """\
Step-by-step reasoning:
- Sentence [4] states that a severe drought in 1923 destroyed most of the wheat harvest.
- Sentence [6] states that the 1923 growing season was the most productive on record, with wheat yields exceeding all prior benchmarks.
- These two claims about the same year and the same crop are logically incompatible: a harvest cannot simultaneously be mostly destroyed by drought and be the most productive on record.
- No other pair of sentences presents a genuine logical contradiction.

<answer>{"pair": [4, 6]}</answer>"""

    return f"""You are a careful logical-reasoning assistant specialising in document analysis.

Your task: identify the single pair of sentences in a financial report that logically contradict each other.

--- WORKED EXAMPLE (non-financial domain) ---

Read the following numbered excerpt:

{example_doc}

{example_reasoning}

--- END EXAMPLE ---

Now analyse the actual financial report below.

{numbered_doc}

This excerpt contains EXACTLY ONE pair of sentences that logically contradict each other. The contradiction is a matter of meaning — the two sentences make claims that cannot both be true — and may not involve shared vocabulary or direct negation.

Think step by step:
1. Identify sentences that make specific, verifiable claims (figures, dates, events, states, ratios, rankings, etc.).
2. Check whether any two sentences assert incompatible facts about the same subject or time period.
3. Confirm that no other pair constitutes a genuine logical contradiction.

State your reasoning, then give your final answer inside <answer> tags as JSON with a single key "pair" whose value is a list of two 1-based sentence indices:

<answer>{{"pair": [i, j]}}</answer>"""
