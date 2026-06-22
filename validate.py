from dataclasses import dataclass


@dataclass
class DocConfig:
    n: int
    distance: int
    distractor_count: int


def validate_document(data: dict, config: "DocConfig") -> tuple[bool, str]:
    """
    Returns (True, "") if data passes all checks, otherwise (False, reason).
    Checks are run in order; first failure short-circuits.
    """
    # 1. data is a dict with required keys
    if not isinstance(data, dict):
        return False, "data is not a dict"
    for key in ("sentences", "contradiction_pair", "distractor_pairs"):
        if key not in data:
            return False, f"missing key '{key}'"

    # 2. sentences is a list of strings with length n, and n in [52, 65]
    sentences = data["sentences"]
    if not isinstance(sentences, list):
        return False, "sentences is not a list"
    if not (52 <= config.n <= 65):
        return False, f"config.n={config.n} is outside allowed range [52, 65]"
    if len(sentences) != config.n:
        return False, f"expected {config.n} sentences, got {len(sentences)}"
    if not all(isinstance(s, str) for s in sentences):
        return False, "not all sentences are strings"

    n = config.n

    # 3. contradiction_pair is a list of exactly 2 distinct integers
    cp = data["contradiction_pair"]
    if not isinstance(cp, list) or len(cp) != 2:
        return False, "contradiction_pair must be a list of exactly 2 elements"
    if not all(isinstance(x, int) for x in cp):
        return False, "contradiction_pair indices must be integers"
    if cp[0] == cp[1]:
        return False, "contradiction_pair indices must be distinct"

    # 4. both indices in [1, n]
    if not (1 <= cp[0] <= n and 1 <= cp[1] <= n):
        return False, f"contradiction_pair indices must be in [1, {n}]"

    # 5. distance is exact
    if abs(cp[1] - cp[0]) != config.distance:
        return (
            False,
            f"contradiction pair distance is {abs(cp[1] - cp[0])}, "
            f"expected {config.distance}",
        )

    # 6. distractor_pairs has the right count and each pair is valid
    dps = data["distractor_pairs"]
    if not isinstance(dps, list):
        return False, "distractor_pairs is not a list"
    if len(dps) != config.distractor_count:
        return (
            False,
            f"expected {config.distractor_count} distractor pairs, got {len(dps)}",
        )
    for idx, dp in enumerate(dps):
        if not isinstance(dp, list) or len(dp) != 2:
            return False, f"distractor pair {idx} must be a list of 2 elements"
        if not all(isinstance(x, int) for x in dp):
            return False, f"distractor pair {idx} indices must be integers"
        if dp[0] == dp[1]:
            return False, f"distractor pair {idx} indices must be distinct"
        if not (1 <= dp[0] <= n and 1 <= dp[1] <= n):
            return False, f"distractor pair {idx} indices must be in [1, {n}]"

    # 7. all pairs satisfy |a - b| >= 3
    all_pairs = [cp] + dps
    for pair in all_pairs:
        if abs(pair[1] - pair[0]) < 3:
            return (
                False,
                f"pair {pair} has separation {abs(pair[1] - pair[0])}, must be >= 3",
            )

    # 8. no sentence index appears in more than one pair
    all_indices = [idx for pair in all_pairs for idx in pair]
    if len(all_indices) != len(set(all_indices)):
        return False, "duplicate sentence indices across pairs"

    return True, ""
