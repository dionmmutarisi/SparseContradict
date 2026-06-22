import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import unittest
from score import (
    chance_accuracy,
    score_prediction,
    binomial_se,
    confidence_interval,
    parse_response,
    aggregate,
)


class TestChanceAccuracy(unittest.TestCase):
    def test_n10(self):
        result = chance_accuracy(10)
        self.assertAlmostEqual(result, 2 / (10 * 9))

    def test_n55(self):
        result = chance_accuracy(55)
        self.assertAlmostEqual(result, 2 / (55 * 54))

    def test_n2(self):
        # Degenerate edge case: only one possible pair
        self.assertAlmostEqual(chance_accuracy(2), 1.0)

    def test_returns_float(self):
        self.assertIsInstance(chance_accuracy(60), float)


class TestScorePrediction(unittest.TestCase):
    def test_exact_match(self):
        self.assertTrue(score_prediction((3, 7), (3, 7)))

    def test_reversed_match(self):
        self.assertTrue(score_prediction((7, 3), (3, 7)))

    def test_wrong_prediction(self):
        self.assertFalse(score_prediction((1, 2), (1, 3)))

    def test_completely_wrong(self):
        self.assertFalse(score_prediction((10, 20), (5, 15)))

    def test_one_index_shared(self):
        self.assertFalse(score_prediction((5, 10), (5, 15)))


class TestBinomialSE(unittest.TestCase):
    def test_fifty_percent(self):
        # SE = sqrt(0.5 * 0.5 / 100) = 0.05
        self.assertAlmostEqual(binomial_se(0.5, 100), 0.05)

    def test_perfect_accuracy(self):
        # SE = 0 when accuracy=1
        self.assertAlmostEqual(binomial_se(1.0, 50), 0.0)

    def test_zero_accuracy(self):
        self.assertAlmostEqual(binomial_se(0.0, 50), 0.0)

    def test_typical_case(self):
        expected = math.sqrt(0.3 * 0.7 / 200)
        self.assertAlmostEqual(binomial_se(0.3, 200), expected)


class TestConfidenceInterval(unittest.TestCase):
    def test_symmetric_at_half(self):
        lo, hi = confidence_interval(0.5, 100)
        self.assertAlmostEqual(lo, 0.5 - 1.96 * 0.05, places=6)
        self.assertAlmostEqual(hi, 0.5 + 1.96 * 0.05, places=6)

    def test_clamp_lower_bound(self):
        lo, hi = confidence_interval(0.01, 10)
        self.assertGreaterEqual(lo, 0.0)

    def test_clamp_upper_bound(self):
        lo, hi = confidence_interval(0.99, 10)
        self.assertLessEqual(hi, 1.0)

    def test_custom_z(self):
        lo, hi = confidence_interval(0.4, 100, z=2.576)
        se = binomial_se(0.4, 100)
        self.assertAlmostEqual(lo, max(0.0, 0.4 - 2.576 * se), places=6)

    def test_lo_less_than_hi(self):
        lo, hi = confidence_interval(0.6, 50)
        self.assertLess(lo, hi)


class TestParseResponse(unittest.TestCase):
    def test_basic_extraction(self):
        text = 'Some reasoning.\n<answer>{"pair": [3, 7]}</answer>'
        self.assertEqual(parse_response(text), (3, 7))

    def test_last_answer_used(self):
        # Prompt contains a worked example answer; model's answer comes last
        text = (
            'Example: <answer>{"pair": [4, 6]}</answer> '
            'My answer: <answer>{"pair": [12, 22]}</answer>'
        )
        self.assertEqual(parse_response(text), (12, 22))

    def test_no_answer_tag_returns_none(self):
        self.assertIsNone(parse_response("The pair is probably [3, 7]."))

    def test_malformed_json_returns_none(self):
        self.assertIsNone(parse_response("<answer>not json</answer>"))

    def test_missing_pair_key_returns_none(self):
        self.assertIsNone(parse_response('<answer>{"indices": [3, 7]}</answer>'))

    def test_reversed_pair_still_parsed(self):
        text = '<answer>{"pair": [22, 12]}</answer>'
        result = parse_response(text)
        self.assertEqual(set(result), {12, 22})

    def test_whitespace_around_json(self):
        text = '<answer>  {"pair": [5, 15]}  </answer>'
        self.assertEqual(parse_response(text), (5, 15))

    def test_multiline_answer_block(self):
        text = '<answer>\n{"pair": [8, 18]}\n</answer>'
        self.assertEqual(parse_response(text), (8, 18))

    def test_empty_answer_tag_returns_none(self):
        self.assertIsNone(parse_response("<answer></answer>"))

    def test_does_not_fall_back_to_integers_in_text(self):
        text = "The contradiction is between sentences 5 and 15."
        self.assertIsNone(parse_response(text))


class TestAggregate(unittest.TestCase):
    def _make_rows(self, model, distance, k, n_total, n_correct, n_sents=58):
        rows = []
        for i in range(n_total):
            rows.append(
                {
                    "model": model,
                    "doc_id": f"d{distance}_k{k}_{i:03d}",
                    "distance": distance,
                    "distractor_count": k,
                    "n_sentences": n_sents,
                    "prediction": [1, 2],
                    "correct": i < n_correct,
                }
            )
        return rows

    def test_single_group_accuracy(self):
        rows = self._make_rows("gpt-4o-mini", 5, 0, 10, 6)
        result = aggregate(rows)
        self.assertEqual(len(result), 1)
        r = result[0]
        self.assertAlmostEqual(r["accuracy"], 0.6)
        self.assertEqual(r["n_documents"], 10)

    def test_two_groups_separated(self):
        rows = (
            self._make_rows("gpt-4o-mini", 5, 0, 10, 3)
            + self._make_rows("gpt-4o-mini", 10, 0, 10, 7)
        )
        result = aggregate(rows)
        self.assertEqual(len(result), 2)
        by_dist = {r["distance"]: r for r in result}
        self.assertAlmostEqual(by_dist[5]["accuracy"], 0.3)
        self.assertAlmostEqual(by_dist[10]["accuracy"], 0.7)

    def test_ci_bounds_within_zero_one(self):
        rows = self._make_rows("gpt-4o-mini", 5, 0, 50, 0)
        r = aggregate(rows)[0]
        self.assertGreaterEqual(r["ci_low"], 0.0)
        self.assertLessEqual(r["ci_high"], 1.0)

    def test_chance_accuracy_computed(self):
        rows = self._make_rows("gpt-4o-mini", 5, 0, 10, 5, n_sents=55)
        r = aggregate(rows)[0]
        expected = chance_accuracy(55)
        self.assertAlmostEqual(r["chance_accuracy"], expected)

    def test_multiple_models(self):
        rows = (
            self._make_rows("gpt-4o-mini", 5, 0, 50, 20)
            + self._make_rows("llama-3.1-8b", 5, 0, 50, 10)
        )
        result = aggregate(rows)
        self.assertEqual(len(result), 2)
        models = {r["model"] for r in result}
        self.assertIn("gpt-4o-mini", models)
        self.assertIn("llama-3.1-8b", models)

    def test_required_keys_present(self):
        rows = self._make_rows("gpt-4o-mini", 5, 2, 50, 25)
        r = aggregate(rows)[0]
        for key in ("model", "distance", "distractor_count", "n_documents",
                    "accuracy", "ci_low", "ci_high", "chance_accuracy"):
            self.assertIn(key, r)


if __name__ == "__main__":
    unittest.main(verbosity=2)
