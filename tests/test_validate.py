import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
from validate import validate_document, DocConfig


def make_good_doc(n=55, distance=10, k=2):
    """Build a minimal valid document dict for config (n, distance, k)."""
    sentences = [f"Sentence {x}." for x in range(1, n + 1)]
    i, j = 5, 5 + distance
    # distractor pairs: [20, 25], [30, 35] — well-separated and unique
    dp1 = [20, 26]
    dp2 = [30, 36]
    distractors = []
    if k >= 1:
        distractors.append(dp1)
    if k >= 2:
        distractors.append(dp2)
    return {
        "sentences": sentences,
        "contradiction_pair": [i, j],
        "distractor_pairs": distractors,
    }


class TestCheck1_KeysAndType(unittest.TestCase):
    def test_not_a_dict(self):
        ok, msg = validate_document("not a dict", DocConfig(55, 10, 0))
        self.assertFalse(ok)
        self.assertIn("not a dict", msg)

    def test_missing_sentences_key(self):
        doc = {"contradiction_pair": [1, 11], "distractor_pairs": []}
        ok, msg = validate_document(doc, DocConfig(55, 10, 0))
        self.assertFalse(ok)
        self.assertIn("sentences", msg)

    def test_missing_contradiction_pair_key(self):
        doc = {"sentences": ["S."] * 55, "distractor_pairs": []}
        ok, msg = validate_document(doc, DocConfig(55, 10, 0))
        self.assertFalse(ok)
        self.assertIn("contradiction_pair", msg)

    def test_missing_distractor_pairs_key(self):
        doc = {"sentences": ["S."] * 55, "contradiction_pair": [1, 11]}
        ok, msg = validate_document(doc, DocConfig(55, 10, 0))
        self.assertFalse(ok)
        self.assertIn("distractor_pairs", msg)


class TestCheck2_Sentences(unittest.TestCase):
    def test_sentences_not_list(self):
        doc = make_good_doc()
        doc["sentences"] = "not a list"
        ok, msg = validate_document(doc, DocConfig(55, 10, 2))
        self.assertFalse(ok)
        self.assertIn("not a list", msg)

    def test_n_out_of_range_low(self):
        doc = make_good_doc(n=55, distance=10, k=0)
        # config.n=51 is below 52
        ok, msg = validate_document(doc, DocConfig(51, 10, 0))
        self.assertFalse(ok)
        self.assertIn("outside allowed range", msg)

    def test_n_out_of_range_high(self):
        doc = make_good_doc(n=55, distance=10, k=0)
        ok, msg = validate_document(doc, DocConfig(66, 10, 0))
        self.assertFalse(ok)
        self.assertIn("outside allowed range", msg)

    def test_wrong_sentence_count(self):
        doc = make_good_doc(n=55, distance=10, k=0)
        doc["sentences"] = doc["sentences"][:50]  # only 50
        ok, msg = validate_document(doc, DocConfig(55, 10, 0))
        self.assertFalse(ok)
        self.assertIn("expected 55 sentences, got 50", msg)

    def test_non_string_sentence(self):
        doc = make_good_doc(n=55, distance=10, k=0)
        doc["sentences"][3] = 42
        ok, msg = validate_document(doc, DocConfig(55, 10, 0))
        self.assertFalse(ok)
        self.assertIn("not all sentences are strings", msg)


class TestCheck3_ContradictionPairFormat(unittest.TestCase):
    def test_cp_not_list(self):
        doc = make_good_doc(n=55, distance=10, k=0)
        doc["contradiction_pair"] = (5, 15)  # tuple, not list
        ok, msg = validate_document(doc, DocConfig(55, 10, 0))
        self.assertFalse(ok)
        self.assertIn("list of exactly 2 elements", msg)

    def test_cp_three_elements(self):
        doc = make_good_doc(n=55, distance=10, k=0)
        doc["contradiction_pair"] = [5, 15, 25]
        ok, msg = validate_document(doc, DocConfig(55, 10, 0))
        self.assertFalse(ok)
        self.assertIn("list of exactly 2 elements", msg)

    def test_cp_non_integer(self):
        doc = make_good_doc(n=55, distance=10, k=0)
        doc["contradiction_pair"] = [5, "15"]
        ok, msg = validate_document(doc, DocConfig(55, 10, 0))
        self.assertFalse(ok)
        self.assertIn("must be integers", msg)

    def test_cp_not_distinct(self):
        doc = make_good_doc(n=55, distance=10, k=0)
        doc["contradiction_pair"] = [5, 5]
        ok, msg = validate_document(doc, DocConfig(55, 10, 0))
        self.assertFalse(ok)
        self.assertIn("distinct", msg)


class TestCheck4_ContradictionPairBounds(unittest.TestCase):
    def test_index_zero(self):
        doc = make_good_doc(n=55, distance=10, k=0)
        doc["contradiction_pair"] = [0, 10]
        ok, msg = validate_document(doc, DocConfig(55, 10, 0))
        self.assertFalse(ok)
        self.assertIn("in [1, 55]", msg)

    def test_index_exceeds_n(self):
        doc = make_good_doc(n=55, distance=10, k=0)
        doc["contradiction_pair"] = [50, 60]
        ok, msg = validate_document(doc, DocConfig(55, 10, 0))
        self.assertFalse(ok)
        self.assertIn("in [1, 55]", msg)


class TestCheck5_Distance(unittest.TestCase):
    def test_wrong_distance(self):
        doc = make_good_doc(n=55, distance=10, k=0)
        doc["contradiction_pair"] = [5, 20]  # distance=15, not 10
        ok, msg = validate_document(doc, DocConfig(55, 10, 0))
        self.assertFalse(ok)
        self.assertIn("expected 10", msg)

    def test_correct_distance(self):
        doc = make_good_doc(n=55, distance=10, k=0)
        ok, _ = validate_document(doc, DocConfig(55, 10, 0))
        self.assertTrue(ok)


class TestCheck6_DistractorPairs(unittest.TestCase):
    def test_wrong_distractor_count(self):
        doc = make_good_doc(n=55, distance=10, k=2)
        ok, msg = validate_document(doc, DocConfig(55, 10, 4))  # expect 4, got 2
        self.assertFalse(ok)
        self.assertIn("expected 4 distractor pairs, got 2", msg)

    def test_distractor_not_list(self):
        doc = make_good_doc(n=55, distance=10, k=0)
        doc["distractor_pairs"] = [[20, 26], (30, 36)]
        ok, msg = validate_document(doc, DocConfig(55, 10, 2))
        # (30, 36) is a tuple → check 6 triggers
        self.assertFalse(ok)

    def test_distractor_out_of_bounds(self):
        doc = make_good_doc(n=55, distance=10, k=0)
        doc["distractor_pairs"] = [[60, 65]]  # both > n=55
        ok, msg = validate_document(doc, DocConfig(55, 10, 1))
        self.assertFalse(ok)
        self.assertIn("in [1, 55]", msg)

    def test_distractor_not_distinct(self):
        doc = make_good_doc(n=55, distance=10, k=0)
        doc["distractor_pairs"] = [[20, 20]]
        ok, msg = validate_document(doc, DocConfig(55, 10, 1))
        self.assertFalse(ok)
        self.assertIn("distinct", msg)


class TestCheck7_MinSeparation(unittest.TestCase):
    def test_contradiction_pair_too_close(self):
        doc = make_good_doc(n=55, distance=5, k=0)
        doc["contradiction_pair"] = [5, 7]  # distance=2 < 3
        ok, msg = validate_document(doc, DocConfig(55, 5, 0))
        # distance check fires first (5≠2), not separation check
        self.assertFalse(ok)

    def test_distractor_pair_separation_2(self):
        doc = make_good_doc(n=55, distance=10, k=0)
        doc["distractor_pairs"] = [[20, 22]]  # |20-22|=2 < 3
        ok, msg = validate_document(doc, DocConfig(55, 10, 1))
        self.assertFalse(ok)
        self.assertIn(">= 3", msg)

    def test_distractor_pair_separation_exactly_3(self):
        doc = make_good_doc(n=55, distance=10, k=0)
        doc["distractor_pairs"] = [[20, 23]]  # |20-23|=3 — passes
        ok, _ = validate_document(doc, DocConfig(55, 10, 1))
        self.assertTrue(ok)


class TestCheck8_NoDuplicateIndices(unittest.TestCase):
    def test_distractor_overlaps_contradiction(self):
        doc = make_good_doc(n=55, distance=10, k=0)
        # contradiction_pair = [5, 15]; distractor uses index 5
        doc["distractor_pairs"] = [[5, 25]]
        ok, msg = validate_document(doc, DocConfig(55, 10, 1))
        self.assertFalse(ok)
        self.assertIn("duplicate", msg)

    def test_two_distractors_share_index(self):
        doc = make_good_doc(n=55, distance=10, k=0)
        doc["distractor_pairs"] = [[20, 26], [26, 36]]  # 26 appears twice
        ok, msg = validate_document(doc, DocConfig(55, 10, 2))
        self.assertFalse(ok)
        self.assertIn("duplicate", msg)


class TestValidDocument(unittest.TestCase):
    def test_valid_no_distractors(self):
        doc = make_good_doc(n=55, distance=10, k=0)
        ok, msg = validate_document(doc, DocConfig(55, 10, 0))
        self.assertTrue(ok)
        self.assertEqual(msg, "")

    def test_valid_with_distractors(self):
        doc = make_good_doc(n=55, distance=10, k=2)
        ok, msg = validate_document(doc, DocConfig(55, 10, 2))
        self.assertTrue(ok)
        self.assertEqual(msg, "")

    def test_valid_boundary_n_52(self):
        sentences = [f"S{x}." for x in range(1, 53)]
        doc = {
            "sentences": sentences,
            "contradiction_pair": [1, 6],
            "distractor_pairs": [],
        }
        ok, msg = validate_document(doc, DocConfig(52, 5, 0))
        self.assertTrue(ok)
        self.assertEqual(msg, "")

    def test_valid_boundary_n_65(self):
        sentences = [f"S{x}." for x in range(1, 66)]
        doc = {
            "sentences": sentences,
            "contradiction_pair": [17, 65],
            "distractor_pairs": [],
        }
        ok, msg = validate_document(doc, DocConfig(65, 48, 0))
        self.assertTrue(ok)
        self.assertEqual(msg, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
