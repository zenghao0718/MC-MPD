import unittest

from util.ddfsd_multishot_eval import build_multishot_support_query_indices, parse_shot_list, sample_metadata_indices


class MultiShotLogicTest(unittest.TestCase):
    def test_shots_are_sorted_and_deduplicated(self):
        self.assertEqual(parse_shot_list("20,0,2,2,1"), [0, 1, 2, 20])

    def test_negative_shot_rejected(self):
        with self.assertRaises(ValueError): parse_shot_list("0,-1")

    def test_fixed_split_is_deterministic_and_disjoint(self):
        support, query = build_multishot_support_query_indices(50, 20, 42)
        self.assertEqual((support, query), build_multishot_support_query_indices(50, 20, 42))
        self.assertFalse(set(support) & set(query)); self.assertEqual(len(support), 20)
        self.assertEqual(support[:5], support[0:5])

    def test_metadata_is_without_replacement(self):
        selected = sample_metadata_indices(100, 20, 7)
        self.assertEqual(len(selected), len(set(selected)))
        with self.assertRaises(ValueError): sample_metadata_indices(10, 11, 7)


if __name__ == "__main__": unittest.main()
