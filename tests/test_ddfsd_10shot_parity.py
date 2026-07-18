import copy
import unittest

from tools.check_ddfsd_10shot_parity import compare_parity_rows


def row(seed, **updates):
    value = {
        "exclude_class": "ADM",
        "seed": str(seed),
        "support_shot": "10",
        "ckpt_step": "15000",
        "checkpoint_model_mode": "dual",
        "branch_mode": "dual",
        "num_real_support": "10",
        "num_fake_support": "10",
        "num_real_query": "90",
        "num_fake_query": "70",
        "acc": "0.8",
        "ap": "0.9",
        "auc": "0.85",
    }
    value.update({key: str(item) for key, item in updates.items()})
    return value


class TenShotParityTest(unittest.TestCase):
    def setUp(self):
        self.new = [row(42), row(101)]
        self.reference = copy.deepcopy(self.new)

    def test_exact_and_tolerance_match_pass(self):
        self.new[0]["acc"] = str(float(self.reference[0]["acc"]) + 1e-6)
        differences, errors = compare_parity_rows(
            self.new, self.reference, expected_seeds=[42, 101], tolerance=1e-6
        )
        self.assertFalse(errors)
        self.assertTrue(differences)
        self.assertTrue(all(item["status"] == "PASS" for item in differences))

    def test_missing_seed_fails(self):
        _, errors = compare_parity_rows(
            self.new[:1], self.reference, expected_seeds=[42, 101]
        )
        self.assertTrue(any("missing seeds" in error for error in errors))

    def test_duplicate_seed_fails(self):
        _, errors = compare_parity_rows(
            self.new + [copy.deepcopy(self.new[0])],
            self.reference,
            expected_seeds=[42, 101],
        )
        self.assertTrue(any("duplicate seed" in error for error in errors))

    def test_query_count_mismatch_fails(self):
        self.new[0]["num_fake_query"] = "69"
        _, errors = compare_parity_rows(
            self.new, self.reference, expected_seeds=[42, 101]
        )
        self.assertTrue(any("num_fake_query differs" in error for error in errors))

    def test_metric_beyond_tolerance_fails_without_mutating_inputs(self):
        before = copy.deepcopy(self.new)
        self.new[0]["auc"] = "0.850002"
        expected_after_change = copy.deepcopy(self.new)
        _, errors = compare_parity_rows(
            self.new, self.reference, expected_seeds=[42, 101], tolerance=1e-6
        )
        self.assertTrue(any("auc difference" in error for error in errors))
        self.assertEqual(self.new, expected_after_change)
        self.assertNotEqual(self.new, before)


if __name__ == "__main__":
    unittest.main()
