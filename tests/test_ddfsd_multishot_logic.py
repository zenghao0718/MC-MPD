import copy
import unittest

from util.ddfsd_multishot_logic import (
    build_multishot_support_query_indices,
    nested_support_indices,
    parse_shot_list,
    resolve_checkpoint_step,
    sample_metadata_indices,
    select_real_and_nearest_fake_logits,
    validate_config_consistency,
)

try:
    import torch
    from model.ddfsd_losses import compute_alpha, compute_prototypes, compute_support_sigmas
except ModuleNotFoundError:
    torch = None


class MultiShotLogicTest(unittest.TestCase):
    def test_shots_are_sorted_and_deduplicated(self):
        self.assertEqual(parse_shot_list("20,0,2,2,1"), [0, 1, 2, 20])

    def test_negative_shot_rejected(self):
        with self.assertRaises(ValueError):
            parse_shot_list("0,-1")

    def test_nested_support_protocol(self):
        candidates, _ = build_multishot_support_query_indices(80, 20, 42)
        for shot in (1, 2, 5, 10, 20):
            self.assertEqual(nested_support_indices(candidates, shot), candidates[:shot])
        self.assertEqual(nested_support_indices(candidates, 1), nested_support_indices(candidates, 20)[:1])
        self.assertEqual(nested_support_indices(candidates, 10), nested_support_indices(candidates, 20)[:10])

    def test_all_shots_share_the_same_fixed_query(self):
        query_by_shot = {}
        for shot in (1, 2, 5, 10, 20):
            candidates, query = build_multishot_support_query_indices(80, 20, 101)
            nested_support_indices(candidates, shot)
            query_by_shot[shot] = query
        for query in query_by_shot.values():
            self.assertEqual(query, query_by_shot[20])
        self.assertFalse(set(nested_support_indices(candidates, 20)) & set(query_by_shot[20]))

    def test_metadata_is_without_replacement(self):
        selected = sample_metadata_indices(100, 20, 7)
        self.assertEqual(len(selected), len(set(selected)))
        with self.assertRaises(ValueError):
            sample_metadata_indices(10, 11, 7)

    @unittest.skipUnless(torch is not None, "torch is unavailable in this Python environment")
    def test_one_shot_dual_alpha_is_half(self):
        support_rgb = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        support_freq = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
        proto_rgb, proto_freq = compute_prototypes(support_rgb, support_freq, model_mode="dual")
        sigma_rgb, sigma_freq = compute_support_sigmas(
            support_rgb, support_freq, proto_rgb, proto_freq
        )
        alpha = compute_alpha(sigma_rgb, sigma_freq, tau_r=0.1)
        self.assertTrue(torch.allclose(sigma_rgb, torch.zeros_like(sigma_rgb)))
        self.assertTrue(torch.allclose(sigma_freq, torch.zeros_like(sigma_freq)))
        self.assertTrue(torch.allclose(alpha, torch.full_like(alpha, 0.5)))

    def test_zero_shot_uses_nearest_fake_prototype(self):
        real, fake = select_real_and_nearest_fake_logits(
            [[-0.2, -3.0, -0.5, -1.5], [-1.0, -0.8, -2.0, -4.0]]
        )
        self.assertEqual(real, [-0.2, -1.0])
        self.assertEqual(fake, [-0.5, -0.8])

    def test_checkpoint_step_resolution(self):
        self.assertEqual(resolve_checkpoint_step(15000, 15000, "model.pth"), 15000)
        self.assertEqual(resolve_checkpoint_step(0, 15000, "model.pth"), 15000)
        self.assertEqual(resolve_checkpoint_step(15000, 0, "model.pth"), 15000)
        self.assertEqual(resolve_checkpoint_step(0, 0, "model.pth"), 0)
        with self.assertRaisesRegex(ValueError, "Requested step 14000.*15000.*model.pth"):
            resolve_checkpoint_step(14000, 15000, "model.pth")

    def test_aggregate_config_consistency(self):
        base = {
            "shot_list": [0, 1, 2, 5, 10, 20], "seeds": [42, 101],
            "zero_shot_metadata_per_class": 1024, "max_shot": 20,
            "max_eval_query_per_class": 0, "ckpt_step": 15000,
            "checkpoint_model_mode": "dual", "model_mode": "dual",
            "branch_mode": "dual", "tau": 0.2, "tau_r": 0.1,
        }
        validate_config_consistency({"ADM": copy.deepcopy(base), "BigGAN": copy.deepcopy(base)})
        for field, different in (("ckpt_step", 14000), ("shot_list", [0, 10]),
                                 ("branch_mode", "rgb-only")):
            configs = {"ADM": copy.deepcopy(base), "BigGAN": copy.deepcopy(base)}
            configs["BigGAN"][field] = different
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                validate_config_consistency(configs)


if __name__ == "__main__":
    unittest.main()
