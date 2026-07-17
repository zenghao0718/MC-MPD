"""Pure-standard-library tests for MS COCOAI grouping/splitting logic."""

import hashlib
import unittest

from tools.build_ms_cocoai_fewshot_manifests import choose_support_groups
from tools.build_ms_cocoai_groups import build_groups


class GroupingTests(unittest.TestCase):
    def test_caption_occurrences_pair_across_shards(self):
        caption = "same caption"
        caption_sha = hashlib.sha256(caption.encode("utf-8")).hexdigest()
        rows = []
        global_index = 0
        for occurrence in range(2):
            for label in range(6):
                rows.append({
                    "caption": caption,
                    "caption_sha256": caption_sha,
                    "label_b": str(label),
                    "shard_index": str((label + occurrence) % 2),
                    "row_index_in_shard": str(occurrence * 10 + label),
                    "global_row_index": str(global_index),
                    "image_path": f"/{occurrence}_{label}.jpg",
                    "image_sha256": f"sha-{occurrence}-{label}",
                })
                global_index += 1
        grouped, groups, anomalies = build_groups(rows)
        self.assertEqual(anomalies, [])
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(grouped), 12)
        self.assertEqual({row["group_id"] for row in grouped}, {
            f"{caption_sha[:16]}_0", f"{caption_sha[:16]}_1"
        })

    def test_unequal_label_counts_fail_grouping(self):
        caption = "broken"
        caption_sha = hashlib.sha256(caption.encode("utf-8")).hexdigest()
        rows = [{
            "caption": caption, "caption_sha256": caption_sha, "label_b": str(label),
            "shard_index": "0", "row_index_in_shard": str(label),
            "global_row_index": str(label), "image_path": f"/{label}.jpg",
            "image_sha256": f"sha-{label}",
        } for label in range(5)]
        grouped, groups, anomalies = build_groups(rows)
        self.assertEqual(grouped, [])
        self.assertEqual(groups, [])
        self.assertEqual(len(anomalies), 1)


class SplitTests(unittest.TestCase):
    def test_support_selection_is_deterministic_and_unique(self):
        group_ids = [f"group-{index:03d}" for index in range(100)]
        selected_a, remaining_a = choose_support_groups(group_ids, 20260717, 50)
        selected_b, remaining_b = choose_support_groups(reversed(group_ids), 20260717, 50)
        self.assertEqual(selected_a, selected_b)
        self.assertEqual(remaining_a, remaining_b)
        self.assertEqual(len(set(selected_a)), 50)
        self.assertFalse(set(selected_a) & set(remaining_a))


if __name__ == "__main__":
    unittest.main()
