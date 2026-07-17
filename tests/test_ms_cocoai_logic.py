"""Pure-standard-library tests for MS COCOAI grouping/splitting logic."""

import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from test_ddfsd_transfer import (
    EXPECTED_FAKE_CLASSES,
    EXPECTED_SOURCE_CLASSES,
    RESULT_FILES,
    prepare_output_directory,
    validate_formal_artifacts,
    validate_manifest_rows,
)
from tools.build_ms_cocoai_fewshot_manifests import (
    GENERATORS,
    SEEDS,
    choose_support_groups,
    partition_support_groups,
    sha256_file,
    task_rows,
    validate_task,
    verify_manifest_lock,
)
from tools.build_ms_cocoai_groups import build_groups


def grouping_row(caption, label, occurrence, image_sha=None):
    caption_sha = hashlib.sha256(caption.encode("utf-8")).hexdigest()
    return {
        "caption": caption,
        "caption_sha256": caption_sha,
        "label_b": str(label),
        "shard_index": str(label % 3),
        "row_index_in_shard": str(occurrence * 100 + label),
        "global_row_index": str(occurrence * 1000 + label),
        "image_path": f"/occurrence_{occurrence}/label_{label}.jpg",
        "image_sha256": image_sha or f"sha-{occurrence}-{label}",
        "occurrence_marker": str(occurrence),
    }


def task_row(group_id, label, image_sha, role="support"):
    return {
        "group_id": group_id,
        "label": str(label),
        "image_path": f"/{role}/{image_sha}.jpg",
        "image_sha256": image_sha,
    }


def fake_complete_groups(count=120):
    groups = {}
    for group_index in range(count):
        group_id = f"group-{group_index:03d}"
        groups[group_id] = {
            label: {
                "group_id": group_id,
                "label_b": str(label),
                "image_path": f"/{group_id}/{label}.jpg",
                "image_sha256": f"sha-{group_id}-{label}",
            }
            for label in range(6)
        }
    return groups


class GroupingTests(unittest.TestCase):
    def test_kth_occurrences_pair_correctly_across_shards(self):
        caption = "same caption crossing shards"
        rows = [
            grouping_row(caption, label, occurrence)
            for occurrence in range(2)
            for label in range(6)
        ]
        random.Random(7).shuffle(rows)
        grouped, groups, anomalies = build_groups(rows)
        self.assertEqual(anomalies, [])
        self.assertEqual(len(groups), 2)
        caption_sha = hashlib.sha256(caption.encode("utf-8")).hexdigest()
        for occurrence in (0, 1):
            group_id = f"{caption_sha[:16]}_{occurrence}"
            members = [row for row in grouped if row["group_id"] == group_id]
            self.assertEqual({int(row["label_b"]) for row in members}, set(range(6)))
            self.assertEqual({row["occurrence_marker"] for row in members}, {str(occurrence)})
            self.assertEqual(
                {row["image_path"] for row in members},
                {f"/occurrence_{occurrence}/label_{label}.jpg" for label in range(6)},
            )
        self.assertGreater(len({row["shard_index"] for row in rows}), 1)

    def test_unequal_label_counts_fail_grouping(self):
        rows = [grouping_row("broken", label, 0) for label in range(5)]
        grouped, groups, anomalies = build_groups(rows)
        self.assertEqual(grouped, [])
        self.assertEqual(groups, [])
        self.assertEqual(len(anomalies), 1)

    def test_duplicate_sha_inside_group_fails(self):
        rows = [
            grouping_row("duplicate", label, 0, image_sha="duplicate-sha" if label < 2 else None)
            for label in range(6)
        ]
        grouped, groups, anomalies = build_groups(rows)
        self.assertEqual(grouped, [])
        self.assertEqual(groups, [])
        self.assertIn("duplicate image_sha256", anomalies[0]["reason"])


class ManifestLeakageTests(unittest.TestCase):
    def make_valid_task(self):
        support = []
        for index in range(10):
            support.extend([
                task_row(f"support-{index}", 0, f"support-real-{index}"),
                task_row(f"support-{index}", 1, f"support-fake-{index}"),
            ])
        query = []
        for index in range(4):
            query.extend([
                task_row(f"query-{index}", 0, f"query-real-{index}", role="query"),
                task_row(f"query-{index}", 1, f"query-fake-{index}", role="query"),
            ])
        return support, query

    def test_support_query_sha_overlap_fails_in_builder_and_evaluator(self):
        support, query = self.make_valid_task()
        query[0]["image_sha256"] = support[0]["image_sha256"]
        with self.assertRaisesRegex(ValueError, "SHA"):
            validate_task(support, query)
        with self.assertRaisesRegex(ValueError, "SHA"):
            validate_manifest_rows(support, query)

    def test_duplicate_query_sha_fails(self):
        support, query = self.make_valid_task()
        query[1]["image_sha256"] = query[0]["image_sha256"]
        with self.assertRaisesRegex(ValueError, "duplicate image SHA"):
            validate_task(support, query)

    def test_duplicate_support_sha_fails(self):
        support, query = self.make_valid_task()
        support[1]["image_sha256"] = support[0]["image_sha256"]
        with self.assertRaisesRegex(ValueError, "duplicate image SHA"):
            validate_manifest_rows(support, query)


class SplitTests(unittest.TestCase):
    def test_five_seed_support_is_disjoint_and_query_is_common(self):
        groups = fake_complete_groups()
        support_pool, query_group_ids = choose_support_groups(groups, 20260717, 50)
        partition = partition_support_groups(support_pool)
        self.assertEqual(set(partition), set(SEEDS))
        for left_index, left_seed in enumerate(SEEDS):
            for right_seed in SEEDS[left_index + 1:]:
                self.assertFalse(set(partition[left_seed]) & set(partition[right_seed]))
        query_signatures = {}
        for generator, fake_label in GENERATORS.items():
            for seed in SEEDS:
                support = task_rows(groups, partition[seed], fake_label, "support", generator, seed)
                query = task_rows(groups, query_group_ids, fake_label, "query", generator, seed)
                validate_task(support, query)
                signature = tuple((row["group_id"], row["image_sha256"]) for row in query)
                query_signatures.setdefault(generator, signature)
                self.assertEqual(signature, query_signatures[generator])
        real_queries = {
            tuple(groups[group_id][0]["image_sha256"] for group_id in query_group_ids)
            for _generator in GENERATORS
        }
        self.assertEqual(len(real_queries), 1)

    def test_support_selection_is_deterministic(self):
        group_ids = [f"group-{index:03d}" for index in range(100)]
        selected_a, remaining_a = choose_support_groups(group_ids, 20260717, 50)
        selected_b, remaining_b = choose_support_groups(reversed(group_ids), 20260717, 50)
        self.assertEqual(selected_a, selected_b)
        self.assertEqual(remaining_a, remaining_b)


class OutputProtectionTests(unittest.TestCase):
    def test_identical_complete_result_is_safely_skipped(self):
        identity = {"checkpoint_sha256": "a", "formal_result": True}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in RESULT_FILES:
                content = "header\nrow\n" if name == "per_image_scores.csv" else "{}"
                (root / name).write_text(content, encoding="utf-8")
            (root / "provenance.json").write_text(json.dumps(identity), encoding="utf-8")
            self.assertTrue(prepare_output_directory(root, identity, overwrite=False))

    def test_mismatched_result_requires_explicit_overwrite(self):
        old_identity = {"checkpoint_sha256": "old", "formal_result": False}
        new_identity = {"checkpoint_sha256": "new", "formal_result": True}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in RESULT_FILES:
                content = "header\nrow\n" if name == "per_image_scores.csv" else "{}"
                (root / name).write_text(content, encoding="utf-8")
            (root / "provenance.json").write_text(json.dumps(old_identity), encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_output_directory(root, new_identity, overwrite=False)
            self.assertFalse(prepare_output_directory(root, new_identity, overwrite=True))


class ManifestLockTests(unittest.TestCase):
    def test_lock_verification_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "support.csv"
            manifest.write_text("label\n0\n", encoding="utf-8")
            lock_path = root / "manifest_lock.json"
            lock = {
                "files": [{"path": str(manifest.resolve()), "sha256": sha256_file(manifest)}]
            }
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            (root / "manifest_sha256.txt").write_text(
                f"{sha256_file(manifest)}  {manifest.resolve()}\n"
                f"{sha256_file(lock_path)}  {lock_path.resolve()}\n",
                encoding="utf-8",
            )
            self.assertEqual(verify_manifest_lock(lock_path)["files"], lock["files"])
            manifest.write_text("label\n1\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                verify_manifest_lock(lock_path)


class FormalValidationTests(unittest.TestCase):
    def test_checkpoint_stats_and_hyperparameters_are_bound(self):
        metadata = {
            "training_scope": "all-source", "dataset": "GenImage", "split": "train",
            "classes": EXPECTED_SOURCE_CLASSES,
        }
        checkpoint = {
            "step": 15000,
            "model_mode": "dual",
            "training_scope": "all-source",
            "source_fake_classes": EXPECTED_FAKE_CLASSES,
            "source_classes": EXPECTED_SOURCE_CLASSES,
            "freq_stats_sha256": "stats-sha",
            "freq_stats_metadata": metadata,
            "config": {"tau": 0.2, "tau_r": 0.1},
        }
        args = SimpleNamespace(branch_mode="dual", tau=0.2, tau_r=0.1)
        _actual, mismatches = validate_formal_artifacts(checkpoint, metadata, "stats-sha", args)
        self.assertEqual(mismatches, [])
        checkpoint["source_fake_classes"] = list(reversed(EXPECTED_FAKE_CLASSES))
        _actual, mismatches = validate_formal_artifacts(checkpoint, metadata, "stats-sha", args)
        self.assertTrue(any("source_fake_classes" in item for item in mismatches))


if __name__ == "__main__":
    unittest.main()
