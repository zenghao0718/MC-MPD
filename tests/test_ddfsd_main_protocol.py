import inspect
import os
import tempfile
import unittest
from unittest import mock

from test_ddfsd import sampling_manifest_rows
from util.ddfsd_main_protocol import (
    InsufficientValidMetadataImages,
    audit_image_paths,
    build_valid_image_indices,
    build_zero_shot_query_indices,
    sample_metadata_paths_lazy_strict,
    sample_metadata_from_valid_indices,
    zero_shot_metadata_classes,
    zero_shot_metadata_manifest_rows,
)

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None

from util.ddfsd_sampling import sample_support_query_indices

try:
    from util.ddfsd_eval import binary_metrics, evaluate_binary_few_shot
except (ImportError, RuntimeError):
    binary_metrics = None
    evaluate_binary_few_shot = None

try:
    from datasets.ddfsd_datasets import (
        FormalEvalImagePathDataset,
        ImagePathDataset,
    )
except (ImportError, RuntimeError):
    FormalEvalImagePathDataset = None
    ImagePathDataset = None


class MainProtocolSamplingTest(unittest.TestCase):
    def test_positive_shots_use_prefix_support_and_remaining_query(self):
        by_shot = {
            shot: sample_support_query_indices(80, shot, seed=42, max_query=None)
            for shot in (5, 10, 20, 30, 50)
        }
        for shot, (support, query) in by_shot.items():
            self.assertEqual(len(support), shot)
            self.assertEqual(len(query), 80 - shot)
            self.assertFalse(set(support) & set(query))
            self.assertEqual(sorted(support + query), list(range(80)))
        self.assertEqual(by_shot[5][0], by_shot[10][0][:5])
        self.assertEqual(by_shot[10][0], by_shot[20][0][:10])
        self.assertEqual(by_shot[20][0], by_shot[30][0][:20])
        self.assertEqual(by_shot[30][0], by_shot[50][0][:30])
        self.assertNotEqual(by_shot[5][1], by_shot[50][1])

    def test_different_seeds_and_unbalanced_class_sizes(self):
        support_42, real_query = sample_support_query_indices(100, 5, 42)
        support_101, _ = sample_support_query_indices(100, 5, 101)
        _, fake_query = sample_support_query_indices(73, 5, 42)
        self.assertNotEqual(support_42, support_101)
        self.assertEqual(len(real_query), 95)
        self.assertEqual(len(fake_query), 68)
        self.assertNotEqual(len(real_query), len(fake_query))

    @unittest.skipUnless(
        evaluate_binary_few_shot is not None, "DDFSD runtime dependencies unavailable"
    )
    def test_index_return_is_opt_in(self):
        parameter = inspect.signature(evaluate_binary_few_shot).parameters[
            "return_indices"
        ]
        self.assertFalse(parameter.default)

    def test_manifest_is_built_from_the_returned_sampling(self):
        sampling = {
            "real": {
                "support_indices": [2, 0],
                "query_indices": [1],
                "paths": ["real-0", "real-1", "real-2"],
            },
            "ADM": {
                "support_indices": [1, 2],
                "query_indices": [0],
                "paths": ["fake-0", "fake-1", "fake-2"],
            },
        }
        rows = sampling_manifest_rows("ADM", 42, 2, sampling)
        real_support = [
            item
            for item in rows
            if item["data_class"] == "real" and item["role"] == "support"
        ]
        self.assertEqual([item["dataset_index"] for item in real_support], [2, 0])
        self.assertEqual([item["support_rank"] for item in real_support], [1, 2])
        self.assertEqual(
            [item["filepath"] for item in real_support], ["real-2", "real-0"]
        )
        self.assertEqual(
            [item["filepath"] for item in rows if item["role"] == "query"],
            ["real-1", "fake-0"],
        )


class BinaryMetricsTest(unittest.TestCase):
    @unittest.skipUnless(
        binary_metrics is not None, "scikit-learn/torch dependencies unavailable"
    )
    def test_unbalanced_acc_and_balanced_acc_are_distinct(self):
        metrics = binary_metrics(
            labels=[0, 0, 0, 1],
            fake_scores=[0.1, 0.2, 0.8, 0.9],
            preds=[0, 0, 1, 1],
        )
        self.assertAlmostEqual(metrics["acc"], 0.75)
        self.assertAlmostEqual(metrics["real_acc"], 2 / 3)
        self.assertAlmostEqual(metrics["fake_acc"], 1.0)
        self.assertAlmostEqual(metrics["balanced_acc"], 5 / 6)
        self.assertNotEqual(metrics["acc"], metrics["balanced_acc"])
        for value in metrics.values():
            self.assertTrue(0.0 <= value <= 1.0)

    @unittest.skipUnless(
        binary_metrics is not None, "scikit-learn/torch dependencies unavailable"
    )
    def test_both_query_classes_are_required(self):
        with self.assertRaisesRegex(ValueError, "both real.*fake"):
            binary_metrics([0, 0], [0.1, 0.2], [0, 0])


@unittest.skipUnless(Image is not None, "Pillow is unavailable")
class MetadataValidityTest(unittest.TestCase):
    def test_full_decode_filter_and_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index in range(4):
                path = os.path.join(directory, f"valid_{index}.png")
                Image.new("RGB", (4, 4), color=(index, 0, 0)).save(path)
                paths.append(path)
            corrupt = os.path.join(directory, "corrupt.png")
            with open(corrupt, "wb") as handle:
                handle.write(b"not an image")
            text_path = os.path.join(directory, "notes.txt")
            with open(text_path, "w", encoding="utf-8") as handle:
                handle.write("not an image either")
            paths.extend([corrupt, text_path])

            invalid = []
            valid = build_valid_image_indices(paths, invalid)
            self.assertEqual(valid, [0, 1, 2, 3])
            self.assertEqual([row["dataset_index"] for row in invalid], [4, 5])
            selected = sample_metadata_from_valid_indices(valid, 3, seed=42)
            self.assertEqual(len(selected), len(set(selected)))
            self.assertTrue(set(selected).issubset(valid))
            self.assertEqual(
                selected, sample_metadata_from_valid_indices(valid, 3, seed=42)
            )
            with self.assertRaisesRegex(ValueError, "only 4 valid images"):
                sample_metadata_from_valid_indices(valid, 5, seed=42)

    def test_lazy_sampling_stops_as_soon_as_requested_count_is_full(self):
        paths = [f"image_{index:04d}.png" for index in range(1100)]
        decoded = []
        selected, invalid, visited = sample_metadata_paths_lazy_strict(
            paths, 1024, "/data", "ADM", "real", 42, decoded.append
        )
        self.assertEqual(len(selected), 1024)
        self.assertEqual(len(set(selected)), 1024)
        self.assertEqual(len(decoded), 1024)
        self.assertEqual(visited, 1024)
        self.assertEqual(invalid, [])
        rows = zero_shot_metadata_manifest_rows(
            paths, selected, "ADM", "real", 42
        )
        self.assertEqual(len(rows), 1024)
        self.assertEqual([row["sample_rank"] for row in rows], list(range(1, 1025)))
        self.assertEqual(
            [row["image_path"] for row in rows], [paths[index] for index in selected]
        )

    def test_lazy_sampling_records_bad_candidate_and_continues(self):
        paths = [f"image_{index}.png" for index in range(8)]
        first = []
        sample_metadata_paths_lazy_strict(
            paths, 1, "/data", "ADM", "real", 42, first.append
        )
        bad_path = first[0]

        def decode(path):
            if path == bad_path:
                raise OSError("corrupt fixture")

        selected, invalid, visited = sample_metadata_paths_lazy_strict(
            paths, 2, "/data", "ADM", "real", 42, decode
        )
        self.assertEqual(len(selected), 2)
        self.assertEqual(visited, 3)
        self.assertEqual([row["filepath"] for row in invalid], [bad_path])
        self.assertEqual(invalid[0]["error_type"], "OSError")

    def test_lazy_sampling_is_reproducible_and_seed_independent(self):
        paths = [f"image_{index:03d}.png" for index in range(30)]
        arguments = (paths, 8, "/data", "ADM", "real")
        first = sample_metadata_paths_lazy_strict(*arguments, 42, lambda path: None)
        repeated = sample_metadata_paths_lazy_strict(
            *arguments, 42, lambda path: None
        )
        different = sample_metadata_paths_lazy_strict(
            *arguments, 101, lambda path: None
        )
        self.assertEqual(first, repeated)
        self.assertNotEqual(first[0], different[0])

    def test_lazy_sampling_fails_after_exhausting_insufficient_valid_images(self):
        paths = ["valid.png", "broken.png"]

        def decode(path):
            if path == "broken.png":
                raise ValueError("bad image")

        with self.assertRaises(InsufficientValidMetadataImages) as raised:
            sample_metadata_paths_lazy_strict(
                paths, 2, "/data", "ADM", "real", 42, decode
            )
        self.assertEqual(raised.exception.visited_count, 2)
        self.assertEqual(
            [row["filepath"] for row in raised.exception.invalid_records],
            ["broken.png"],
        )

    def test_lazy_sampling_does_not_depend_on_builtin_hash(self):
        with mock.patch("builtins.hash", side_effect=AssertionError("unstable hash")):
            selected, _, _ = sample_metadata_paths_lazy_strict(
                ["a.png", "b.png"],
                1,
                "/data",
                "ADM",
                "real",
                42,
                lambda path: None,
            )
        self.assertEqual(len(selected), 1)

    def test_lazy_manifest_candidates_are_strictly_decodable(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index in range(6):
                path = os.path.join(directory, f"valid_{index}.png")
                Image.new("RGB", (4, 4), color=(index, 0, 0)).save(path)
                paths.append(path)
            corrupt = os.path.join(directory, "corrupt.png")
            with open(corrupt, "wb") as handle:
                handle.write(b"broken")
            paths.append(corrupt)
            selected, _, _ = sample_metadata_paths_lazy_strict(
                paths, 5, directory, "ADM", "real", 42
            )
            self.assertEqual(len(selected), 5)
            for index in selected:
                with Image.open(paths[index]) as image:
                    image.convert("RGB").load()


@unittest.skipUnless(
    Image is not None and FormalEvalImagePathDataset is not None,
    "Pillow/DDFSD dataset dependencies are unavailable",
)
class StrictFormalImageDatasetTest(unittest.TestCase):
    def test_formal_eval_never_replaces_corrupt_path(self):
        with tempfile.TemporaryDirectory() as directory:
            corrupt = os.path.join(directory, "00_corrupt.png")
            valid = os.path.join(directory, "01_valid.png")
            with open(corrupt, "wb") as handle:
                handle.write(b"broken")
            Image.new("RGB", (4, 4), color=(10, 20, 30)).save(valid)

            training_dataset = ImagePathDataset(directory, transform=None)
            strict_dataset = FormalEvalImagePathDataset(directory, transform=None)
            # Existing training behavior remains tolerant and advances to the valid image.
            training_image, _ = training_dataset[0]
            self.assertEqual(training_image.getpixel((0, 0)), (10, 20, 30))
            # Formal evaluation keeps index/path identity and fails at the corrupt path.
            self.assertEqual(strict_dataset.paths[0], corrupt)
            with self.assertRaisesRegex(RuntimeError, "00_corrupt.png"):
                strict_dataset[0]
            strict_image, _ = strict_dataset[1]
            self.assertEqual(strict_dataset.paths[1], valid)
            self.assertEqual(strict_image.getpixel((0, 0)), (10, 20, 30))
            audit = audit_image_paths(strict_dataset.paths, "real", "val")
            self.assertEqual([row["filepath"] for row in audit], [corrupt])

    def test_manifest_path_matches_strict_dataset_index(self):
        with tempfile.TemporaryDirectory() as directory:
            for index in range(3):
                Image.new("RGB", (4, 4), color=(index, 0, 0)).save(
                    os.path.join(directory, f"{index}.png")
                )
            dataset = FormalEvalImagePathDataset(directory, transform=None)
            sampling = {
                "real": {
                    "support_indices": [2],
                    "query_indices": [0, 1],
                    "paths": dataset.paths,
                },
                "ADM": {
                    "support_indices": [1],
                    "query_indices": [0, 2],
                    "paths": dataset.paths,
                },
            }
            rows = sampling_manifest_rows("ADM", 42, 1, sampling)
            for row in rows:
                self.assertEqual(
                    row["filepath"], dataset.paths[int(row["dataset_index"])]
                )


class ZeroShotProtocolTest(unittest.TestCase):
    def test_metadata_excludes_held_out_class(self):
        classes = zero_shot_metadata_classes("ADM")
        self.assertEqual(classes[0], "real")
        self.assertEqual(len(classes), 6)
        self.assertNotIn("ADM", classes)
        self.assertEqual(
            set(classes[1:]), {"BigGAN", "glide", "Midjourney", "SD", "VQDM"}
        )

    def test_query_uses_all_val_indices_without_balancing(self):
        real, fake = build_zero_shot_query_indices(9, 5)
        self.assertEqual(real, list(range(9)))
        self.assertEqual(fake, list(range(5)))


if __name__ == "__main__":
    unittest.main()
