import copy
import json
import os
import tempfile
import unittest

from util.ddfsd_main_protocol_validation import (
    FORMAL_PROTOCOL,
    config_filename,
    required_result_files,
    validate_aggregate_configs,
    validate_completed_result,
)


CLASSES = ("ADM", "BigGAN", "glide", "Midjourney", "SD", "VQDM")


def expected_config(root, class_name="ADM", shot=10):
    return {
        "protocol": FORMAL_PROTOCOL,
        "git_commit": "abc123",
        "exclude_class": class_name,
        "shot": shot,
        "seeds": [42, 101, 102, 103, 104],
        "data_root": os.path.join(root, "data"),
        "ckpt_path": os.path.join(root, f"exclude_{class_name}", "model.pth"),
        "ckpt_step": 15000,
        "freq_stats_path": os.path.join(root, f"exclude_{class_name}", "freq_stats.pt"),
        "checkpoint_model_mode": "dual",
        "model_mode": "dual",
        "branch_mode": "dual",
        "tau": 0.2,
        "tau_r": 0.1,
        "max_eval_query_per_class": 0,
        "zero_shot_metadata_per_class": 1024,
        "strict_formal_eval_images": True,
    }


def snapshot_tree(root):
    snapshot = {}
    for directory, _, files in os.walk(root):
        for filename in files:
            path = os.path.join(directory, filename)
            with open(path, "rb") as handle:
                snapshot[os.path.relpath(path, root)] = handle.read()
    return snapshot


class CompletionValidationTest(unittest.TestCase):
    def test_only_matching_complete_result_is_skippable(self):
        for shot in (0, 10):
            with self.subTest(shot=shot), tempfile.TemporaryDirectory() as root:
                output = os.path.join(root, "exclude_ADM", f"shot_{shot}")
                os.makedirs(output)
                expected = expected_config(root, shot=shot)
                for filename in required_result_files(shot):
                    path = os.path.join(output, filename)
                    with open(path, "w", encoding="utf-8") as handle:
                        handle.write("present\n")
                with open(
                    os.path.join(output, config_filename(shot)),
                    "w",
                    encoding="utf-8",
                ) as handle:
                    json.dump(expected, handle)
                before = snapshot_tree(output)
                self.assertEqual(validate_completed_result(output, expected), [])
                self.assertEqual(snapshot_tree(output), before)

    def test_request_differences_are_all_rejected_without_writes(self):
        mutations = {
            "seeds": [42],
            "shot": 5,
            "ckpt_path": "different/checkpoint.pth",
            "tau": 0.3,
            "protocol": "fixed-query exploratory protocol",
            "max_eval_query_per_class": 128,
        }
        for field, value in mutations.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as root:
                output = os.path.join(root, "exclude_ADM", "shot_10")
                os.makedirs(output)
                expected = expected_config(root)
                actual = copy.deepcopy(expected)
                actual[field] = value
                for filename in required_result_files(10):
                    path = os.path.join(output, filename)
                    with open(path, "w", encoding="utf-8") as handle:
                        handle.write("present\n")
                with open(
                    os.path.join(output, config_filename(10)),
                    "w",
                    encoding="utf-8",
                ) as handle:
                    json.dump(actual, handle)
                before = snapshot_tree(output)
                errors = validate_completed_result(output, expected)
                self.assertTrue(any(error.startswith(f"{field}:") for error in errors))
                self.assertEqual(snapshot_tree(output), before)

    def test_all_config_differences_are_reported_together(self):
        with tempfile.TemporaryDirectory() as root:
            output = os.path.join(root, "exclude_ADM", "shot_10")
            os.makedirs(output)
            expected = expected_config(root)
            actual = copy.deepcopy(expected)
            mutations = {
                "seeds": [42],
                "ckpt_path": "different/model.pth",
                "tau": 0.4,
                "protocol": "fixed-query",
                "max_eval_query_per_class": 16,
            }
            actual.update(mutations)
            for filename in required_result_files(10):
                with open(
                    os.path.join(output, filename), "w", encoding="utf-8"
                ) as handle:
                    handle.write("present\n")
            with open(
                os.path.join(output, config_filename(10)), "w", encoding="utf-8"
            ) as handle:
                json.dump(actual, handle)
            errors = validate_completed_result(output, expected)
            for field in mutations:
                self.assertTrue(any(error.startswith(f"{field}:") for error in errors))


def aggregate_records(root, shots=(0, 5)):
    records = []
    for class_name in CLASSES:
        for shot in shots:
            result_dir = os.path.join(root, f"exclude_{class_name}", f"shot_{shot}")
            config = expected_config(root, class_name, shot)
            config_path = os.path.join(result_dir, config_filename(shot))
            records.append(
                {
                    "exclude_class": class_name,
                    "shot": shot,
                    "result_dir": result_dir,
                    "config_path": config_path,
                    "config": config,
                }
            )
    return records


class AggregateConfigurationValidationTest(unittest.TestCase):
    def test_consistent_matrix_passes(self):
        with tempfile.TemporaryDirectory() as root:
            records = aggregate_records(root)
            self.assertEqual(
                validate_aggregate_configs(
                    records, CLASSES, (0, 5), [42, 101, 102, 103, 104], 15000
                ),
                [],
            )

    def test_git_tau_protocol_and_class_shot_mismatches_fail(self):
        mutations = (
            ("git_commit", "different"),
            ("tau_r", 0.2),
            ("protocol", "fixed-query exploratory protocol"),
            ("exclude_class", "BigGAN"),
            ("shot", 30),
        )
        for field, value in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as root:
                records = aggregate_records(root)
                target = records[0]
                if field in {"exclude_class", "shot"}:
                    target[field] = value
                    target["config"][field] = value
                else:
                    target["config"][field] = value
                errors = validate_aggregate_configs(
                    records, CLASSES, (0, 5), [42, 101, 102, 103, 104], 15000
                )
                self.assertTrue(errors)
                self.assertTrue(any(field in error for error in errors))

    def test_cross_class_checkpoint_or_freq_stats_reuse_fails(self):
        for field in ("ckpt_path", "freq_stats_path"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as root:
                records = aggregate_records(root)
                biggan_value = next(
                    record["config"][field]
                    for record in records
                    if record["exclude_class"] == "BigGAN"
                )
                for record in records:
                    if record["exclude_class"] == "ADM":
                        record["config"][field] = biggan_value
                errors = validate_aggregate_configs(
                    records, CLASSES, (0, 5), [42, 101, 102, 103, 104], 15000
                )
                self.assertTrue(any("shared by" in error for error in errors))
                self.assertTrue(any("exclude_ADM" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
