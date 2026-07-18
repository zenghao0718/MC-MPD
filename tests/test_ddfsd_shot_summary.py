import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest

from util.ddfsd_main_protocol import FAKE_CLASSES, FORMAL_SHOTS
from util.ddfsd_main_protocol_validation import FORMAL_PROTOCOL, config_filename


FIELDS = [
    "checkpoint_model_mode",
    "model_mode",
    "branch_mode",
    "exclude_class",
    "seed",
    "shot",
    "support_shot",
    "ckpt_step",
    "acc",
    "real_acc",
    "fake_acc",
    "balanced_acc",
    "ap",
    "auc",
    "num_real_support",
    "num_fake_support",
    "num_real_query",
    "num_fake_query",
    "alpha_mean",
    "alpha_min",
    "alpha_max",
    "freq_stats_path",
    "ckpt_path",
]


def write_rows(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def fixture_paths(root, class_name):
    return (
        os.path.join(root, "resources", f"exclude_{class_name}", "model.pth"),
        os.path.join(root, "stats", f"exclude_{class_name}", "freq_stats.pt"),
    )


def fixture_row(root, class_name, shot):
    ckpt_path, freq_stats_path = fixture_paths(root, class_name)
    return {
        "checkpoint_model_mode": "dual",
        "model_mode": "dual",
        "branch_mode": "dual",
        "exclude_class": class_name,
        "seed": 42,
        "shot": shot if shot == 0 else "",
        "support_shot": shot if shot > 0 else "",
        "ckpt_step": 15000,
        "acc": 0.8,
        "real_acc": 0.75,
        "fake_acc": 0.85,
        "balanced_acc": 0.8,
        "ap": 0.9,
        "auc": 0.88,
        "num_real_support": shot,
        "num_fake_support": shot,
        "num_real_query": 100 - shot,
        "num_fake_query": 80 - shot,
        "alpha_mean": 0.5,
        "alpha_min": 0.4,
        "alpha_max": 0.6,
        "freq_stats_path": freq_stats_path,
        "ckpt_path": ckpt_path,
    }


def fixture_config(root, class_name, shot):
    ckpt_path, freq_stats_path = fixture_paths(root, class_name)
    return {
        "protocol": FORMAL_PROTOCOL,
        "git_commit": "same-commit",
        "exclude_class": class_name,
        "shot": shot,
        "seeds": [42],
        "data_root": os.path.join(root, "data"),
        "ckpt_path": ckpt_path,
        "ckpt_step": 15000,
        "freq_stats_path": freq_stats_path,
        "checkpoint_model_mode": "dual",
        "model_mode": "dual",
        "branch_mode": "dual",
        "tau": 0.2,
        "tau_r": 0.1,
        "max_eval_query_per_class": 0,
        "zero_shot_metadata_per_class": 1024,
        "strict_formal_eval_images": True,
    }


def build_fixture_tree(directory, parity_mismatch=False, config_mutation=None):
    input_root = os.path.join(directory, "input")
    reference_root = os.path.join(directory, "reference")
    for class_name in FAKE_CLASSES:
        for shot in FORMAL_SHOTS:
            result_dir = os.path.join(
                input_root, f"exclude_{class_name}", f"shot_{shot}"
            )
            filename = (
                "ddfsd_zero_shot_per_seed.csv"
                if shot == 0
                else "ddfsd_eval_per_seed.csv"
            )
            write_rows(
                os.path.join(result_dir, filename),
                [fixture_row(directory, class_name, shot)],
            )
            config = fixture_config(directory, class_name, shot)
            if config_mutation and (class_name, shot) == config_mutation[:2]:
                config[config_mutation[2]] = config_mutation[3]
            with open(
                os.path.join(result_dir, config_filename(shot)),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(config, handle)
        reference = fixture_row(directory, class_name, 10)
        if parity_mismatch and class_name == "ADM":
            reference["acc"] = 0.81
        write_rows(
            os.path.join(
                reference_root,
                f"exclude_{class_name}",
                "formal_eval",
                "step_15000",
                "ddfsd_eval_per_seed.csv",
            ),
            [reference],
        )
    return input_root, reference_root


class ShotSummaryGateTest(unittest.TestCase):
    def test_parity_failure_prevents_formal_report(self):
        with tempfile.TemporaryDirectory() as directory:
            input_root, reference_root = build_fixture_tree(
                directory, parity_mismatch=True
            )
            output_dir = os.path.join(directory, "summary")

            completed = subprocess.run(
                [
                    sys.executable,
                    "tools/summarize_ddfsd_shot_ablation_main_protocol.py",
                    "--input_root",
                    input_root,
                    "--output_dir",
                    output_dir,
                    "--reference_main_root",
                    reference_root,
                    "--eval_seeds",
                    "42",
                ],
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("parity failed", completed.stderr.lower())
            self.assertTrue(
                os.path.isfile(os.path.join(output_dir, "ddfsd_10shot_parity.csv"))
            )
            self.assertFalse(
                os.path.exists(
                    os.path.join(
                        output_dir, "DDFSD_shot_ablation_main_protocol_report.md"
                    )
                )
            )

    def test_config_failure_prevents_parity_and_formal_report(self):
        with tempfile.TemporaryDirectory() as directory:
            input_root, reference_root = build_fixture_tree(
                directory,
                config_mutation=("ADM", 0, "protocol", "fixed-query exploratory"),
            )
            output_dir = os.path.join(directory, "summary")
            completed = subprocess.run(
                [
                    sys.executable,
                    "tools/summarize_ddfsd_shot_ablation_main_protocol.py",
                    "--input_root",
                    input_root,
                    "--output_dir",
                    output_dir,
                    "--reference_main_root",
                    reference_root,
                    "--eval_seeds",
                    "42",
                ],
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("configuration validation failed", completed.stderr.lower())
            self.assertFalse(os.path.exists(output_dir))


if __name__ == "__main__":
    unittest.main()
