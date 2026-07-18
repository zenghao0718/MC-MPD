import csv
import os
import subprocess
import sys
import tempfile
import unittest

from util.ddfsd_main_protocol import FAKE_CLASSES, FORMAL_SHOTS


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
]


def write_rows(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def fixture_row(class_name, shot):
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
    }


class ShotSummaryGateTest(unittest.TestCase):
    def test_parity_failure_prevents_formal_report(self):
        with tempfile.TemporaryDirectory() as directory:
            input_root = os.path.join(directory, "input")
            reference_root = os.path.join(directory, "reference")
            output_dir = os.path.join(directory, "summary")
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
                        [fixture_row(class_name, shot)],
                    )
                reference = fixture_row(class_name, 10)
                if class_name == "ADM":
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


if __name__ == "__main__":
    unittest.main()
