import copy
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from util.ddfsd_main_protocol_validation import (
    FORMAL_PROTOCOL,
    config_filename,
    resolve_reference_csv,
    validate_aggregate_configs,
    validate_completed_result,
)


CLASSES = ("ADM", "BigGAN", "glide", "Midjourney", "SD", "VQDM")


def expected_config(
    root, class_name="ADM", shot=10, seeds=None, metadata_count=1024
):
    seeds = list(seeds or [42, 101, 102, 103, 104])
    config = {
        "protocol": FORMAL_PROTOCOL,
        "git_commit": "abc123",
        "exclude_class": class_name,
        "shot": shot,
        "seeds": seeds,
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
        "zero_shot_metadata_per_class": metadata_count,
        "strict_formal_eval_images": True,
    }
    if shot == 0:
        config.update(
            {
                "held_out_class": class_name,
                "metadata_classes": [
                    "real",
                    *[name for name in CLASSES if name != class_name],
                ],
                "metadata_samples_per_class": metadata_count,
                "metadata_sampling_mode": "deterministic_lazy_strict_until_full",
                "full_train_audit": False,
            }
        )
    return config


def write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rewrite_csv(path, rows):
    with open(path, newline="", encoding="utf-8") as handle:
        fields = csv.DictReader(handle).fieldnames
    write_csv(path, fields, rows)


def write_complete_result(output, config):
    os.makedirs(output, exist_ok=True)
    shot = int(config["shot"])
    class_name = config["exclude_class"]
    seeds = [int(seed) for seed in config["seeds"]]
    metrics = {
        "acc": 0.8,
        "real_acc": 0.75,
        "fake_acc": 0.85,
        "balanced_acc": 0.8,
        "ap": 0.9,
        "auc": 0.88,
    }
    per_rows = []
    for seed in seeds:
        per_rows.append(
            {
                "checkpoint_model_mode": config["checkpoint_model_mode"],
                "model_mode": config["model_mode"],
                "branch_mode": config["branch_mode"],
                "exclude_class": class_name,
                "seed": seed,
                "shot": shot if shot == 0 else "",
                "support_shot": shot if shot > 0 else "",
                "ckpt_step": config["ckpt_step"],
                **metrics,
                "num_real_support": 0 if shot == 0 else shot,
                "num_fake_support": 0 if shot == 0 else shot,
                "num_real_query": 2,
                "num_fake_query": 2,
                "ckpt_path": config["ckpt_path"],
                "freq_stats_path": config["freq_stats_path"],
            }
        )
    per_fields = [
        "checkpoint_model_mode",
        "model_mode",
        "branch_mode",
        "exclude_class",
        "seed",
        "shot",
        "support_shot",
        "ckpt_step",
        *metrics,
        "num_real_support",
        "num_fake_support",
        "num_real_query",
        "num_fake_query",
        "ckpt_path",
        "freq_stats_path",
    ]
    per_name = "ddfsd_zero_shot_per_seed.csv" if shot == 0 else "ddfsd_eval_per_seed.csv"
    write_csv(os.path.join(output, per_name), per_fields, per_rows)

    summary = {
        "exclude_class": class_name,
        "shot": shot if shot == 0 else "",
        "support_shot": shot if shot > 0 else "",
        "ckpt_step": config["ckpt_step"],
        "eval_seeds": ",".join(map(str, seeds)),
    }
    for field, value in metrics.items():
        summary[f"{field}_mean"] = value
        summary[f"{field}_std"] = 0.0
    summary_fields = [
        "exclude_class",
        "shot",
        "support_shot",
        "ckpt_step",
        "eval_seeds",
        *[
            f"{field}_{suffix}"
            for field in metrics
            for suffix in ("mean", "std")
        ],
    ]
    summary_name = "ddfsd_zero_shot_summary.csv" if shot == 0 else "ddfsd_eval_summary.csv"
    write_csv(os.path.join(output, summary_name), summary_fields, [summary])

    audit_fields = [
        "data_class",
        "split",
        "dataset_index",
        "filepath",
        "error_type",
        "error",
    ]
    write_csv(os.path.join(output, "formal_val_invalid_images.csv"), audit_fields, [])
    if shot == 0:
        invalid_fields = [
            "held_out_class",
            "metadata_class",
            "seed",
            "image_path",
            "error_type",
            "error_message",
        ]
        write_csv(os.path.join(output, "zero_shot_invalid_images.csv"), invalid_fields, [])
        manifest_rows = []
        for metadata_class in config["metadata_classes"]:
            for seed in seeds:
                for rank in range(1, int(config["metadata_samples_per_class"]) + 1):
                    manifest_rows.append(
                        {
                            "held_out_class": class_name,
                            "metadata_class": metadata_class,
                            "seed": seed,
                            "sample_rank": rank,
                            "dataset_index": rank - 1,
                            "image_path": os.path.join(
                                str(config["data_root"]), metadata_class, f"{rank}.png"
                            ),
                        }
                    )
        write_csv(
            os.path.join(output, "zero_shot_metadata_manifest.csv"),
            [
                "held_out_class",
                "metadata_class",
                "seed",
                "sample_rank",
                "dataset_index",
                "image_path",
            ],
            manifest_rows,
        )
    else:
        manifest_rows = []
        for seed in seeds:
            for data_class in ("real", class_name):
                for index in range(shot):
                    manifest_rows.append(
                        {
                            "exclude_class": class_name,
                            "seed": seed,
                            "shot": shot,
                            "data_class": data_class,
                            "split": "val",
                            "dataset_index": index,
                            "filepath": f"{data_class}_{index}.png",
                            "role": "support",
                            "support_rank": index + 1,
                        }
                    )
                for index in range(shot, shot + 2):
                    manifest_rows.append(
                        {
                            "exclude_class": class_name,
                            "seed": seed,
                            "shot": shot,
                            "data_class": data_class,
                            "split": "val",
                            "dataset_index": index,
                            "filepath": f"{data_class}_{index}.png",
                            "role": "query",
                            "support_rank": "",
                        }
                    )
        write_csv(
            os.path.join(output, "support_query_manifest.csv"),
            [
                "exclude_class",
                "seed",
                "shot",
                "data_class",
                "split",
                "dataset_index",
                "filepath",
                "role",
                "support_rank",
            ],
            manifest_rows,
        )
    with open(os.path.join(output, "eval.log"), "w", encoding="utf-8") as handle:
        handle.write("evaluation complete\n")
    with open(
        os.path.join(output, config_filename(shot)), "w", encoding="utf-8"
    ) as handle:
        json.dump(config, handle)


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
                expected = expected_config(
                    root, shot=shot, seeds=[42, 101], metadata_count=2
                )
                write_complete_result(output, expected)
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
                expected = expected_config(root)
                actual = copy.deepcopy(expected)
                actual[field] = value
                write_complete_result(output, actual)
                before = snapshot_tree(output)
                errors = validate_completed_result(output, expected)
                self.assertTrue(any(error.startswith(f"{field}:") for error in errors))
                self.assertEqual(snapshot_tree(output), before)

    def test_all_config_differences_are_reported_together(self):
        with tempfile.TemporaryDirectory() as root:
            output = os.path.join(root, "exclude_ADM", "shot_10")
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
            write_complete_result(output, actual)
            errors = validate_completed_result(output, expected)
            for field in mutations:
                self.assertTrue(any(error.startswith(f"{field}:") for error in errors))

    def test_old_full_audit_zero_shot_result_is_not_reused(self):
        with tempfile.TemporaryDirectory() as root:
            output = os.path.join(root, "exclude_ADM", "shot_0")
            expected = expected_config(
                root, shot=0, seeds=[42, 101], metadata_count=2
            )
            actual = copy.deepcopy(expected)
            actual["metadata_sampling_mode"] = "full_train_strict_audit_then_sample"
            actual["full_train_audit"] = True
            write_complete_result(output, actual)
            errors = validate_completed_result(output, expected)
            self.assertTrue(
                any(error.startswith("metadata_sampling_mode:") for error in errors)
            )
            self.assertTrue(
                any(error.startswith("full_train_audit:") for error in errors)
            )

    def test_per_seed_missing_or_duplicate_seed_is_rejected(self):
        for mutation in ("missing", "duplicate"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as root:
                config = expected_config(root, seeds=[42, 101])
                output = os.path.join(root, "exclude_ADM", "shot_10")
                write_complete_result(output, config)
                path = os.path.join(output, "ddfsd_eval_per_seed.csv")
                rows = read_csv(path)
                if mutation == "missing":
                    rows = rows[:1]
                else:
                    rows.append(dict(rows[0]))
                rewrite_csv(path, rows)
                errors = validate_completed_result(output, config)
                self.assertTrue(any("seed" in error.lower() for error in errors))

    def test_non_finite_metric_is_rejected(self):
        for value in ("nan", "inf"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as root:
                config = expected_config(root, seeds=[42])
                output = os.path.join(root, "exclude_ADM", "shot_10")
                write_complete_result(output, config)
                path = os.path.join(output, "ddfsd_eval_per_seed.csv")
                rows = read_csv(path)
                rows[0]["acc"] = value
                rewrite_csv(path, rows)
                errors = validate_completed_result(output, config)
                self.assertTrue(any("not finite" in error for error in errors))

    def test_summary_empty_or_multiple_rows_is_rejected(self):
        for mutation in ("empty", "multiple"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as root:
                config = expected_config(root, seeds=[42])
                output = os.path.join(root, "exclude_ADM", "shot_10")
                write_complete_result(output, config)
                path = os.path.join(output, "ddfsd_eval_summary.csv")
                rows = read_csv(path)
                rewrite_csv(path, [] if mutation == "empty" else rows + rows)
                errors = validate_completed_result(output, config)
                self.assertTrue(any("exactly one row" in error for error in errors))

    def test_zero_shot_manifest_missing_row_or_wrong_group_count_is_rejected(self):
        for mutation in ("missing", "wrong_group"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as root:
                config = expected_config(
                    root, shot=0, seeds=[42], metadata_count=2
                )
                output = os.path.join(root, "exclude_ADM", "shot_0")
                write_complete_result(output, config)
                path = os.path.join(output, "zero_shot_metadata_manifest.csv")
                rows = read_csv(path)
                if mutation == "missing":
                    rows.pop()
                else:
                    rows[0]["metadata_class"] = config["metadata_classes"][1]
                rewrite_csv(path, rows)
                errors = validate_completed_result(output, config)
                self.assertTrue(
                    any("group" in error.lower() or "rows" in error for error in errors)
                )

    def test_zero_shot_manifest_duplicate_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            config = expected_config(root, shot=0, seeds=[42], metadata_count=2)
            output = os.path.join(root, "exclude_ADM", "shot_0")
            write_complete_result(output, config)
            path = os.path.join(output, "zero_shot_metadata_manifest.csv")
            rows = read_csv(path)
            rows[1]["image_path"] = rows[0]["image_path"]
            rewrite_csv(path, rows)
            errors = validate_completed_result(output, config)
            self.assertTrue(any("duplicate image_path" in error for error in errors))

    def test_zero_shot_manifest_rejects_held_out_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            config = expected_config(root, shot=0, seeds=[42], metadata_count=2)
            output = os.path.join(root, "exclude_ADM", "shot_0")
            write_complete_result(output, config)
            config["metadata_classes"][1] = "ADM"
            with open(
                os.path.join(output, config_filename(0)), "w", encoding="utf-8"
            ) as handle:
                json.dump(config, handle)
            errors = validate_completed_result(output, config)
            self.assertTrue(any("held-out class ADM" in error for error in errors))

    def test_positive_support_count_error_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            config = expected_config(root, seeds=[42])
            output = os.path.join(root, "exclude_ADM", "shot_10")
            write_complete_result(output, config)
            path = os.path.join(output, "ddfsd_eval_per_seed.csv")
            rows = read_csv(path)
            rows[0]["num_real_support"] = "9"
            rewrite_csv(path, rows)
            errors = validate_completed_result(output, config)
            self.assertTrue(any("num_real_support=9" in error for error in errors))

    def test_positive_manifest_support_query_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            config = expected_config(root, seeds=[42])
            output = os.path.join(root, "exclude_ADM", "shot_10")
            write_complete_result(output, config)
            path = os.path.join(output, "support_query_manifest.csv")
            rows = read_csv(path)
            query = next(
                row
                for row in rows
                if row["data_class"] == "real" and row["role"] == "query"
            )
            query["dataset_index"] = "0"
            rewrite_csv(path, rows)
            errors = validate_completed_result(output, config)
            self.assertTrue(any("support/query overlap" in error for error in errors))

    def test_positive_manifest_query_count_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            config = expected_config(root, seeds=[42])
            output = os.path.join(root, "exclude_ADM", "shot_10")
            write_complete_result(output, config)
            path = os.path.join(output, "support_query_manifest.csv")
            rows = read_csv(path)
            removed = False
            kept = []
            for row in rows:
                if not removed and row["data_class"] == "real" and row["role"] == "query":
                    removed = True
                    continue
                kept.append(row)
            rewrite_csv(path, kept)
            errors = validate_completed_result(output, config)
            self.assertTrue(any("query count" in error for error in errors))


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
            ("metadata_sampling_mode", "full_train_strict_audit_then_sample"),
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


class ReferenceCsvResolutionTest(unittest.TestCase):
    @staticmethod
    def _touch(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("seed,shot,ckpt_step\n")

    def test_explicit_reference_has_priority(self):
        with tempfile.TemporaryDirectory() as root:
            explicit = os.path.join(root, "explicit.csv")
            common = os.path.join(root, "formal_eval", "reference.csv")
            compatibility = os.path.join(
                root, "formal_eval", "step_15000", "reference.csv"
            )
            for path in (explicit, common, compatibility):
                self._touch(path)
            self.assertEqual(
                resolve_reference_csv(explicit, [common, compatibility]),
                os.path.normcase(os.path.realpath(explicit)),
            )

    def test_unique_default_candidate_is_selected(self):
        with tempfile.TemporaryDirectory() as root:
            common = os.path.join(root, "formal_eval", "reference.csv")
            compatibility = os.path.join(
                root, "formal_eval", "step_15000", "reference.csv"
            )
            self._touch(common)
            self.assertEqual(
                resolve_reference_csv(None, [common, compatibility]),
                os.path.normcase(os.path.realpath(common)),
            )

    def test_no_default_candidate_fails_and_lists_candidates(self):
        with tempfile.TemporaryDirectory() as root:
            candidates = [
                os.path.join(root, "common.csv"),
                os.path.join(root, "step.csv"),
            ]
            with self.assertRaisesRegex(FileNotFoundError, "common.csv"):
                resolve_reference_csv(None, candidates)

    def test_two_default_candidates_are_ambiguous(self):
        with tempfile.TemporaryDirectory() as root:
            candidates = [
                os.path.join(root, "common.csv"),
                os.path.join(root, "step.csv"),
            ]
            for path in candidates:
                self._touch(path)
            with self.assertRaisesRegex(ValueError, "Set REFERENCE_CSV explicitly"):
                resolve_reference_csv(None, candidates)

    def test_resolver_cli_works_outside_repository_cwd(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tool = os.path.join(repo_root, "tools", "resolve_ddfsd_reference_csv.py")
        with tempfile.TemporaryDirectory() as root:
            reference = os.path.join(root, "reference.csv")
            self._touch(reference)
            completed = subprocess.run(
                [sys.executable, tool, "--candidate", reference],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                completed.stdout.strip(), os.path.normcase(os.path.realpath(reference))
            )


class ShellRunnerContractTest(unittest.TestCase):
    @staticmethod
    def _script(name):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(
            os.path.join(repo_root, "scripts", name), encoding="utf-8"
        ) as handle:
            return handle.read()

    def test_formal_runner_uses_repo_root_and_optional_reference_template(self):
        script = self._script("run_ddfsd_shot_ablation_main_protocol.sh")
        self.assertIn('SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"', script)
        self.assertIn('cd "${REPO_ROOT}"', script)
        self.assertIn('if [[ -n "${REFERENCE_CSV_TEMPLATE+x}" ]]', script)
        self.assertNotIn("REFERENCE_CSV_TEMPLATE=${REFERENCE_CSV_TEMPLATE:-", script)
        self.assertIn('"${REFERENCE_TEMPLATE_ARGS[@]}"', script)

    def test_metadata_1024_gate_precedes_data_access(self):
        script = self._script("run_ddfsd_shot_ablation_main_protocol.sh")
        gate = script.index(
            "Formal zero-shot evaluation requires ZERO_SHOT_METADATA_PER_CLASS=1024."
        )
        data_check = script.index('[[ -d "${DATA_ROOT}" ]]')
        self.assertLess(gate, data_check)

    def test_adm_parity_is_recomputed_via_atomic_temporary_file(self):
        script = self._script("run_ddfsd_adm_10shot_parity_precheck.sh")
        completion = script.index("tools/check_ddfsd_shot_completion.py")
        temporary = script.index('PARITY_TMP="${PARITY_CSV}.tmp.$$"')
        self.assertLess(completion, temporary)
        self.assertIn('--output_csv "${PARITY_TMP}"', script)
        self.assertIn('mv -f -- "${PARITY_TMP}" "${PARITY_CSV}"', script)
        self.assertIn("trap cleanup_parity_tmp EXIT", script)
        self.assertNotIn("Refusing to overwrite existing parity CSV", script)


class AdmParityRerunIntegrationTest(unittest.TestCase):
    @staticmethod
    def _bash_path():
        candidates = (
            shutil.which("bash"),
            r"D:\Git\bin\bash.exe",
            r"C:\Program Files\Git\bin\bash.exe",
        )
        return next((path for path in candidates if path and os.path.isfile(path)), None)

    def test_first_run_rerun_failure_cleanup_and_skip_zero(self):
        bash = self._bash_path()
        if bash is None:
            self.skipTest("Bash is unavailable")
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script = os.path.join(
            repo_root, "scripts", "run_ddfsd_adm_10shot_parity_precheck.sh"
        ).replace("\\", "/")
        with tempfile.TemporaryDirectory() as root:
            data_root = os.path.join(root, "data")
            os.makedirs(data_root)
            ckpt_path = os.path.join(root, "exclude_ADM", "model.pth")
            freq_path = os.path.join(root, "exclude_ADM", "freq_stats.pt")
            os.makedirs(os.path.dirname(ckpt_path))
            for path in (ckpt_path, freq_path):
                with open(path, "wb") as handle:
                    handle.write(b"fixture")
            output = os.path.join(root, "output")
            config = expected_config(root, seeds=[42, 101, 102, 103, 104])
            config.update(
                {
                    "git_commit": "test-commit",
                    "data_root": data_root,
                    "ckpt_path": ckpt_path,
                    "freq_stats_path": freq_path,
                }
            )
            write_complete_result(output, config)
            reference = os.path.join(root, "reference.csv")
            shutil.copyfile(os.path.join(output, "ddfsd_eval_per_seed.csv"), reference)
            parity = os.path.join(output, "ddfsd_10shot_parity.csv")
            environment = os.environ.copy()
            environment.update(
                {
                    "PATH": "/mingw64/bin:/usr/bin:/bin",
                    "PYTHON_BIN": sys.executable.replace("\\", "/"),
                    "DATA_ROOT": data_root.replace("\\", "/"),
                    "CKPT_PATH": ckpt_path.replace("\\", "/"),
                    "FREQ_STATS_PATH": freq_path.replace("\\", "/"),
                    "OUTPUT_DIR": output.replace("\\", "/"),
                    "REFERENCE_CSV": reference.replace("\\", "/"),
                    "PARITY_CSV": parity.replace("\\", "/"),
                    "GIT_COMMIT": "test-commit",
                    "SKIP_COMPLETED": "1",
                }
            )

            def run():
                return subprocess.run(
                    [bash, "-c", f'/usr/bin/bash "{script}"'],
                    cwd=root,
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                )

            first = run()
            self.assertEqual(first.returncode, 0, first.stderr)
            with open(parity, "rb") as handle:
                valid_parity = handle.read()
            rerun = run()
            self.assertEqual(rerun.returncode, 0, rerun.stderr)
            with open(parity, "rb") as handle:
                self.assertEqual(handle.read(), valid_parity)

            reference_rows = read_csv(reference)
            reference_rows[0]["acc"] = "0.1"
            rewrite_csv(reference, reference_rows)
            failed = run()
            self.assertNotEqual(failed.returncode, 0)
            with open(parity, "rb") as handle:
                self.assertEqual(handle.read(), valid_parity)
            self.assertEqual(glob.glob(parity + ".tmp.*"), [])

            environment["SKIP_COMPLETED"] = "0"
            refused = run()
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("Refusing to overwrite complete", refused.stderr)
            with open(parity, "rb") as handle:
                self.assertEqual(handle.read(), valid_parity)


if __name__ == "__main__":
    unittest.main()
