"""Validate and summarize the formal DDFSD main-protocol shot ablation."""

import argparse
import csv
import math
import os
import statistics
import sys
from typing import Iterable, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.check_ddfsd_10shot_parity import compare_parity_rows, read_csv
from util.ddfsd_main_protocol import FAKE_CLASSES, FORMAL_SEEDS, FORMAL_SHOTS
from util.ddfsd_main_protocol_validation import (
    load_result_configs,
    normalize_path,
    resolve_reference_csv,
    validate_aggregate_configs,
)


METRICS = ("acc", "real_acc", "fake_acc", "balanced_acc", "ap", "auc")


def write_csv(path: str, rows: Iterable[dict], fields: List[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values):
    values = [float(value) for value in values]
    return statistics.mean(values), (
        statistics.pstdev(values) if len(values) > 1 else 0.0
    )


def parse_int_list(value: str) -> List[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Expected a non-empty unique integer list, got {value!r}.")
    return values


def render(template: str, **values) -> str:
    return os.path.abspath(template.format(**values))


def resolve_class_reference_csvs(
    reference_main_root,
    classes,
    ckpt_step,
    explicit_template=None,
    common_template=(
        "{reference_main_root}/exclude_{class_name}/formal_eval/"
        "ddfsd_eval_per_seed.csv"
    ),
    step_template=(
        "{reference_main_root}/exclude_{class_name}/formal_eval/step_{step}/"
        "ddfsd_eval_per_seed.csv"
    ),
):
    resolved = {}
    values = {"reference_main_root": os.path.abspath(reference_main_root)}
    for class_name in classes:
        render_values = {
            **values,
            "class_name": class_name,
            "class": class_name,
            "step": ckpt_step,
        }
        explicit_path = (
            render(explicit_template, **render_values)
            if explicit_template is not None
            else None
        )
        candidates = [
            render(common_template, **render_values),
            render(step_template, **render_values),
        ]
        resolved[class_name] = resolve_reference_csv(explicit_path, candidates)
    return resolved


def _result_csv(result_dir: str, shot: int) -> str:
    filename = (
        "ddfsd_zero_shot_per_seed.csv" if shot == 0 else "ddfsd_eval_per_seed.csv"
    )
    return os.path.join(result_dir, filename)


def load_formal_rows(
    input_root,
    shot_dir_template,
    classes,
    shots,
    seeds,
    ckpt_step,
    configs_by_key=None,
):
    all_rows = []
    by_class_shot = {}
    expected_seed_set = set(seeds)
    for class_name in classes:
        for shot in shots:
            result_dir = render(
                shot_dir_template,
                input_root=input_root,
                class_name=class_name,
                **{"class": class_name},
                shot=shot,
                step=ckpt_step,
            )
            path = _result_csv(result_dir, shot)
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Missing formal shot result for {class_name} shot={shot}: {path}"
                )
            rows = read_csv(path)
            config = (configs_by_key or {}).get((class_name, shot))
            seen = set()
            for raw in rows:
                row = dict(raw)
                if row.get("exclude_class") != class_name:
                    raise ValueError(
                        f"exclude_class mismatch in {path}: {row.get('exclude_class')!r}"
                    )
                row_shot = int(row.get("shot") or row.get("support_shot", -1))
                if row_shot != shot:
                    raise ValueError(
                        f"Shot mismatch in {path}: expected {shot}, got {row_shot}"
                    )
                seed = int(row["seed"])
                if seed in seen:
                    raise ValueError(f"Duplicate seed {seed} in {path}")
                seen.add(seed)
                if int(row["ckpt_step"]) != ckpt_step:
                    raise ValueError(
                        f"Checkpoint step mismatch in {path}: expected {ckpt_step}, got {row['ckpt_step']}"
                    )
                if config is not None:
                    row_config_fields = (
                        "checkpoint_model_mode",
                        "model_mode",
                        "branch_mode",
                    )
                    for field in row_config_fields:
                        if row.get(field) != str(config.get(field, "")):
                            raise ValueError(
                                f"{field} row/config mismatch in {path} seed={seed}: "
                                f"row={row.get(field)!r}, config={config.get(field)!r}"
                            )
                    for field in ("ckpt_path", "freq_stats_path"):
                        if normalize_path(row.get(field, "")) != normalize_path(
                            config.get(field, "")
                        ):
                            raise ValueError(
                                f"{field} row/config mismatch in {path} seed={seed}: "
                                f"row={row.get(field)!r}, config={config.get(field)!r}"
                            )
                for metric in METRICS:
                    value = float(row[metric])
                    if not math.isfinite(value):
                        raise ValueError(f"Non-finite {metric} in {path} seed={seed}")
                row["shot"] = shot
                all_rows.append(row)
            if seen != expected_seed_set:
                raise ValueError(
                    f"Seed mismatch in {path}: missing={sorted(expected_seed_set - seen)}, "
                    f"unexpected={sorted(seen - expected_seed_set)}"
                )
            by_class_shot[(class_name, shot)] = rows
    return all_rows, by_class_shot


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize formal DDFSD shot ablation")
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--reference_main_root", required=True)
    parser.add_argument("--ckpt_step", type=int, default=15000)
    parser.add_argument("--classes", default=",".join(FAKE_CLASSES))
    parser.add_argument("--shots", default=",".join(map(str, FORMAL_SHOTS)))
    parser.add_argument("--eval_seeds", default=",".join(map(str, FORMAL_SEEDS)))
    parser.add_argument(
        "--shot_dir_template",
        default="{input_root}/exclude_{class_name}/shot_{shot}",
    )
    parser.add_argument(
        "--reference_csv_template",
        default=None,
    )
    parser.add_argument(
        "--reference_common_csv_template",
        default=(
            "{reference_main_root}/exclude_{class_name}/formal_eval/"
            "ddfsd_eval_per_seed.csv"
        ),
    )
    parser.add_argument(
        "--reference_step_csv_template",
        default=(
            "{reference_main_root}/exclude_{class_name}/formal_eval/step_{step}/"
            "ddfsd_eval_per_seed.csv"
        ),
    )
    parser.add_argument("--parity_tolerance", type=float, default=1e-6)
    return parser.parse_args()


def main():
    args = parse_args()
    classes = [item.strip() for item in args.classes.split(",") if item.strip()]
    shots = parse_int_list(args.shots)
    seeds = parse_int_list(args.eval_seeds)
    if classes != list(FAKE_CLASSES):
        raise ValueError(
            f"Formal aggregation requires classes in order {list(FAKE_CLASSES)}, got {classes}"
        )
    if shots != list(FORMAL_SHOTS):
        raise ValueError(
            f"Formal aggregation requires shots {list(FORMAL_SHOTS)}, got {shots}"
        )
    output_dir = os.path.abspath(
        args.output_dir or os.path.join(args.input_root, "summary")
    )

    config_records = load_result_configs(
        os.path.abspath(args.input_root),
        args.shot_dir_template,
        classes,
        shots,
        args.ckpt_step,
    )
    config_errors = validate_aggregate_configs(
        config_records, classes, shots, seeds, args.ckpt_step
    )
    if config_errors:
        raise SystemExit(
            "Formal configuration validation failed; no report or figures were generated:\n- "
            + "\n- ".join(config_errors)
        )
    configs_by_key = {
        (record["exclude_class"], record["shot"]): record["config"]
        for record in config_records
    }

    all_rows, by_class_shot = load_formal_rows(
        os.path.abspath(args.input_root),
        args.shot_dir_template,
        classes,
        shots,
        seeds,
        args.ckpt_step,
        configs_by_key=configs_by_key,
    )

    try:
        references_by_class = resolve_class_reference_csvs(
            args.reference_main_root,
            classes,
            args.ckpt_step,
            explicit_template=args.reference_csv_template,
            common_template=args.reference_common_csv_template,
            step_template=args.reference_step_csv_template,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Reference CSV resolution failed: {exc}") from exc

    parity_rows = []
    parity_errors = []
    for class_name in classes:
        reference_path = references_by_class[class_name]
        rows, errors = compare_parity_rows(
            by_class_shot[(class_name, 10)],
            read_csv(reference_path),
            seeds,
            args.parity_tolerance,
        )
        parity_rows.extend(rows)
        parity_errors.extend(f"{class_name}: {error}" for error in errors)
    parity_path = os.path.join(output_dir, "ddfsd_10shot_parity.csv")
    write_csv(
        parity_path,
        parity_rows,
        [
            "exclude_class",
            "seed",
            "field",
            "new_value",
            "reference_value",
            "difference",
            "status",
        ],
    )
    if parity_errors:
        raise SystemExit(
            "10-shot parity failed; formal aggregation/report generation stopped:\n- "
            + "\n- ".join(parity_errors)
        )

    all_rows.sort(
        key=lambda row: (
            classes.index(row["exclude_class"]),
            int(row["seed"]),
            int(row["shot"]),
        )
    )
    preferred_fields = [
        "checkpoint_model_mode",
        "model_mode",
        "branch_mode",
        "exclude_class",
        "seed",
        "shot",
        "support_shot",
        "split",
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
        "zero_shot_metadata_per_class",
        "freq_stats_path",
        "ckpt_path",
    ]
    present = {field for row in all_rows for field in row}
    all_fields = [field for field in preferred_fields if field in present]
    all_fields.extend(sorted(present - set(all_fields)))
    write_csv(
        os.path.join(output_dir, "ddfsd_shot_ablation_per_seed_all.csv"),
        all_rows,
        all_fields,
    )

    class_summary = []
    for class_name in classes:
        for shot in shots:
            group = [
                row
                for row in all_rows
                if row["exclude_class"] == class_name and int(row["shot"]) == shot
            ]
            result = {"exclude_class": class_name, "shot": shot}
            for metric in METRICS:
                result[f"{metric}_mean"], result[f"{metric}_std"] = mean_std(
                    row[metric] for row in group
                )
            class_summary.append(result)
    summary_fields = ["exclude_class", "shot"] + [
        field for metric in METRICS for field in (f"{metric}_mean", f"{metric}_std")
    ]
    write_csv(
        os.path.join(output_dir, "ddfsd_shot_ablation_per_class_summary.csv"),
        class_summary,
        summary_fields,
    )

    macro_seed_rows = []
    for seed in seeds:
        for shot in shots:
            group = [
                row
                for row in all_rows
                if int(row["seed"]) == seed and int(row["shot"]) == shot
            ]
            if len(group) != len(classes):
                raise ValueError(
                    f"Macro group seed={seed} shot={shot} has {len(group)} classes"
                )
            macro_seed_rows.append(
                {
                    "seed": seed,
                    "shot": shot,
                    **{
                        metric: statistics.mean(float(row[metric]) for row in group)
                        for metric in METRICS
                    },
                }
            )
    macro_summary = []
    for shot in shots:
        group = [row for row in macro_seed_rows if row["shot"] == shot]
        result = {"shot": shot}
        for metric in METRICS:
            result[f"{metric}_mean"], result[f"{metric}_std"] = mean_std(
                row[metric] for row in group
            )
        macro_summary.append(result)
    macro_fields = ["shot"] + [
        field for metric in METRICS for field in (f"{metric}_mean", f"{metric}_std")
    ]
    write_csv(
        os.path.join(output_dir, "ddfsd_shot_ablation_macro_summary.csv"),
        macro_summary,
        macro_fields,
    )

    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    import matplotlib.pyplot as plt

    for metric in METRICS:
        plt.figure(figsize=(8, 5))
        plt.plot(
            shots,
            [row[f"{metric}_mean"] for row in macro_summary],
            marker="o",
            linewidth=2.5,
            label="Six-class Macro",
        )
        plt.axvline(10, color="gray", linestyle="--", label="10-shot parity anchor")
        plt.xticks(shots)
        plt.xlabel("Shot")
        plt.ylabel(metric.replace("_", " ").upper())
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"macro_{metric}_vs_shot.png"), dpi=180)
        plt.close()

    macro_lookup = {row["shot"]: row for row in macro_summary}
    lines = [
        "# DDFSD Main-Protocol Shot Ablation",
        "",
        "All values below come from the new formal CSVs. The 10-shot parity gate passed before this report was generated.",
        "",
        "| Shot | ACC | Real ACC | Fake ACC | Balanced ACC | AP | AUC |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for shot in shots:
        row = macro_lookup[shot]
        cells = [str(shot)]
        for metric in METRICS:
            cells.append(f"{row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_std']:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "Macro is computed by averaging the six held-out classes within each seed, then reporting mean and population standard deviation across seeds.",
            "",
        ]
    )
    with open(
        os.path.join(output_dir, "DDFSD_shot_ablation_main_protocol_report.md"),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write("\n".join(lines))


if __name__ == "__main__":
    main()
