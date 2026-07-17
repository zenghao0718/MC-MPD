"""Validate and aggregate six-class DDFSD multi-shot result directories."""

import argparse
import csv
import json
import os
import statistics

from util.ddfsd_multishot_logic import validate_config_consistency


CLASSES = ["ADM", "BigGAN", "glide", "Midjourney", "SD", "VQDM"]


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def mean_std(values):
    values = list(map(float, values))
    return statistics.mean(values), statistics.pstdev(values) if len(values) > 1 else 0.0


def fmt(mean, std):
    return f"{mean:.4f} ± {std:.4f}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True,
                        help="Parent containing exclude_<class>/multishot/multishot_per_seed.csv")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--result_dir_template", default="",
                        help="Optional class result directory template supporting {class} and {step}.")
    parser.add_argument("--ckpt_step", type=int, default=0)
    return parser.parse_args()


def locate_result_dir(input_root, class_name, template="", ckpt_step=0):
    if template:
        result = template.replace("{class}", class_name).replace("{step}", str(ckpt_step))
        return os.path.abspath(result)
    candidates = [os.path.join(input_root, f"exclude_{class_name}", "multishot"),
                  os.path.join(input_root, f"exclude_{class_name}")]
    matches = [path for path in candidates if os.path.isfile(os.path.join(path, "multishot_per_seed.csv"))]
    if not matches:
        raise FileNotFoundError(f"Missing multishot result directory for {class_name}; checked {candidates}")
    if len(matches) > 1:
        raise ValueError(f"Multiple result directories found for {class_name}: {matches}")
    return matches[0]


def validate_class_rows(class_name, rows, config, source_path):
    if not rows:
        raise ValueError(f"Empty per-seed CSV for {class_name}: {source_path}")
    if any(row.get("exclude_class") != class_name for row in rows):
        raise ValueError(f"exclude_class mismatch in {source_path}; expected {class_name}")
    actual = {(int(row["seed"]), int(row["shot"])) for row in rows}
    expected = {(int(seed), int(shot)) for seed in config["seeds"] for shot in config["shot_list"]}
    if actual != expected:
        raise ValueError(
            f"Seed/shot set mismatch for {class_name}: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    keys = [(int(row["seed"]), int(row["shot"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate seed/shot rows for {class_name}: {source_path}")
    for row in rows:
        checks = {
            "ckpt_step": int(row["ckpt_step"]),
            "checkpoint_model_mode": row["checkpoint_model_mode"],
            "model_mode": row["model_mode"],
            "branch_mode": row["branch_mode"],
            "ckpt_path": os.path.abspath(row["ckpt_path"]),
            "freq_stats_path": os.path.abspath(row["freq_stats_path"]) if row["freq_stats_path"] else "",
        }
        for field, value in checks.items():
            expected_value = config[field]
            if field in {"ckpt_path", "freq_stats_path"} and expected_value:
                expected_value = os.path.abspath(expected_value)
            if value != expected_value:
                raise ValueError(
                    f"{class_name} row/config mismatch for {field}: row={value!r}, config={expected_value!r}"
                )


def main():
    args = parse_args(); output = args.output_dir or args.input_root
    rows = []
    configs = {}
    for name in CLASSES:
        result_dir = locate_result_dir(
            args.input_root, name, template=args.result_dir_template, ckpt_step=args.ckpt_step
        )
        path = os.path.join(result_dir, "multishot_per_seed.csv")
        config_path = os.path.join(result_dir, "multishot_config.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing multishot_per_seed.csv for {name}: {path}")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Missing multishot_config.json for {name}: {config_path}")
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
        if config.get("exclude_class") != name:
            raise ValueError(
                f"Config exclude_class mismatch for directory {result_dir}: "
                f"expected {name}, got {config.get('exclude_class')!r}"
            )
        if not config.get("ckpt_path"):
            raise ValueError(f"Config has empty ckpt_path for {name}: {config_path}")
        if config.get("checkpoint_model_mode") != "rgb-only" and not config.get("freq_stats_path"):
            raise ValueError(f"Config has empty freq_stats_path for {name}: {config_path}")
        configs[name] = config
        class_rows = read_csv(path)
        validate_class_rows(name, class_rows, config, path)
        rows.extend(class_rows)
    validate_config_consistency(configs)
    checkpoint_paths = {os.path.abspath(config["ckpt_path"]) for config in configs.values()}
    if len(checkpoint_paths) != len(CLASSES):
        mapping = ", ".join(f"{name}={configs[name]['ckpt_path']}" for name in CLASSES)
        raise ValueError(f"Checkpoint paths must be class-specific and unique: {mapping}")
    frequency_paths = {
        os.path.abspath(config["freq_stats_path"])
        for config in configs.values() if config["checkpoint_model_mode"] != "rgb-only"
    }
    frequency_classes = sum(config["checkpoint_model_mode"] != "rgb-only" for config in configs.values())
    if len(frequency_paths) != frequency_classes:
        mapping = ", ".join(f"{name}={configs[name]['freq_stats_path']}" for name in CLASSES)
        raise ValueError(f"Frequency-stat paths must be class-specific and unique: {mapping}")
    keys = [(r["exclude_class"], int(r["seed"]), int(r["shot"])) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate exclude_class/seed/shot rows found.")
    seeds = list(configs[CLASSES[0]]["seeds"]); shots = list(configs[CLASSES[0]]["shot_list"])
    expected = {(c, s, k) for c in CLASSES for s in seeds for k in shots}
    missing = expected - set(keys)
    if missing:
        raise ValueError(f"Missing class/seed/shot results: {sorted(missing)[:10]}")
    rows.sort(key=lambda r: (CLASSES.index(r["exclude_class"]), int(r["seed"]), int(r["shot"])))
    all_fields = list(rows[0]); write_csv(os.path.join(output, "ddfsd_multishot_per_seed_all.csv"), rows, all_fields)

    class_summary = []
    for name in CLASSES:
        for shot in shots:
            group = [r for r in rows if r["exclude_class"] == name and int(r["shot"]) == shot]
            result = {"exclude_class": name, "shot": shot}
            for metric in ("acc", "ap", "auc"):
                result[f"{metric}_mean"], result[f"{metric}_std"] = mean_std(r[metric] for r in group)
            class_summary.append(result)
    write_csv(os.path.join(output, "ddfsd_multishot_per_class_summary.csv"), class_summary,
              ["exclude_class", "shot", "acc_mean", "acc_std", "ap_mean", "ap_std", "auc_mean", "auc_std"])

    macro_seed = []
    for seed in seeds:
        for shot in shots:
            group = [r for r in rows if int(r["seed"]) == seed and int(r["shot"]) == shot]
            macro_seed.append({"seed": seed, "shot": shot,
                               **{m: statistics.mean(float(r[m]) for r in group) for m in ("acc", "ap", "auc")}})
    macro = []
    for shot in shots:
        group = [r for r in macro_seed if r["shot"] == shot]; result = {"shot": shot}
        for metric in ("acc", "ap", "auc"):
            result[f"{metric}_mean"], result[f"{metric}_std"] = mean_std(r[metric] for r in group)
        macro.append(result)
    write_csv(os.path.join(output, "ddfsd_multishot_macro_summary.csv"), macro,
              ["shot", "acc_mean", "acc_std", "ap_mean", "ap_std", "auc_mean", "auc_std"])

    figures = os.path.join(output, "figures"); os.makedirs(figures, exist_ok=True)
    import matplotlib.pyplot as plt
    def plot_metric(metric, macro_only=False):
        plt.figure(figsize=(8, 5))
        if not macro_only:
            for name in CLASSES:
                group = [r for r in class_summary if r["exclude_class"] == name]
                plt.plot(shots, [r[f"{metric}_mean"] for r in group], marker="o", alpha=.65, label=name)
        macro_values = [r[f"{metric}_mean"] for r in macro]
        plt.plot(shots, macro_values, marker="o", linewidth=3, color="black", label="Macro")
        if 10 in shots: plt.axvline(10, linestyle="--", color="gray", label="10-shot primary")
        plt.xticks(shots); plt.xlabel("Shot"); plt.ylabel(metric.upper()); plt.grid(alpha=.25); plt.legend()
        plt.tight_layout(); prefix = "macro_" if macro_only else ""
        plt.savefig(os.path.join(figures, f"{prefix}{metric}_vs_shot.png"), dpi=180); plt.close()
    for metric in ("acc", "ap", "auc"):
        plot_metric(metric); plot_metric(metric, True)
    plt.figure(figsize=(8, 5))
    for name in CLASSES:
        group = [r for r in rows if r["exclude_class"] == name and r.get("alpha_mean", "") != ""]
        values = [statistics.mean(float(r["alpha_mean"]) for r in group if int(r["shot"]) == s)
                  if any(int(r["shot"]) == s for r in group) else float("nan") for s in shots]
        plt.plot(shots, values, marker="o", label=name)
    if 10 in shots: plt.axvline(10, linestyle="--", color="gray")
    plt.xticks(shots); plt.xlabel("Shot"); plt.ylabel("Alpha"); plt.grid(alpha=.25); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(figures, "alpha_vs_shot.png"), dpi=180); plt.close()

    lookup = {(r["exclude_class"], r["shot"]): r for r in class_summary}
    macro_lookup = {r["shot"]: r for r in macro}
    lines = ["# DDFSD Multi-shot Evaluation", "", "10-shot is the pre-specified primary setting; the best setting is determined by observed results.", "",
             "| Shot | ADM ACC | BigGAN ACC | glide ACC | Midjourney ACC | SD ACC | VQDM ACC | Macro ACC | Macro AP | Macro AUC |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    best_class_acc = {name: max(lookup[(name, s)]["acc_mean"] for s in shots) for name in CLASSES}
    best_macro = {metric: max(macro_lookup[s][f"{metric}_mean"] for s in shots)
                  for metric in ("acc", "ap", "auc")}
    for shot in shots:
        cells = ["**10**" if shot == 10 else str(shot)]
        for name in CLASSES:
            item = lookup[(name, shot)]; value = fmt(item["acc_mean"], item["acc_std"])
            cells.append(f"**{value}**" if item["acc_mean"] == best_class_acc[name] else value)
        m = macro_lookup[shot]
        for metric in ("acc", "ap", "auc"):
            value = fmt(m[f"{metric}_mean"], m[f"{metric}_std"])
            cells.append(f"**{value}**" if m[f"{metric}_mean"] == best_macro[metric] else value)
        lines.append("| " + " | ".join(cells) + " |")
    for name in CLASSES:
        lines += ["", f"## {name}", "", "| Shot | Per-seed ACC / AP / AUC | Mean ACC | Mean AP | Mean AUC |",
                  "|---:|---|---:|---:|---:|"]
        for shot in shots:
            group = [r for r in rows if r["exclude_class"] == name and int(r["shot"]) == shot]
            detail = "; ".join(f"{r['seed']}: {float(r['acc']):.4f}/{float(r['ap']):.4f}/{float(r['auc']):.4f}" for r in group)
            s = lookup[(name, shot)]
            lines.append(f"| {shot} | {detail} | {fmt(s['acc_mean'], s['acc_std'])} | {fmt(s['ap_mean'], s['ap_std'])} | {fmt(s['auc_mean'], s['auc_std'])} |")
    with open(os.path.join(output, "DDFSD_multishot_report.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__": main()
