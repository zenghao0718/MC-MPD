"""Validate and aggregate six-class DDFSD multi-shot result directories."""

import argparse
import csv
import os
import statistics
from collections import defaultdict


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
    return parser.parse_args()


def main():
    args = parse_args(); output = args.output_dir or args.input_root
    rows = []
    for name in CLASSES:
        candidates = [os.path.join(args.input_root, f"exclude_{name}", "multishot", "multishot_per_seed.csv"),
                      os.path.join(args.input_root, f"exclude_{name}", "multishot_per_seed.csv")]
        path = next((p for p in candidates if os.path.isfile(p)), None)
        if path is None:
            raise FileNotFoundError(f"Missing multishot_per_seed.csv for {name}; checked {candidates}")
        class_rows = read_csv(path)
        if any(row["exclude_class"] != name for row in class_rows):
            raise ValueError(f"Class mismatch in {path}")
        rows.extend(class_rows)
    keys = [(r["exclude_class"], int(r["seed"]), int(r["shot"])) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate exclude_class/seed/shot rows found.")
    seeds = sorted({int(r["seed"]) for r in rows}); shots = sorted({int(r["shot"]) for r in rows})
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
