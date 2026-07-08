# -*- coding: utf-8 -*-
"""Generate DDFSD markdown summaries for the "remaining 4" exclude classes
(glide, Midjourney, SD, VQDM), plus a 4-class combined summary and merged CSVs.

This is a read-only analysis/reporting script. It never re-runs training or evaluation;
it only reads existing CSVs produced by:
  - scripts/eval_ddfsd_formal_all_steps.sh  -> formal_eval/ddfsd_eval_summary_all_steps.csv
  - scripts/eval_ddfsd_branch_modes.sh      -> branch_modes/ddfsd_<CLASS>_branch_modes_summary.csv
  - scripts/eval_ddfsd_alpha_grid.sh        -> alpha_grid/ddfsd_<CLASS>_alpha_grid_summary.csv
  - tools/parse_ddfsd_train_alpha_loss.py   -> csv/ddfsd_<CLASS>_train_alpha_loss_by_ckpt.csv

All numeric conclusions are computed directly from these CSVs. Where evidence is weak or
missing, the wording is deliberately conservative ("当前数据支持/部分支持/不支持/不能确认"),
never "证明"/"proves".

Subcommands:
  per-class  -- write <OUTPUT_PATH>/DDFSD_<CLASS>_FULL_SUMMARY.md for one class
  combined   -- write <RUN_ROOT>/DDFSD_REMAINING4_FULL_SUMMARY.md for glide/Midjourney/SD/VQDM
                and the remaining4 merged CSVs under <RUN_ROOT>/csv/
  all6       -- best-effort ADM/BigGAN + remaining4 combined summary + merged CSVs
                (only uses whatever ADM/BigGAN CSVs actually exist; never fabricates)
"""

import argparse
import csv
import glob
import os
import statistics
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

FIXED_HPARAMS = {
    "total_steps": 15000,
    "save_interval": 2500,
    "eval_interval": 2500,
    "batch_size": 16,
    "num_support_train": 5,
    "num_query_train": 5,
    "num_support_val": 10,
    "num_support_test": 10,
    "tau": 0.2,
    "tau_r": 0.1,
    "m_rf": 1.2,
    "m_ff": 0.6,
    "lambda_ff": 0.5,
    "lambda_sep_target": 0.03,
    "branch_dropout_dual_prob": 0.90,
    "branch_dropout_rgb_prob": 0.05,
    "branch_dropout_freq_prob": 0.05,
}
FORMAL_EVAL_STEPS_DEFAULT = [2500, 5000, 7500, 10000, 12500, 15000]
BRANCH_MODE_STEPS_DEFAULT = [2500, 5000, 7500, 10000, 12500, 15000]
ALPHA_GRID_STEPS_DEFAULT = [7500, 12500, 15000]
EVAL_SEEDS_DEFAULT = "42,101,102,103,104"


def read_csv(path: str) -> List[Dict[str, str]]:
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value, default=None) -> Optional[float]:
    if value is None or value == "" or value == "NA":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default=None) -> Optional[int]:
    f = to_float(value, default=None)
    if f is None:
        return default
    return int(round(f))


def fmt(value, digits=4) -> str:
    f = to_float(value)
    if f is None:
        return "NA"
    return f"{f:.{digits}f}"


def fmt_pm(mean_value, std_value, digits=4) -> str:
    m = fmt(mean_value, digits)
    s = fmt(std_value, digits)
    if m == "NA":
        return "NA"
    return f"{m}/{s}"


def get_git_commit(repo_root: str) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "NA"


def write_csv(path: str, rows: List[Dict], fieldnames: List[str]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Path resolution (standardized layout for glide/Midjourney/SD/VQDM; legacy
# fallback layout for ADM/BigGAN where folder/field names differ historically)
# ---------------------------------------------------------------------------

def class_output_path(run_root: str, exclude_class: str) -> str:
    return os.path.join(run_root, "ddfsd_10pct_steps15000", f"exclude_{exclude_class}")


def find_formal_eval_summary_csv(output_path: str) -> str:
    candidates = [
        os.path.join(output_path, "formal_eval", "ddfsd_eval_summary_all_steps.csv"),
        os.path.join(output_path, "ddfsd_eval_summary.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


def find_branch_modes_summary_csv(output_path: str, exclude_class: str) -> str:
    candidates = [
        os.path.join(output_path, "branch_modes", f"ddfsd_{exclude_class}_branch_modes_summary.csv"),
        os.path.join(output_path, "branch_modes", "ddfsd_adm_branch_modes_summary.csv"),
        os.path.join(output_path, "branch_diagnosis", "ddfsd_adm_branch_modes_summary.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    matches = glob.glob(os.path.join(output_path, "**", "*branch_modes_summary.csv"), recursive=True)
    return matches[0] if matches else ""


def find_alpha_grid_summary_csv(output_path: str, exclude_class: str) -> str:
    candidates = [
        os.path.join(output_path, "alpha_grid", f"ddfsd_{exclude_class}_alpha_grid_summary.csv"),
        os.path.join(output_path, f"alpha_grid_{exclude_class.lower()}", "ddfsd_alpha_grid_summary.csv"),
        os.path.join(output_path, "alpha_grid", "ddfsd_alpha_grid_summary.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    matches = glob.glob(os.path.join(output_path, "**", "*alpha_grid_summary.csv"), recursive=True)
    return matches[0] if matches else ""


def find_train_alpha_loss_csv(output_path: str, exclude_class: str) -> str:
    path = os.path.join(output_path, "csv", f"ddfsd_{exclude_class}_train_alpha_loss_by_ckpt.csv")
    return path if os.path.exists(path) else ""


def normalize_branch_row(row: Dict) -> Dict:
    """Map legacy field names (alpha_mean/min/max) to the standardized
    adaptive_alpha_mean/min/max naming used for the remaining-4 classes."""
    row = dict(row)
    if "adaptive_alpha_mean" not in row and "alpha_mean" in row:
        row["adaptive_alpha_mean"] = row["alpha_mean"]
    if "adaptive_alpha_min" not in row and "alpha_min" in row:
        row["adaptive_alpha_min"] = row["alpha_min"]
    if "adaptive_alpha_max" not in row and "alpha_max" in row:
        row["adaptive_alpha_max"] = row["alpha_max"]
    return row


def normalize_alpha_grid_row(row: Dict) -> Dict:
    row = dict(row)
    for key in ("adaptive_alpha_min", "adaptive_alpha_max", "used_alpha_min", "used_alpha_max"):
        row.setdefault(key, "NA")
    return row


# ---------------------------------------------------------------------------
# Judgement helpers (conservative wording only)
# ---------------------------------------------------------------------------

def judge_ratio(count: int, total: int, true_label: str, false_label: str) -> str:
    if total == 0:
        return "不能确认（无可比较的 checkpoint）"
    if count == total:
        return f"{true_label}（{count}/{total}）"
    if count == 0:
        return f"{false_label}（{count}/{total}）"
    return f"当前数据部分支持（{count}/{total} 个 checkpoint 上成立）"


def branch_modes_judgements(branch_rows: List[Dict]) -> Dict:
    steps = sorted({to_int(r["ckpt_step"]) for r in branch_rows if r.get("ckpt_step") is not None})
    by_step_mode = {}
    for r in branch_rows:
        by_step_mode[(to_int(r["ckpt_step"]), r["branch_mode"])] = r

    freq_gt_rgb = 0
    dual_is_best = 0
    total = 0
    alpha_first = None
    alpha_last = None
    alpha_series = []
    for step in steps:
        dual = by_step_mode.get((step, "dual"))
        rgb = by_step_mode.get((step, "rgb-only"))
        freq = by_step_mode.get((step, "freq-only"))
        if not (dual and rgb and freq):
            continue
        total += 1
        auc_dual = to_float(dual.get("auc_mean"))
        auc_rgb = to_float(rgb.get("auc_mean"))
        auc_freq = to_float(freq.get("auc_mean"))
        if auc_freq is not None and auc_rgb is not None and auc_freq > auc_rgb:
            freq_gt_rgb += 1
        if auc_dual is not None and auc_rgb is not None and auc_freq is not None:
            if auc_dual >= auc_rgb and auc_dual >= auc_freq:
                dual_is_best += 1
        alpha_mean = to_float(dual.get("adaptive_alpha_mean"))
        if alpha_mean is not None:
            alpha_series.append((step, alpha_mean))

    if alpha_series:
        alpha_first = alpha_series[0][1]
        alpha_last = alpha_series[-1][1]

    alpha_trend_up = None
    if alpha_first is not None and alpha_last is not None:
        alpha_trend_up = alpha_last > alpha_first + 0.01

    return {
        "steps_compared": total,
        "freq_gt_rgb_count": freq_gt_rgb,
        "dual_is_best_count": dual_is_best,
        "alpha_first": alpha_first,
        "alpha_last": alpha_last,
        "alpha_trend_up": alpha_trend_up,
        "alpha_series": alpha_series,
    }


def alpha_grid_judgements(alpha_rows: List[Dict]) -> Dict:
    steps = sorted({to_int(r["ckpt_step"]) for r in alpha_rows if r.get("ckpt_step") is not None})
    by_step_mode = {}
    for r in alpha_rows:
        by_step_mode[(to_int(r["ckpt_step"]), r["alpha_mode"])] = r

    fixed_beats_adaptive = 0
    total = 0
    best_fixed_per_step = {}
    for step in steps:
        adaptive = by_step_mode.get((step, "adaptive"))
        if not adaptive:
            continue
        ap_adaptive = to_float(adaptive.get("ap_mean"))
        fixed_candidates = []
        for mode_key in ("0", "0.25", "0.5", "0.75", "1"):
            row = by_step_mode.get((step, mode_key))
            if row is None:
                continue
            fixed_candidates.append((mode_key, to_float(row.get("ap_mean")), row))
        if not fixed_candidates or ap_adaptive is None:
            continue
        total += 1
        best_mode, best_ap, best_row = max(fixed_candidates, key=lambda item: (item[1] if item[1] is not None else -1))
        best_fixed_per_step[step] = (best_mode, best_ap, best_row, ap_adaptive)
        if best_ap is not None and best_ap > ap_adaptive:
            fixed_beats_adaptive += 1

    return {
        "steps_compared": total,
        "fixed_beats_adaptive_count": fixed_beats_adaptive,
        "best_fixed_per_step": best_fixed_per_step,
    }


def loss_near_zero(train_rows: List[Dict], key: str, threshold: float = 0.01) -> Tuple[int, int]:
    count = 0
    total = 0
    for row in train_rows:
        val = to_float(row.get(key))
        if val is None:
            continue
        total += 1
        if val < threshold:
            count += 1
    return count, total


# ---------------------------------------------------------------------------
# Per-class markdown
# ---------------------------------------------------------------------------

def build_formal_table(formal_rows: List[Dict]) -> Tuple[str, List[int]]:
    rows = sorted(formal_rows, key=lambda r: to_int(r["ckpt_step"]) or 0)
    tested_steps = [to_int(r["ckpt_step"]) for r in rows]
    lines = ["| ckpt_step | ACC mean/std | AP mean/std | AUC mean/std |", "|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['ckpt_step']} | {fmt_pm(r.get('acc_mean'), r.get('acc_std'))} | "
            f"{fmt_pm(r.get('ap_mean'), r.get('ap_std'))} | {fmt_pm(r.get('auc_mean'), r.get('auc_std'))} |"
        )
    return "\n".join(lines), tested_steps


def build_branch_table(branch_rows: List[Dict]) -> str:
    rows = sorted(branch_rows, key=lambda r: (to_int(r["ckpt_step"]) or 0, r["branch_mode"]))
    lines = [
        "| ckpt_step | branch_mode | ACC mean/std | AP mean/std | AUC mean/std | adaptive_alpha_mean | adaptive_alpha_min | adaptive_alpha_max |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        r = normalize_branch_row(r)
        lines.append(
            f"| {r['ckpt_step']} | {r['branch_mode']} | {fmt_pm(r.get('acc_mean'), r.get('acc_std'))} | "
            f"{fmt_pm(r.get('ap_mean'), r.get('ap_std'))} | {fmt_pm(r.get('auc_mean'), r.get('auc_std'))} | "
            f"{fmt(r.get('adaptive_alpha_mean'))} | {fmt(r.get('adaptive_alpha_min'))} | {fmt(r.get('adaptive_alpha_max'))} |"
        )
    return "\n".join(lines)


def build_alpha_grid_table(alpha_rows: List[Dict]) -> str:
    order = {"adaptive": 0, "0": 1, "0.25": 2, "0.5": 3, "0.75": 4, "1": 5}
    rows = sorted(
        alpha_rows,
        key=lambda r: (to_int(r["ckpt_step"]) or 0, order.get(r["alpha_mode"], 99)),
    )
    lines = [
        "| ckpt_step | alpha_mode | fixed_alpha | ACC mean/std | AP mean/std | AUC mean/std | adaptive_alpha_mean | used_alpha_mean | used_alpha_min | used_alpha_max |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        r = normalize_alpha_grid_row(r)
        alpha_label = "adaptive" if r["alpha_mode"] == "adaptive" else r["alpha_mode"]
        lines.append(
            f"| {r['ckpt_step']} | {alpha_label} | {r.get('fixed_alpha', '')} | "
            f"{fmt_pm(r.get('acc_mean'), r.get('acc_std'))} | {fmt_pm(r.get('ap_mean'), r.get('ap_std'))} | "
            f"{fmt_pm(r.get('auc_mean'), r.get('auc_std'))} | {fmt(r.get('adaptive_alpha_mean'))} | "
            f"{fmt(r.get('used_alpha_mean'))} | {fmt(r.get('used_alpha_min'))} | {fmt(r.get('used_alpha_max'))} |"
        )
    return "\n".join(lines)


def build_train_alpha_loss_table(train_rows: List[Dict]) -> str:
    rows = sorted(train_rows, key=lambda r: to_int(r["ckpt_step"]) or 0)
    lines = [
        "| ckpt_step | matched_train_step | alpha_mean | alpha_min | alpha_max | loss_rf | loss_ff | loss_sep | loss_dual | loss_total |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['ckpt_step']} | {r.get('matched_train_step', 'NA')} | {fmt(r.get('alpha_mean'))} | "
            f"{fmt(r.get('alpha_min'))} | {fmt(r.get('alpha_max'))} | {fmt(r.get('loss_rf'), 6)} | "
            f"{fmt(r.get('loss_ff'), 6)} | {fmt(r.get('loss_sep'), 6)} | {fmt(r.get('loss_dual'), 6)} | "
            f"{fmt(r.get('loss_total'), 6)} |"
        )
    return "\n".join(lines)


def best_ckpt_by_ap(formal_rows: List[Dict]) -> Optional[Dict]:
    rows = [r for r in formal_rows if to_float(r.get("ap_mean")) is not None]
    if not rows:
        return None
    return max(rows, key=lambda r: to_float(r["ap_mean"]))


def build_per_class_summary(
    exclude_class: str,
    run_root: str,
    output_path: str,
    data_root: str,
    repo_root: str,
) -> str:
    formal_csv = find_formal_eval_summary_csv(output_path)
    branch_csv = find_branch_modes_summary_csv(output_path, exclude_class)
    alpha_csv = find_alpha_grid_summary_csv(output_path, exclude_class)
    train_csv = find_train_alpha_loss_csv(output_path, exclude_class)

    formal_rows = read_csv(formal_csv)
    branch_rows = [normalize_branch_row(r) for r in read_csv(branch_csv)]
    alpha_rows = [normalize_alpha_grid_row(r) for r in read_csv(alpha_csv)]
    train_rows = read_csv(train_csv)

    git_commit = get_git_commit(repo_root)

    formal_table, tested_steps = build_formal_table(formal_rows)
    branch_table = build_branch_table(branch_rows)
    alpha_table = build_alpha_grid_table(alpha_rows)
    train_table = build_train_alpha_loss_table(train_rows)

    bj = branch_modes_judgements(branch_rows)
    aj = alpha_grid_judgements(alpha_rows)
    best_formal = best_ckpt_by_ap(formal_rows)
    rf_zero_count, rf_total = loss_near_zero(train_rows, "loss_rf")
    ff_zero_count, ff_total = loss_near_zero(train_rows, "loss_ff")

    freq_vs_rgb_judgement = judge_ratio(
        bj["freq_gt_rgb_count"], bj["steps_compared"], "当前数据支持", "当前数据不支持"
    )
    dual_best_judgement = judge_ratio(
        bj["dual_is_best_count"], bj["steps_compared"], "当前数据支持", "当前数据不支持"
    )
    if bj["alpha_trend_up"] is None:
        alpha_trend_judgement = "不能确认（分支诊断数据不足）"
    elif bj["alpha_trend_up"]:
        alpha_trend_judgement = (
            f"当前数据支持（adaptive_alpha_mean 从 step{bj['alpha_series'][0][0]} 的 "
            f"{bj['alpha_series'][0][1]:.4f} 上升到 step{bj['alpha_series'][-1][0]} 的 "
            f"{bj['alpha_series'][-1][1]:.4f}）"
        )
    else:
        alpha_trend_judgement = (
            f"当前数据不支持（adaptive_alpha_mean 从 step{bj['alpha_series'][0][0]} 的 "
            f"{bj['alpha_series'][0][1]:.4f} 到 step{bj['alpha_series'][-1][0]} 的 "
            f"{bj['alpha_series'][-1][1]:.4f}，未见明显上升趋势）"
        )
    fixed_alpha_judgement = judge_ratio(
        aj["fixed_beats_adaptive_count"], aj["steps_compared"], "当前数据支持", "当前数据不支持"
    )

    best_alpha_lines = []
    for step, (mode, ap, row, ap_adaptive) in sorted(aj["best_fixed_per_step"].items()):
        best_alpha_lines.append(
            f"  - step{step}: 最优固定 alpha={mode}（AP={fmt(ap)}），adaptive AP={fmt(ap_adaptive)}"
        )
    best_alpha_block = "\n".join(best_alpha_lines) if best_alpha_lines else "  - 无可比较数据"

    if best_formal is not None:
        best_ckpt_line = (
            f"最优 checkpoint（按正式 dual 测试 AP）：step{best_formal['ckpt_step']}"
            f"（AP={fmt(best_formal.get('ap_mean'))}, AUC={fmt(best_formal.get('auc_mean'))}, "
            f"ACC={fmt(best_formal.get('acc_mean'))}）"
        )
    else:
        best_ckpt_line = "最优 checkpoint：不能确认（正式测试数据缺失）"

    loss_rf_judgement = (
        f"{rf_zero_count}/{rf_total} 个 checkpoint 的 loss_rf < 0.01" if rf_total else "不能确认（训练日志解析失败或缺失）"
    )
    loss_ff_judgement = (
        f"{ff_zero_count}/{ff_total} 个 checkpoint 的 loss_ff < 0.01" if ff_total else "不能确认（训练日志解析失败或缺失）"
    )

    missing_note_lines = []
    missing_formal = sorted(set(FORMAL_EVAL_STEPS_DEFAULT) - set(tested_steps))
    if missing_formal:
        missing_note_lines.append(f"- 正式 dual 测试缺失的 checkpoint：{missing_formal}")
    if not formal_rows:
        missing_note_lines.append("- 未找到正式 dual 测试结果 CSV。")
    if not branch_rows:
        missing_note_lines.append("- 未找到三模式诊断结果 CSV。")
    if not alpha_rows:
        missing_note_lines.append("- 未找到 fixed alpha 诊断结果 CSV。")
    if not train_rows:
        missing_note_lines.append("- 未找到训练期 alpha/loss 解析结果 CSV。")
    missing_block = "\n".join(missing_note_lines) if missing_note_lines else "- 无缺失，六个 checkpoint 均已完成三条诊断链路。"

    md = f"""# DDFSD {exclude_class} 全量结果汇总（exclude_{exclude_class}）

- 仓库：`https://github.com/zenghao0718/MC-MPD`
- 分支：`exp-ddfsd-dual-domain-margin-v1`
- git commit：`{git_commit}`
- 工作目录：`{repo_root}`

## 1. 实验基本信息

| 字段 | 值 |
|---|---|
| exclude_class | {exclude_class} |
| data_root | {data_root} |
| output_path | {output_path} |
| freq_stats_path | {output_path}/freq_stats.pt |
| total_steps | {FIXED_HPARAMS['total_steps']} |
| checkpoint steps | {FORMAL_EVAL_STEPS_DEFAULT} |
| 正式测试实际测试的 checkpoint | {tested_steps if tested_steps else '（无，见下方缺失说明）'} |
| eval seeds | {EVAL_SEEDS_DEFAULT} |
| support shot | {FIXED_HPARAMS['num_support_test']} |
| branch dropout (dual/rgb/freq) | {FIXED_HPARAMS['branch_dropout_dual_prob']}/{FIXED_HPARAMS['branch_dropout_rgb_prob']}/{FIXED_HPARAMS['branch_dropout_freq_prob']} |
| tau | {FIXED_HPARAMS['tau']} |
| tau_r | {FIXED_HPARAMS['tau_r']} |
| m_rf | {FIXED_HPARAMS['m_rf']} |
| m_ff | {FIXED_HPARAMS['m_ff']} |
| lambda_ff | {FIXED_HPARAMS['lambda_ff']} |
| lambda_sep_target | {FIXED_HPARAMS['lambda_sep_target']} |
| git commit | {git_commit} |

**数据缺失说明**：
{missing_block}

## 2. 正式 dual 5-seed 10-shot 测试（split=val, eval_seeds={EVAL_SEEDS_DEFAULT}）

{formal_table if formal_rows else '（无数据）'}

## 3. 三模式诊断（dual / rgb-only / freq-only, support_shot=10, 5-seed）

{branch_table if branch_rows else '（无数据）'}

## 4. fixed alpha 诊断（branch_mode=dual, support_shot=10, 5-seed）

{alpha_table if alpha_rows else '（无数据）'}

每个 checkpoint 的 adaptive 确切数值（非模糊描述）：

{chr(10).join(
    f"  - step{step}: adaptive_alpha_mean={fmt(row.get('adaptive_alpha_mean'))}, "
    f"adaptive_alpha_min={fmt(row.get('adaptive_alpha_min'))}, adaptive_alpha_max={fmt(row.get('adaptive_alpha_max'))}"
    for step, row in sorted({
        to_int(r['ckpt_step']): r for r in alpha_rows if r.get('alpha_mode') == 'adaptive'
    }.items())
) if alpha_rows else '  - 无数据'}

## 5. 训练期 alpha/loss 表（按 checkpoint 匹配最近的训练日志记录）

{train_table if train_rows else '（无数据）'}

注：`sigma_rgb_mean`/`sigma_freq_mean`/`sigma_diff_mean` 在训练日志（`train_ddfsd.py` 的
`logger.logkv_mean` 调用）中未被记录，因此本表和源 CSV 中一律填 `NA`，不做编造。

## 6. 类内结论（克制表述，仅基于以上数据）

1. freq-only 是否高于 rgb-only（按 AUC）：{freq_vs_rgb_judgement}
2. dual 是否最高（按 AUC，同时不低于 rgb-only 和 freq-only）：{dual_best_judgement}
3. adaptive alpha 是否随训练/测试推进逐渐偏向 RGB：{alpha_trend_judgement}
4. fixed alpha 是否超过 adaptive（按 AP）：{fixed_alpha_judgement}
{best_alpha_block}
5. {best_ckpt_line}
6. loss_rf 是否基本为 0（训练期，checkpoint 附近记录）：{loss_rf_judgement}
7. loss_ff 是否基本为 0（训练期，checkpoint 附近记录）：{loss_ff_judgement}

以上结论均直接由上述 CSV 计算得出，不代表因果证明，仅描述当前一次训练/评测下观察到的现象。
"""
    return md


# ---------------------------------------------------------------------------
# Combined (remaining4 / all6) summary
# ---------------------------------------------------------------------------

def load_class_data(run_root: str, exclude_class: str) -> Dict:
    output_path = class_output_path(run_root, exclude_class)
    formal_csv = find_formal_eval_summary_csv(output_path)
    branch_csv = find_branch_modes_summary_csv(output_path, exclude_class)
    alpha_csv = find_alpha_grid_summary_csv(output_path, exclude_class)
    train_csv = find_train_alpha_loss_csv(output_path, exclude_class)
    return {
        "exclude_class": exclude_class,
        "output_path": output_path,
        "formal_rows": read_csv(formal_csv),
        "branch_rows": [normalize_branch_row(r) for r in read_csv(branch_csv)],
        "alpha_rows": [normalize_alpha_grid_row(r) for r in read_csv(alpha_csv)],
        "train_rows": read_csv(train_csv),
        "formal_csv": formal_csv,
        "branch_csv": branch_csv,
        "alpha_csv": alpha_csv,
        "train_csv": train_csv,
    }


CN_DIGITS = {1: "一", 2: "两", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八"}


def cn_count_label(n: int) -> str:
    return CN_DIGITS.get(n, str(n))


def build_combined_markdown(class_data_list: List[Dict], title: str, run_root: str, repo_root: str) -> str:
    git_commit = get_git_commit(repo_root)
    n_label = cn_count_label(len(class_data_list))

    # Table 1: best formal ckpt per class
    t1_lines = [
        "| exclude_class | best_ckpt_by_AP | ACC mean/std | AP mean/std | AUC mean/std |",
        "|---|---|---|---|---|",
    ]
    best_formal_by_class = {}
    for cd in class_data_list:
        best = best_ckpt_by_ap(cd["formal_rows"])
        best_formal_by_class[cd["exclude_class"]] = best
        if best is None:
            t1_lines.append(f"| {cd['exclude_class']} | NA | NA | NA | NA |")
        else:
            t1_lines.append(
                f"| {cd['exclude_class']} | {best['ckpt_step']} | {fmt_pm(best.get('acc_mean'), best.get('acc_std'))} | "
                f"{fmt_pm(best.get('ap_mean'), best.get('ap_std'))} | {fmt_pm(best.get('auc_mean'), best.get('auc_std'))} |"
            )

    # Table 2: three-mode best at that class's best formal ckpt
    t2_lines = [
        "| exclude_class | ckpt_step | best_branch_by_AP | dual_AP | freq_only_AP | rgb_only_AP | dual_ACC | freq_only_ACC | rgb_only_ACC | adaptive_alpha_mean |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cd in class_data_list:
        best_formal = best_formal_by_class[cd["exclude_class"]]
        step = to_int(best_formal["ckpt_step"]) if best_formal else None
        by_mode = {r["branch_mode"]: r for r in cd["branch_rows"] if to_int(r["ckpt_step"]) == step}
        dual, freq, rgb = by_mode.get("dual"), by_mode.get("freq-only"), by_mode.get("rgb-only")
        if not (dual and freq and rgb):
            t2_lines.append(f"| {cd['exclude_class']} | {step if step else 'NA'} | NA | NA | NA | NA | NA | NA | NA | NA |")
            continue
        aps = {"dual": to_float(dual.get("ap_mean")), "freq-only": to_float(freq.get("ap_mean")), "rgb-only": to_float(rgb.get("ap_mean"))}
        best_mode = max(aps, key=lambda k: aps[k] if aps[k] is not None else -1)
        t2_lines.append(
            f"| {cd['exclude_class']} | {step} | {best_mode} | {fmt(dual.get('ap_mean'))} | {fmt(freq.get('ap_mean'))} | "
            f"{fmt(rgb.get('ap_mean'))} | {fmt(dual.get('acc_mean'))} | {fmt(freq.get('acc_mean'))} | {fmt(rgb.get('acc_mean'))} | "
            f"{fmt(dual.get('adaptive_alpha_mean'))} |"
        )

    # Table 3: fixed alpha best result per class
    t3_lines = [
        "| exclude_class | ckpt_step | best_alpha_by_AP | adaptive_AP | best_fixed_alpha_AP | best_fixed_alpha | adaptive_alpha_mean | used_alpha_mean |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cd in class_data_list:
        aj = alpha_grid_judgements(cd["alpha_rows"])
        best_step, best_entry = None, None
        for step, entry in aj["best_fixed_per_step"].items():
            if best_entry is None or (entry[1] or -1) > (best_entry[1] or -1):
                best_step, best_entry = step, entry
        if best_step is None:
            t3_lines.append(f"| {cd['exclude_class']} | NA | NA | NA | NA | NA | NA | NA |")
            continue
        best_mode, best_ap, best_row, ap_adaptive = best_entry
        overall_best = "adaptive" if (ap_adaptive is not None and best_ap is not None and ap_adaptive >= best_ap) else best_mode
        adaptive_row = next(
            (r for r in cd["alpha_rows"] if to_int(r["ckpt_step"]) == best_step and r["alpha_mode"] == "adaptive"),
            {},
        )
        t3_lines.append(
            f"| {cd['exclude_class']} | {best_step} | {overall_best} | {fmt(ap_adaptive)} | {fmt(best_ap)} | {best_mode} | "
            f"{fmt(adaptive_row.get('adaptive_alpha_mean'))} | {fmt(best_row.get('used_alpha_mean'))} |"
        )

    # Table 4: full train alpha/loss table across all classes and checkpoints
    t4_lines = [
        "| exclude_class | ckpt_step | matched_train_step | alpha_mean | alpha_min | alpha_max | loss_rf | loss_ff | loss_sep | loss_dual | loss_total |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cd in class_data_list:
        for r in sorted(cd["train_rows"], key=lambda r: to_int(r["ckpt_step"]) or 0):
            t4_lines.append(
                f"| {cd['exclude_class']} | {r['ckpt_step']} | {r.get('matched_train_step', 'NA')} | "
                f"{fmt(r.get('alpha_mean'))} | {fmt(r.get('alpha_min'))} | {fmt(r.get('alpha_max'))} | "
                f"{fmt(r.get('loss_rf'), 6)} | {fmt(r.get('loss_ff'), 6)} | {fmt(r.get('loss_sep'), 6)} | "
                f"{fmt(r.get('loss_dual'), 6)} | {fmt(r.get('loss_total'), 6)} |"
            )

    # Table 5: phenomenon judgement table
    t5_lines = [
        "| exclude_class | freq-only 是否高于 rgb-only | dual 是否最高 | adaptive alpha 是否后期偏 RGB | fixed alpha 是否超过 adaptive | loss_rf 是否接近 0 | loss_ff 是否接近 0 | 简短结论 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cd in class_data_list:
        bj = branch_modes_judgements(cd["branch_rows"])
        aj = alpha_grid_judgements(cd["alpha_rows"])
        rf_zero_count, rf_total = loss_near_zero(cd["train_rows"], "loss_rf")
        ff_zero_count, ff_total = loss_near_zero(cd["train_rows"], "loss_ff")
        freq_j = judge_ratio(bj["freq_gt_rgb_count"], bj["steps_compared"], "支持", "不支持")
        dual_j = judge_ratio(bj["dual_is_best_count"], bj["steps_compared"], "支持", "不支持")
        if bj["alpha_trend_up"] is None:
            alpha_j = "不能确认"
        else:
            alpha_j = "支持" if bj["alpha_trend_up"] else "不支持"
        fixed_j = judge_ratio(aj["fixed_beats_adaptive_count"], aj["steps_compared"], "支持", "不支持")
        rf_j = f"{rf_zero_count}/{rf_total}" if rf_total else "NA"
        ff_j = f"{ff_zero_count}/{ff_total}" if ff_total else "NA"
        short_conclusion = (
            f"dual最高{dual_j}；freq高于rgb{freq_j}；fixed alpha超过adaptive{fixed_j}"
        )
        t5_lines.append(
            f"| {cd['exclude_class']} | {freq_j} | {dual_j} | {alpha_j} | {fixed_j} | {rf_j} | {ff_j} | {short_conclusion} |"
        )

    class_names = ", ".join(cd["exclude_class"] for cd in class_data_list)
    md = f"""# {title}

- 仓库：`https://github.com/zenghao0718/MC-MPD`
- 分支：`exp-ddfsd-dual-domain-margin-v1`
- git commit：`{git_commit}`
- 覆盖类别：{class_names}
- TensorBoard 启动命令：`tensorboard --logdir {run_root} --host 0.0.0.0 --port 6006`
- TensorBoard logdir：`{run_root}`

## 表 1：{n_label}类正式 dual 最佳结果（按 AP 选择 checkpoint）

{chr(10).join(t1_lines)}

## 表 2：{n_label}类三模式最佳结果（在各自表1的最佳 checkpoint 上比较）

{chr(10).join(t2_lines)}

## 表 3：{n_label}类 fixed alpha 最佳结果（在 alpha_grid 已测的 checkpoint 中，取 adaptive AP 最高的一个）

{chr(10).join(t3_lines)}

## 表 4：每个存档点训练期 alpha/loss 总表（全部类别 x 全部 checkpoint）

{chr(10).join(t4_lines)}

## 表 5：{n_label}类现象判断表

{chr(10).join(t5_lines)}

## 说明

- 所有数值均直接来自各类别的 `formal_eval/branch_modes/alpha_grid/csv` 目录下的 CSV，未做任何人工调整。
- "支持/不支持/不能确认" 为对当前一次训练+评测观察到的现象的描述，不构成因果证明。
- 各类别详细数据与逐 checkpoint、逐 seed 结果见各自的 `DDFSD_<CLASS>_FULL_SUMMARY.md`。
"""
    return md


def write_combined_csvs(class_data_list: List[Dict], out_dir: str, prefix: str):
    def merge(key: str, filename: str):
        all_rows = []
        fieldnames = None
        for cd in class_data_list:
            for row in cd[key]:
                fieldnames = fieldnames or list(row.keys())
                all_rows.append(row)
        if fieldnames:
            write_csv(os.path.join(out_dir, filename), all_rows, fieldnames)
        return len(all_rows)

    counts = {
        "formal_rows": merge("formal_rows", f"{prefix}_formal_eval_summary_all.csv"),
        "branch_rows": merge("branch_rows", f"{prefix}_branch_modes_summary_all.csv"),
        "alpha_rows": merge("alpha_rows", f"{prefix}_alpha_grid_summary_all.csv"),
        "train_rows": merge("train_rows", f"{prefix}_train_alpha_loss_by_ckpt_all.csv"),
    }
    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_per_class(args):
    md = build_per_class_summary(
        exclude_class=args.exclude_class,
        run_root=args.run_root,
        output_path=args.output_path,
        data_root=args.data_root,
        repo_root=args.repo_root,
    )
    out_path = os.path.join(args.output_path, f"DDFSD_{args.exclude_class}_FULL_SUMMARY.md")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(md)
    print(f"Wrote {out_path}")


def cmd_combined(args):
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    class_data_list = [load_class_data(args.run_root, c) for c in classes]

    for cd in class_data_list:
        missing = [k for k in ("formal_csv", "branch_csv", "alpha_csv", "train_csv") if not cd[k]]
        if missing:
            print(f"WARNING: exclude_class={cd['exclude_class']} missing sources: {missing}", file=sys.stderr)

    md = build_combined_markdown(
        class_data_list,
        title="DDFSD Remaining-4 (glide/Midjourney/SD/VQDM) Alpha & Branch 诊断总结",
        run_root=args.run_root,
        repo_root=args.repo_root,
    )
    out_path = os.path.join(args.run_root, "DDFSD_REMAINING4_FULL_SUMMARY.md")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(md)
    print(f"Wrote {out_path}")

    csv_dir = os.path.join(args.run_root, "csv")
    counts = write_combined_csvs(class_data_list, csv_dir, "ddfsd_remaining4")
    print(f"Merged CSVs into {csv_dir}: {counts}")


def cmd_all6(args):
    remaining_classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    legacy_classes = ["ADM", "BigGAN"]
    all_classes = legacy_classes + remaining_classes

    class_data_list = []
    available_legacy = []
    for c in all_classes:
        cd = load_class_data(args.run_root, c)
        has_any = any(cd[k] for k in ("formal_csv", "branch_csv", "alpha_csv", "train_csv"))
        if c in legacy_classes:
            if has_any:
                available_legacy.append(c)
                class_data_list.append(cd)
        else:
            class_data_list.append(cd)

    if not available_legacy:
        note_path = os.path.join(args.run_root, "DDFSD_ALL6_ALPHA_BRANCH_SUMMARY.md")
        with open(note_path, "w", encoding="utf-8") as handle:
            handle.write(
                "# DDFSD ALL6 Alpha & Branch 诊断总结\n\n"
                "未在当前路径找到 ADM / BigGAN 对应 csv，因此 ALL6 表未生成，"
                "仅生成了 remaining4 表（见 DDFSD_REMAINING4_FULL_SUMMARY.md）。\n"
            )
        print(f"No ADM/BigGAN data found; wrote placeholder note to {note_path}")
        return

    md = build_combined_markdown(
        class_data_list,
        title="DDFSD ALL6 (ADM/BigGAN/glide/Midjourney/SD/VQDM) Alpha & Branch 诊断总结",
        run_root=args.run_root,
        repo_root=args.repo_root,
    )
    missing_legacy = set(legacy_classes) - set(available_legacy)
    if missing_legacy:
        md += f"\n\n**注**：未找到 {sorted(missing_legacy)} 的对应 csv，上表中这些类别的相关行/列可能为 NA，未做编造。\n"

    out_path = os.path.join(args.run_root, "DDFSD_ALL6_ALPHA_BRANCH_SUMMARY.md")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(md)
    print(f"Wrote {out_path}")

    csv_dir = os.path.join(args.run_root, "csv")
    counts = write_combined_csvs(class_data_list, csv_dir, "ddfsd_all6")
    print(f"Merged CSVs into {csv_dir}: {counts}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate DDFSD remaining-4 / all6 markdown summaries and merged CSVs.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_per_class = sub.add_parser("per-class")
    p_per_class.add_argument("--exclude_class", required=True)
    p_per_class.add_argument("--run_root", required=True)
    p_per_class.add_argument("--output_path", required=True)
    p_per_class.add_argument("--data_root", default="/root/autodl-tmp/data")
    p_per_class.add_argument("--repo_root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p_per_class.set_defaults(func=cmd_per_class)

    p_combined = sub.add_parser("combined")
    p_combined.add_argument("--run_root", required=True)
    p_combined.add_argument("--classes", default="glide,Midjourney,SD,VQDM")
    p_combined.add_argument("--repo_root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p_combined.set_defaults(func=cmd_combined)

    p_all6 = sub.add_parser("all6")
    p_all6.add_argument("--run_root", required=True)
    p_all6.add_argument("--classes", default="glide,Midjourney,SD,VQDM")
    p_all6.add_argument("--repo_root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p_all6.set_defaults(func=cmd_all6)

    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
