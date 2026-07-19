#!/usr/bin/env python3
"""Benchmark the inference efficiency of a trained DDFSD dual-domain model.

This utility intentionally uses synthetic tensors: they reproduce the formal
inference computation graph without making any detector-quality claim.
"""

import argparse
import csv
import importlib.metadata
import json
import logging
import math
import os
import platform
import statistics
import subprocess
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from model.ddfsd import DDFSDDualDomainNet
from model.ddfsd_losses import (
    compute_alpha,
    compute_prototypes,
    compute_query_logits,
    compute_support_sigmas,
)
from util.ddfsd_frequency import load_frequency_stats

METHOD = "DDFSD"
MODEL_MODE = "dual"
SUPPORT_SHOT = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--freq_stats_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--warmup_iters", type=int, default=50)
    parser.add_argument("--measure_iters", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ddfsd_efficiency")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(output_dir / "benchmark.log", mode="w", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def validate_args(args: argparse.Namespace) -> Tuple[Path, Path, Path, torch.device]:
    ckpt_path = Path(args.ckpt_path).expanduser().resolve()
    stats_path = Path(args.freq_stats_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not stats_path.is_file():
        raise FileNotFoundError(f"Frequency stats not found: {stats_path}")
    if args.input_size <= 0 or args.input_size % 2:
        raise ValueError("--input_size must be a positive even integer for Haar-DWT.")
    for name in ("batch_size", "measure_iters", "repeats"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive.")
    if args.warmup_iters < 0:
        raise ValueError("--warmup_iters must be non-negative.")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Formal throughput requires a CUDA device; CPU results are not accepted.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Formal DDFSD throughput must run on an AutoDL CUDA GPU.")
    gpu_index = device.index if device.index is not None else torch.cuda.current_device()
    device_count = torch.cuda.device_count()
    if gpu_index < 0 or gpu_index >= device_count:
        raise ValueError(
            f"Invalid CUDA device index {gpu_index}; this host exposes {device_count} CUDA device(s)."
        )
    torch.cuda.set_device(gpu_index)
    device = torch.device("cuda", gpu_index)
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Use a new output directory for each benchmark run."
        )
    return ckpt_path, stats_path, output_dir, device


def config_value(checkpoint: Dict[str, Any], name: str, default: Any = None) -> Any:
    if checkpoint.get(name) is not None:
        return checkpoint[name]
    config = checkpoint.get("config", checkpoint.get("args", {}))
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def load_model(ckpt_path: Path, stats_path: Path, device: torch.device):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise KeyError(f"DDFSD checkpoint must contain a 'model' entry: {ckpt_path}")
    model_mode = config_value(checkpoint, "model_mode", "dual")
    if model_mode != MODEL_MODE:
        raise ValueError(
            f"This benchmark only accepts model_mode=dual checkpoints; checkpoint records {model_mode!r}."
        )
    stats = load_frequency_stats(str(stats_path))
    model = DDFSDDualDomainNet(pretrained=False, model_mode=MODEL_MODE)
    model.set_freq_stats(stats["mean"], stats["std"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    tau = float(config_value(checkpoint, "tau", 0.2))
    tau_r = float(config_value(checkpoint, "tau_r", 0.1))
    if not math.isfinite(tau) or tau <= 0 or not math.isfinite(tau_r) or tau_r <= 0:
        raise ValueError(f"Checkpoint tau/tau_r must be finite and positive, got tau={tau}, tau_r={tau_r}.")
    return model, checkpoint, tau, tau_r


def autocast_context(precision: str):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if precision == "fp16" else nullcontext()


def require_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} contains non-finite values.")


def prepare_support_cache(model, input_size: int, seed: int, device: torch.device, precision: str, tau_r: float):
    generator = torch.Generator(device=device).manual_seed(seed)
    support_images = torch.rand(
        2 * SUPPORT_SHOT, 3, input_size, input_size, generator=generator, device=device, dtype=torch.float32
    )
    with torch.inference_mode(), autocast_context(precision):
        outputs = model(support_images)
    z_rgb, z_freq = outputs.get("z_rgb"), outputs.get("z_freq")
    if z_rgb is None or z_freq is None:
        raise RuntimeError("Dual model forward must return both z_rgb and z_freq.")
    embedding_dim = z_rgb.shape[-1]
    support_rgb = z_rgb.float().reshape(2, SUPPORT_SHOT, embedding_dim).permute(1, 0, 2).unsqueeze(0)
    support_freq = z_freq.float().reshape(2, SUPPORT_SHOT, embedding_dim).permute(1, 0, 2).unsqueeze(0)
    expected = (1, SUPPORT_SHOT, 2, embedding_dim)
    if tuple(support_rgb.shape) != expected or tuple(support_freq.shape) != expected:
        raise RuntimeError(f"Unexpected support layout: {support_rgb.shape}, {support_freq.shape}; expected {expected}.")
    proto_rgb, proto_freq = compute_prototypes(support_rgb, support_freq, model_mode=MODEL_MODE)
    sigma_rgb, sigma_freq = compute_support_sigmas(support_rgb, support_freq, proto_rgb, proto_freq)
    alpha = compute_alpha(sigma_rgb, sigma_freq, tau_r=tau_r)
    expected_proto = (1, 2, embedding_dim)
    if tuple(proto_rgb.shape) != expected_proto or tuple(proto_freq.shape) != expected_proto:
        raise RuntimeError(
            f"Unexpected prototype layout: {proto_rgb.shape}, {proto_freq.shape}; expected {expected_proto}."
        )
    if tuple(alpha.shape) != (1, 2):
        raise RuntimeError(f"Unexpected alpha layout: {alpha.shape}; expected (1, 2).")
    for name, value in (("proto_rgb", proto_rgb), ("proto_freq", proto_freq), ("alpha", alpha)):
        require_finite(name, value)
    return proto_rgb, proto_freq, alpha


def run_query_inference(model, query_images, proto_rgb, proto_freq, alpha, tau, precision):
    with autocast_context(precision):
        outputs = model(query_images)
        query_rgb = outputs.get("z_rgb")
        query_freq = outputs.get("z_freq")
        if query_rgb is None or query_freq is None:
            raise RuntimeError("Dual model forward must return both z_rgb and z_freq.")
        expected_prefix = (query_images.shape[0],)
        if query_rgb.ndim != 2 or query_freq.ndim != 2 or query_rgb.shape[:1] != expected_prefix or query_freq.shape[:1] != expected_prefix:
            raise RuntimeError(
                f"Unexpected query embedding shapes: z_rgb={tuple(query_rgb.shape)}, "
                f"z_freq={tuple(query_freq.shape)}."
            )
        query_out = compute_query_logits(
            query_rgb=query_rgb.unsqueeze(0), query_freq=query_freq.unsqueeze(0),
            proto_rgb=proto_rgb, proto_freq=proto_freq, alpha=alpha,
            tau=tau, branch_mode=MODEL_MODE, model_mode=MODEL_MODE,
        )
    return query_out["logits"]


def validate_core_modules_called(model, uncalled_modules) -> Dict[str, bool]:
    """Reject a core branch only when it and every descendant are uncalled."""

    uncalled = set(uncalled_modules)
    module_names = {name for name, _ in model.named_modules()}
    result = {}
    for core_name in ("rgb_backbone", "rgb_projector", "freq_backbone", "freq_projector"):
        branch_names = sorted(
            name for name in module_names if name == core_name or name.startswith(f"{core_name}.")
        )
        if not branch_names:
            raise RuntimeError(f"Required DDFSD core module is missing: {core_name}.")
        result[core_name] = any(name not in uncalled for name in branch_names)
    missing = [name for name, called in result.items() if not called]
    if missing:
        raise RuntimeError(
            "fvcore did not call any module in these DDFSD core branches: " + ", ".join(missing)
        )
    return result


def model_complexity(model, input_size: int, device: torch.device) -> Dict[str, Any]:
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    if not 0 < trainable_params <= total_params:
        raise ValueError(f"Invalid parameter counts: trainable={trainable_params}, total={total_params}.")
    model_size_bytes = 0
    for tensor in model.state_dict().values():
        bytes_per_element = 4 if tensor.is_floating_point() else tensor.element_size()
        model_size_bytes += tensor.numel() * bytes_per_element
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError as exc:
        raise RuntimeError(
            "Missing optional dependency 'fvcore'. Install it in the AutoDL environment before running "
            "the formal efficiency benchmark."
        ) from exc
    flop_input = torch.rand(1, 3, input_size, input_size, device=device, dtype=torch.float32)
    with torch.inference_mode():
        analysis = FlopCountAnalysis(model, flop_input)
        flops = int(analysis.total())
        unsupported = dict(sorted((str(key), int(value)) for key, value in analysis.unsupported_ops().items()))
        uncalled = sorted(str(item) for item in analysis.uncalled_modules())
        core_module_call_check = validate_core_modules_called(model, uncalled)
    if flops <= 0:
        raise RuntimeError(f"fvcore returned an invalid FLOP count: {flops}.")
    return {
        "trainable_params": trainable_params, "trainable_params_m": trainable_params / 1e6,
        "total_params": total_params, "total_params_m": total_params / 1e6,
        "model_size_bytes": model_size_bytes, "model_size_mb": model_size_bytes / (1024 ** 2),
        "flops": flops, "flops_g": flops / 1e9, "flops_tool": "fvcore.nn.FlopCountAnalysis",
        "unsupported_ops": unsupported, "uncalled_modules": uncalled,
        "core_module_call_check": core_module_call_check,
    }


def benchmark_throughput(model, query_images, cache, tau, args):
    proto_rgb, proto_freq, alpha = cache
    expected_shape = (1, args.batch_size, 2)
    expected_device = query_images.device
    model_device = next(model.parameters()).device
    devices = {
        "model": model_device, "query_images": query_images.device, "proto_rgb": proto_rgb.device,
        "proto_freq": proto_freq.device, "alpha": alpha.device,
    }
    mismatched = {name: str(value) for name, value in devices.items() if value != expected_device}
    if mismatched:
        raise RuntimeError(f"Benchmark tensors/model are not all on {expected_device}: {mismatched}.")
    expected_query_shape = (args.batch_size, 3, args.input_size, args.input_size)
    if tuple(query_images.shape) != expected_query_shape:
        raise RuntimeError(f"Query tensor shape is {tuple(query_images.shape)}, expected {expected_query_shape}.")
    with torch.cuda.device(expected_device), torch.inference_mode():
        logits = run_query_inference(model, query_images, proto_rgb, proto_freq, alpha, tau, args.precision)
        if tuple(logits.shape) != expected_shape:
            raise RuntimeError(f"Query logits shape is {tuple(logits.shape)}, expected {expected_shape}.")
        require_finite("query logits", logits)
        for _ in range(args.warmup_iters):
            run_query_inference(model, query_images, proto_rgb, proto_freq, alpha, tau, args.precision)
        torch.cuda.synchronize(expected_device)
        rows = []
        for repeat in range(1, args.repeats + 1):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.measure_iters):
                run_query_inference(model, query_images, proto_rgb, proto_freq, alpha, tau, args.precision)
            end.record()
            torch.cuda.synchronize(expected_device)
            elapsed_ms = float(start.elapsed_time(end))
            total_images = args.batch_size * args.measure_iters
            if not math.isfinite(elapsed_ms) or elapsed_ms <= 0 or total_images <= 0:
                raise RuntimeError(
                    f"Invalid timing in repeat {repeat}: elapsed_ms={elapsed_ms}, total_images={total_images}."
                )
            images_per_second = total_images / (elapsed_ms / 1000.0)
            if not math.isfinite(images_per_second) or images_per_second <= 0:
                raise RuntimeError(f"Invalid throughput in repeat {repeat}: {images_per_second}.")
            rows.append({
                "method": METHOD, "model_mode": MODEL_MODE, "precision": args.precision,
                "batch_size": args.batch_size, "input_size": args.input_size,
                "warmup_iters": args.warmup_iters, "measure_iters": args.measure_iters,
                "repeat": repeat, "elapsed_ms": elapsed_ms, "total_images": total_images,
                "images_per_second": images_per_second,
            })
    values = [row["images_per_second"] for row in rows]
    summary = {"throughput_mean": statistics.fmean(values), "throughput_std": statistics.pstdev(values),
               "throughput_min": min(values), "throughput_max": max(values)}
    return rows, summary


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def environment_record(args, ckpt_path, stats_path, checkpoint, tau, tau_r, device):
    gpu_index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(gpu_index)
    checkpoint_step = checkpoint.get("step")
    if isinstance(checkpoint_step, torch.Tensor):
        checkpoint_step = checkpoint_step.item() if checkpoint_step.numel() == 1 else None
    return {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "git_commit": git_value("rev-parse", "HEAD"), "git_branch": git_value("branch", "--show-current"),
        "python_version": platform.python_version(), "platform": platform.platform(),
        "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(), "timm_version": package_version("timm"),
        "fvcore_version": package_version("fvcore"), "cuda_available": torch.cuda.is_available(),
        "gpu_name": properties.name, "gpu_total_memory_mb": properties.total_memory / (1024 ** 2),
        "device": str(device), "precision": args.precision, "batch_size": args.batch_size,
        "input_size": args.input_size, "warmup_iters": args.warmup_iters,
        "measure_iters": args.measure_iters, "repeats": args.repeats, "seed": args.seed,
        "checkpoint_path": str(ckpt_path), "checkpoint_step": checkpoint_step,
        "checkpoint_model_mode": MODEL_MODE, "freq_stats_path": str(stats_path), "tau": tau, "tau_r": tau_r,
    }


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, environment, complexity, raw_rows, summary, alpha):
    unsupported = json.dumps(complexity["unsupported_ops"], ensure_ascii=False)
    uncalled = json.dumps(complexity["uncalled_modules"], ensure_ascii=False)
    raw_table = "\n".join(
        f'| {row["repeat"]} | {row["elapsed_ms"]:.3f} | {row["total_images"]} | {row["images_per_second"]:.3f} |'
        for row in raw_rows
    )
    text = f"""# DDFSD效率测试报告

## 测试对象

DDFSD Dual（`model_mode=dual`），checkpoint：`{environment['checkpoint_path']}`，step：`{environment['checkpoint_step']}`，Git commit：`{environment['git_commit']}`。

## 测试环境与固定协议

- GPU：{environment['gpu_name']}；PyTorch：{environment['torch_version']}；CUDA：{environment['torch_cuda_version']}
- 输入：`[batch, 3, {environment['input_size']}, {environment['input_size']}]`；precision：`{environment['precision']}`；batch size：{environment['batch_size']}
- warmup / measure / repeats：{environment['warmup_iters']} / {environment['measure_iters']} / {environment['repeats']}
- support shot：{SUPPORT_SHOT}（前 {SUPPORT_SHOT} 张 real，后 {SUPPORT_SHOT} 张 fake）；tau：{environment['tau']}；tau_r：{environment['tau_r']}

## 五项效率指标

| Method | Trainable Params (M) ↓ | Total Params (M) ↓ | Model Size (MB) ↓ | FLOPs (G) ↓ | Throughput (images/s) ↑ |
|---|---:|---:|---:|---:|---:|
| DDFSD | {complexity['trainable_params_m']:.6f} | {complexity['total_params_m']:.6f} | {complexity['model_size_mb']:.6f} | {complexity['flops_g']:.6f} | {summary['throughput_mean']:.3f} ± {summary['throughput_std']:.3f} |

`Trainable Params` 是训练期间参与梯度更新的参数量；`Total Params` 是模型实际包含的全部参数量。Model Size 是完整 `state_dict` 按 FP32 部署口径计算的参数与 buffer 大小，不是训练 checkpoint 文件大小。

## Throughput 原始轮次

| Repeat | Elapsed (ms) | Total images | Images/s |
|---:|---:|---:|---:|
{raw_table}

## FLOPs 说明

工具：`{complexity['flops_tool']}`。Unsupported ops：`{unsupported}`。Uncalled modules：`{uncalled}`。

Basic tensor operations used by Haar-DWT may not be fully counted by the profiling tool.

Core module call check: `{json.dumps(complexity['core_module_call_check'], ensure_ascii=False)}`.

## 计时边界与声明

计时范围包含：已驻留 GPU 的 Query Tensor经过完整双分支 `model.forward`，与缓存 RGB/Frequency prototypes 计算距离，使用缓存 adaptive alpha 融合并输出二分类 logits。计时不包含 checkpoint/模型加载、Support 编码与初始化、prototype/alpha 一次性构建、磁盘读取、图像解码、DataLoader、CPU→GPU 传输、指标计算及文件写入。

Support、prototype 和 adaptive alpha 在所有 warmup、measurement 与 repeat 之前构建一次并固定复用。alpha 范围为 {float(alpha.min()):.6f}–{float(alpha.max()):.6f}（均值 {float(alpha.mean()):.6f}）。随机输入只用于效率计算和复现正式推理计算图，不代表任何检测性能结论。
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        ckpt_path, stats_path, output_dir, device = validate_args(args)
    except Exception as exc:
        print(f"DDFSD efficiency benchmark validation failed: {exc}", file=sys.stderr)
        return 1
    logger = configure_logging(output_dir)
    try:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        model, checkpoint, tau, tau_r = load_model(ckpt_path, stats_path, device)
        logger.info("Loaded dual checkpoint %s and frequency stats %s", ckpt_path, stats_path)
        complexity = model_complexity(model, args.input_size, device)
        cache = prepare_support_cache(model, args.input_size, args.seed, device, args.precision, tau_r)
        generator = torch.Generator(device=device).manual_seed(args.seed + 1)
        query_images = torch.rand(
            args.batch_size, 3, args.input_size, args.input_size,
            generator=generator, device=device, dtype=torch.float32,
        )
        raw_rows, throughput = benchmark_throughput(model, query_images, cache, tau, args)
        environment = environment_record(args, ckpt_path, stats_path, checkpoint, tau, tau_r, device)
        alpha = cache[2]
        run_config = vars(args).copy()
        run_config.update({"checkpoint_path": str(ckpt_path), "freq_stats_path": str(stats_path),
                           "output_dir": str(output_dir), "model_mode": MODEL_MODE,
                           "support_shot": SUPPORT_SHOT, "tau": tau, "tau_r": tau_r})
        summary = {
            "method": METHOD, "model_mode": MODEL_MODE, "input_size": args.input_size,
            "precision": args.precision, "batch_size": args.batch_size, **complexity, **throughput,
            "support_shot": SUPPORT_SHOT, "alpha_mean": float(alpha.mean()),
            "alpha_min": float(alpha.min()), "alpha_max": float(alpha.max()),
            "gpu_name": environment["gpu_name"], "checkpoint_step": environment["checkpoint_step"],
            "checkpoint_path": str(ckpt_path), "freq_stats_path": str(stats_path),
            "git_commit": environment["git_commit"],
            "core_modules_called": ";".join(
                f"{name}={str(called).lower()}"
                for name, called in complexity["core_module_call_check"].items()
            ),
        }
        write_json(output_dir / "environment.json", environment)
        write_json(output_dir / "run_config.json", run_config)
        write_json(output_dir / "model_complexity.json", complexity)
        write_csv(output_dir / "throughput_raw.csv", raw_rows, list(raw_rows[0]))
        summary_fields = [
            "method", "model_mode", "input_size", "precision", "batch_size",
            "trainable_params", "trainable_params_m", "total_params", "total_params_m",
            "model_size_bytes", "model_size_mb", "flops", "flops_g", "flops_tool",
            "throughput_mean", "throughput_std", "throughput_min", "throughput_max",
            "support_shot", "alpha_mean", "alpha_min", "alpha_max", "gpu_name",
            "checkpoint_step", "checkpoint_path", "freq_stats_path", "git_commit",
            "core_modules_called",
        ]
        summary_row = {field: summary.get(field) for field in summary_fields}
        write_csv(output_dir / "efficiency_summary.csv", [summary_row], summary_fields)
        write_report(output_dir / "DDFSD效率测试报告.md", environment, complexity, raw_rows, throughput, alpha)
        logger.info("Benchmark completed successfully; outputs written to %s", output_dir)
        return 0
    except Exception:
        logger.exception("DDFSD efficiency benchmark failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
