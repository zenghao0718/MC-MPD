import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm


ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

DATA_DIRS = [
    ("ADM", "train", "ai"),
    ("ADM", "val", "ai"),
    ("BigGAN", "train", "ai"),
    ("BigGAN", "val", "ai"),
    ("glide", "train", "ai"),
    ("glide", "val", "ai"),
    ("Midjourney", "train", "ai"),
    ("Midjourney", "val", "ai"),
    ("SD", "train", "ai"),
    ("SD", "val", "ai"),
    ("VQDM", "train", "ai"),
    ("VQDM", "val", "ai"),
    ("real", "train", "nature"),
    ("real", "val", "nature"),
]

FAKE_CLASSES = ["ADM", "BigGAN", "glide", "Midjourney", "SD", "VQDM"]

DIFF_FOLDERS = [
    "ADM/val/ai",
    "real/val/nature",
    "BigGAN/val/ai",
    "VQDM/val/ai",
    "ADM/train/ai",
    "real/train/nature",
]

RNG_SEED = 20260709

PIXEL_COLUMNS = [
    "key",
    "folder",
    "raw_relative_path",
    "resized_relative_path",
    "raw_suffix",
    "raw_size",
    "online_size",
    "resized_size",
    "mae",
    "max_abs",
    "mse",
    "psnr",
    "status",
]

HIGHFREQ_COLUMNS = [
    "key",
    "folder",
    "raw_relative_path",
    "resized_relative_path",
    "raw_suffix",
    "laplacian_mae",
    "status",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check equivalence between raw GenImage and resize256 PNG images."
    )
    parser.add_argument(
        "--raw-root",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--resized-root",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
    )
    parser.add_argument("--sample-per-folder", default=50, type=int)
    parser.add_argument("--size", default=256, type=int)
    return parser.parse_args()


def is_image(path):
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def image_paths(root):
    if not root.exists():
        return []
    return sorted(
        (path for path in root.rglob("*") if is_image(path)),
        key=lambda path: path.as_posix().lower(),
    )


def relative_key(path, root):
    rel = path.relative_to(root)
    parent = rel.parent.as_posix()
    if parent == ".":
        return rel.stem.lower()
    return f"{parent}/{rel.stem}".lower()


def key_map(root):
    mapping = {}
    collisions = {}
    for path in image_paths(root):
        key = relative_key(path, root)
        if key in mapping:
            collisions.setdefault(key, [mapping[key]]).append(path)
        else:
            mapping[key] = path
    return mapping, collisions


def count_images(folder):
    return len(image_paths(folder))


def sample_list(items, max_count, rng):
    items = sorted(items)
    if len(items) <= max_count:
        return items
    return sorted(rng.sample(items, max_count))


def size_str(size):
    return f"{size[0]}x{size[1]}"


def resize_short_edge_rgb(image, target_short_edge):
    image = image.convert("RGB")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {width}x{height}")

    if width <= height:
        new_width = target_short_edge
        new_height = int(target_short_edge * height / width)
    else:
        new_height = target_short_edge
        new_width = int(target_short_edge * width / height)

    new_width = max(1, new_width)
    new_height = max(1, new_height)
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    if (new_width, new_height) == image.size:
        return image.copy()
    return image.resize((new_width, new_height), resampling)


def pixel_metrics(image_a, image_b):
    arr_a = np.asarray(image_a, dtype=np.uint8).astype(np.int16)
    arr_b = np.asarray(image_b, dtype=np.uint8).astype(np.int16)
    diff = arr_a - arr_b
    abs_diff = np.abs(diff)
    mae = float(abs_diff.mean())
    max_abs = int(abs_diff.max())
    mse = float((diff.astype(np.float32) ** 2).mean())
    psnr = math.inf if mse == 0.0 else float(20.0 * math.log10(255.0 / math.sqrt(mse)))
    return mae, max_abs, mse, psnr


def laplacian(image):
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return None
    return (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4.0 * gray[1:-1, 1:-1]
    )


def laplacian_mae(image_a, image_b):
    lap_a = laplacian(image_a)
    lap_b = laplacian(image_b)
    if lap_a is None or lap_b is None:
        return math.nan
    return float(np.abs(lap_a - lap_b).mean())


def write_key_check(raw_root, resized_root, out_dir):
    raw, raw_collisions = key_map(raw_root)
    resized, resized_collisions = key_map(resized_root)
    all_keys = sorted(set(raw) | set(resized))
    rows = []
    for key in all_keys:
        raw_path = raw.get(key)
        resized_path = resized.get(key)
        rows.append(
            {
                "key": key,
                "raw_relative_path": raw_path.relative_to(raw_root).as_posix()
                if raw_path
                else "",
                "resized_relative_path": resized_path.relative_to(resized_root).as_posix()
                if resized_path
                else "",
                "in_raw": key in raw,
                "in_resized": key in resized,
            }
        )
    df = pd.DataFrame(rows)
    path = out_dir / "relative_key_check.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return {
        "path": path,
        "raw_count": len(raw),
        "resized_count": len(resized),
        "common_count": len(set(raw) & set(resized)),
        "only_raw_count": len(set(raw) - set(resized)),
        "only_resized_count": len(set(resized) - set(raw)),
        "only_raw_head": [
            raw[key].relative_to(raw_root).as_posix()
            for key in sorted(set(raw) - set(resized))[:20]
        ],
        "only_resized_head": [
            resized[key].relative_to(resized_root).as_posix()
            for key in sorted(set(resized) - set(raw))[:20]
        ],
        "keys_equal": set(raw) == set(resized),
        "raw_key_collisions": raw_collisions,
        "resized_key_collisions": resized_collisions,
    }


def write_data_counts(raw_root, resized_root, out_dir):
    rows = []
    for class_name, split, subdir in DATA_DIRS:
        rel = Path(class_name) / split / subdir
        rows.append(
            {
                "class": class_name,
                "split": split,
                "subdir": subdir,
                "raw_count": count_images(raw_root / rel),
                "resized_count": count_images(resized_root / rel),
            }
        )
    df = pd.DataFrame(rows)
    path = out_dir / "data_counts.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df, path


def filename_stem_set(folder):
    return {path.stem.lower() for path in image_paths(folder)}


def write_train_val_overlap(resized_root, out_dir):
    checks = [(class_name, "ai") for class_name in FAKE_CLASSES]
    checks.append(("real", "nature"))

    overlap_blocks = []
    total_overlap = 0
    for class_name, subdir in checks:
        train_names = filename_stem_set(resized_root / class_name / "train" / subdir)
        val_names = filename_stem_set(resized_root / class_name / "val" / subdir)
        overlap = sorted(train_names & val_names)
        total_overlap += len(overlap)
        if overlap:
            overlap_blocks.append(
                f"[{class_name}] overlap_count={len(overlap)}\n"
                + "\n".join(overlap[:100])
            )

    path = out_dir / "train_val_filename_overlap.txt"
    if overlap_blocks:
        path.write_text("\n\n".join(overlap_blocks) + "\n", encoding="utf-8")
    else:
        path.write_text("no overlap\n", encoding="utf-8")
    return {
        "path": path,
        "total_overlap": total_overlap,
        "has_overlap": total_overlap > 0,
    }


def write_suffix_counts(resized_root, out_dir):
    counts = {}
    for path in image_paths(resized_root):
        suffix = path.suffix.lower()
        counts[suffix] = counts.get(suffix, 0) + 1
    rows = [
        {
            "suffix": suffix,
            "count": count,
            "ok_png": suffix == ".png",
        }
        for suffix, count in sorted(counts.items())
    ]
    df = pd.DataFrame(rows)
    path = out_dir / "suffix_counts.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    all_png = bool(len(df) and set(df["suffix"]) == {".png"})
    non_png_count = int(df[df["suffix"] != ".png"]["count"].sum()) if len(df) else 0
    return df, path, all_png, non_png_count


def write_size_check(resized_root, out_dir, sample_per_folder, size, rng):
    rows = []
    for class_name, split, subdir in DATA_DIRS:
        folder_rel = f"{class_name}/{split}/{subdir}"
        folder = resized_root / class_name / split / subdir
        paths = image_paths(folder)
        sampled = sample_list(paths, sample_per_folder, rng)
        for path in tqdm(sampled, desc=f"size {folder_rel}", leave=False):
            rel = path.relative_to(resized_root).as_posix()
            row = {
                "relative_path": rel,
                "folder": folder_rel,
                "width": None,
                "height": None,
                "short_edge": None,
                "ok": False,
                "status": "error",
            }
            try:
                with Image.open(path) as image:
                    width, height = image.size
                short_edge = min(width, height)
                row.update(
                    {
                        "width": width,
                        "height": height,
                        "short_edge": short_edge,
                        "ok": short_edge == size,
                        "status": "ok" if short_edge == size else "bad_short_edge",
                    }
                )
            except Exception as exc:
                row["status"] = f"error: {exc}"
            rows.append(row)
    df = pd.DataFrame(rows)
    path = out_dir / "size_check.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df, path


def folder_key_maps(root, folder_rel):
    folder = root / Path(folder_rel)
    mapping = {}
    for path in image_paths(folder):
        key = relative_key(path, root)
        mapping[key] = path
    return mapping


def raw_suffix_group(suffix):
    suffix = suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "jpg/jpeg"
    if suffix == ".png":
        return "png"
    return suffix.lstrip(".") or "unknown"


def summarize_pixel_by_group(pixel_df, group_col):
    rows = []
    for group_value, rows_df in pixel_df.groupby(group_col, sort=True):
        ok_rows = rows_df[rows_df["status"] == "ok"]
        rows.append(
            {
                group_col: group_value,
                "sample_count": int(len(rows_df)),
                "valid_count": int(len(ok_rows)),
                "size_mismatch_count": int((rows_df["status"] == "size_mismatch").sum()),
                "mae_mean": float(ok_rows["mae"].mean()) if len(ok_rows) else math.nan,
                "mae_median": float(ok_rows["mae"].median()) if len(ok_rows) else math.nan,
                "mae_max": float(ok_rows["mae"].max()) if len(ok_rows) else math.nan,
                "mse_mean": float(ok_rows["mse"].mean()) if len(ok_rows) else math.nan,
                "psnr_mean": float(ok_rows["psnr"].mean()) if len(ok_rows) else math.nan,
                "max_abs_mean": float(ok_rows["max_abs"].mean()) if len(ok_rows) else math.nan,
                "max_abs_max": float(ok_rows["max_abs"].max()) if len(ok_rows) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_highfreq_by_group(highfreq_df, group_col):
    rows = []
    for group_value, rows_df in highfreq_df.groupby(group_col, sort=True):
        ok_rows = rows_df[rows_df["status"] == "ok"]
        rows.append(
            {
                group_col: group_value,
                "sample_count": int(len(rows_df)),
                "valid_count": int(len(ok_rows)),
                "laplacian_mae_mean": float(ok_rows["laplacian_mae"].mean())
                if len(ok_rows)
                else math.nan,
                "laplacian_mae_median": float(ok_rows["laplacian_mae"].median())
                if len(ok_rows)
                else math.nan,
                "laplacian_mae_max": float(ok_rows["laplacian_mae"].max())
                if len(ok_rows)
                else math.nan,
            }
        )
    return pd.DataFrame(rows)


def write_pixel_and_highfreq_checks(raw_root, resized_root, out_dir, sample_per_folder, size, rng):
    pixel_rows = []
    highfreq_rows = []

    for folder_rel in DIFF_FOLDERS:
        raw_map = folder_key_maps(raw_root, folder_rel)
        resized_map = folder_key_maps(resized_root, folder_rel)
        sampled_keys = sample_list(raw_map.keys() & resized_map.keys(), sample_per_folder, rng)

        for key in tqdm(sampled_keys, desc=f"diff {folder_rel}", leave=False):
            raw_path = raw_map[key]
            resized_path = resized_map[key]
            raw_rel = raw_path.relative_to(raw_root).as_posix()
            resized_rel = resized_path.relative_to(resized_root).as_posix()
            suffix = raw_path.suffix.lower()

            pixel_row = {
                "key": key,
                "folder": folder_rel,
                "raw_relative_path": raw_rel,
                "resized_relative_path": resized_rel,
                "raw_suffix": raw_suffix_group(suffix),
                "raw_size": None,
                "online_size": None,
                "resized_size": None,
                "mae": math.nan,
                "max_abs": math.nan,
                "mse": math.nan,
                "psnr": math.nan,
                "status": "error",
            }
            highfreq_row = {
                "key": key,
                "folder": folder_rel,
                "raw_relative_path": raw_rel,
                "resized_relative_path": resized_rel,
                "raw_suffix": raw_suffix_group(suffix),
                "laplacian_mae": math.nan,
                "status": "error",
            }

            try:
                with Image.open(raw_path) as raw_image:
                    raw_original_size = raw_image.size
                    online = resize_short_edge_rgb(raw_image, size)
                with Image.open(resized_path) as resized_image:
                    saved = resized_image.convert("RGB")

                pixel_row["raw_size"] = size_str(raw_original_size)
                pixel_row["online_size"] = size_str(online.size)
                pixel_row["resized_size"] = size_str(saved.size)

                if online.size != saved.size:
                    pixel_row["status"] = "size_mismatch"
                    highfreq_row["status"] = "size_mismatch"
                else:
                    mae, max_abs, mse, psnr = pixel_metrics(online, saved)
                    lap_mae = laplacian_mae(online, saved)
                    pixel_row.update(
                        {
                            "mae": mae,
                            "max_abs": max_abs,
                            "mse": mse,
                            "psnr": psnr,
                            "status": "ok",
                        }
                    )
                    highfreq_row.update(
                        {
                            "laplacian_mae": lap_mae,
                            "status": "ok",
                        }
                    )
            except Exception as exc:
                pixel_row["status"] = f"error: {exc}"
                highfreq_row["status"] = f"error: {exc}"

            pixel_rows.append(pixel_row)
            highfreq_rows.append(highfreq_row)

    pixel_df = pd.DataFrame(pixel_rows, columns=PIXEL_COLUMNS)
    highfreq_df = pd.DataFrame(highfreq_rows, columns=HIGHFREQ_COLUMNS)

    pixel_per_path = out_dir / "pixel_diff_per_image.csv"
    highfreq_per_path = out_dir / "highfreq_diff_per_image.csv"
    pixel_df.to_csv(pixel_per_path, index=False, encoding="utf-8-sig")
    highfreq_df.to_csv(highfreq_per_path, index=False, encoding="utf-8-sig")

    pixel_summary_df = summarize_pixel_by_group(pixel_df, "folder")
    pixel_summary_path = out_dir / "pixel_diff_summary.csv"
    pixel_summary_df.to_csv(pixel_summary_path, index=False, encoding="utf-8-sig")

    pixel_suffix_summary_df = summarize_pixel_by_group(pixel_df, "raw_suffix")
    pixel_suffix_summary_path = out_dir / "pixel_diff_by_raw_suffix.csv"
    pixel_suffix_summary_df.to_csv(pixel_suffix_summary_path, index=False, encoding="utf-8-sig")

    highfreq_summary_df = summarize_highfreq_by_group(highfreq_df, "folder")
    highfreq_summary_path = out_dir / "highfreq_diff_summary.csv"
    highfreq_summary_df.to_csv(highfreq_summary_path, index=False, encoding="utf-8-sig")

    highfreq_suffix_summary_df = summarize_highfreq_by_group(highfreq_df, "raw_suffix")
    highfreq_suffix_summary_path = out_dir / "highfreq_diff_by_raw_suffix.csv"
    highfreq_suffix_summary_df.to_csv(
        highfreq_suffix_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "pixel_df": pixel_df,
        "pixel_per_path": pixel_per_path,
        "pixel_summary_df": pixel_summary_df,
        "pixel_summary_path": pixel_summary_path,
        "pixel_suffix_summary_df": pixel_suffix_summary_df,
        "pixel_suffix_summary_path": pixel_suffix_summary_path,
        "highfreq_df": highfreq_df,
        "highfreq_per_path": highfreq_per_path,
        "highfreq_summary_df": highfreq_summary_df,
        "highfreq_summary_path": highfreq_summary_path,
        "highfreq_suffix_summary_df": highfreq_suffix_summary_df,
        "highfreq_suffix_summary_path": highfreq_suffix_summary_path,
    }


def fmt_float(value, digits=6):
    if pd.isna(value):
        return "n/a"
    if math.isinf(float(value)):
        return "inf"
    return f"{float(value):.{digits}f}"


def read_preprocess_log(resized_root):
    path = resized_root / "preprocess_resize256_png_log.json"
    if not path.exists():
        return path, None
    try:
        return path, json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return path, None


def write_markdown_summary(
    raw_root,
    resized_root,
    out_dir,
    key_info,
    counts_df,
    counts_path,
    overlap_info,
    suffix_df,
    suffix_path,
    all_png,
    non_png_count,
    size_df,
    size_path,
    diff_info,
):
    expected_raw_root = Path("/root/autodl-tmp/GenImage").resolve()
    raw_root_correct = raw_root.resolve() == expected_raw_root
    counts_equal = bool((counts_df["raw_count"] == counts_df["resized_count"]).all())
    count_mismatches = counts_df[counts_df["raw_count"] != counts_df["resized_count"]]

    size_fail_df = size_df[size_df["ok"] != True] if len(size_df) else size_df
    size_pass = len(size_df) > 0 and len(size_fail_df) == 0

    pixel_df = diff_info["pixel_df"]
    ok_pixel = pixel_df[pixel_df["status"] == "ok"] if len(pixel_df) else pixel_df
    mismatch_count = int((pixel_df["status"] == "size_mismatch").sum()) if len(pixel_df) else 0

    highfreq_df = diff_info["highfreq_df"]
    ok_highfreq = highfreq_df[highfreq_df["status"] == "ok"] if len(highfreq_df) else highfreq_df

    pixel_mae_mean = ok_pixel["mae"].mean() if len(ok_pixel) else math.nan
    pixel_mae_max = ok_pixel["mae"].max() if len(ok_pixel) else math.nan
    pixel_max_abs_max = ok_pixel["max_abs"].max() if len(ok_pixel) else math.nan
    lap_mean = ok_highfreq["laplacian_mae"].mean() if len(ok_highfreq) else math.nan
    lap_max = ok_highfreq["laplacian_mae"].max() if len(ok_highfreq) else math.nan

    pixel_near_zero = (
        len(ok_pixel) > 0
        and pixel_mae_mean <= 0.01
        and pixel_mae_max <= 0.01
        and pixel_max_abs_max <= 1
        and mismatch_count == 0
    )
    freq_near_zero = len(ok_highfreq) > 0 and lap_mean <= 0.01 and lap_max <= 0.01

    pixel_suffix = diff_info["pixel_suffix_summary_df"]
    high_suffix = diff_info["highfreq_suffix_summary_df"]

    log_path, log_data = read_preprocess_log(resized_root)

    lines = []
    lines.append("# Resize256 PNG equivalence summary")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- Raw root: `{raw_root}`")
    lines.append(f"- Resized root: `{resized_root}`")
    lines.append(f"- Output dir: `{out_dir}`")
    lines.append(f"- Raw root is correct: {raw_root_correct}")
    lines.append("")
    lines.append("## 1. Relative key check")
    lines.append(f"- Raw image key count: {key_info['raw_count']}")
    lines.append(f"- Resize256 PNG key count: {key_info['resized_count']}")
    lines.append(f"- Common key count: {key_info['common_count']}")
    lines.append(f"- Only in raw: {key_info['only_raw_count']}")
    lines.append(f"- Only in resize256: {key_info['only_resized_count']}")
    lines.append(
        "- Result: "
        + ("raw/resized keys are identical." if key_info["keys_equal"] else "raw/resized keys are NOT identical.")
    )
    if key_info["only_raw_head"]:
        lines.append("- First raw-only keys:")
        lines.extend(f"  - `{item}`" for item in key_info["only_raw_head"])
    if key_info["only_resized_head"]:
        lines.append("- First resize-only keys:")
        lines.extend(f"  - `{item}`" for item in key_info["only_resized_head"])
    lines.append(f"- CSV: `{key_info['path']}`")
    lines.append("")
    lines.append("## 2. Data counts")
    lines.append(f"- CSV: `{counts_path}`")
    lines.append("- Result: " + ("all listed counts match." if counts_equal else "some listed counts do NOT match."))
    if not counts_equal:
        for _, row in count_mismatches.iterrows():
            lines.append(
                f"  - `{row['class']}/{row['split']}/{row['subdir']}`: "
                f"raw={row['raw_count']}, resize256_png={row['resized_count']}"
            )
    lines.append("")
    lines.append("## 3. Train/val filename overlap")
    lines.append(f"- Report: `{overlap_info['path']}`")
    lines.append(
        "- Result: "
        + ("no overlap." if not overlap_info["has_overlap"] else f"overlap found, total={overlap_info['total_overlap']}.")
    )
    lines.append("")
    lines.append("## 4. Output suffix check")
    lines.append(f"- CSV: `{suffix_path}`")
    if len(suffix_df):
        for _, row in suffix_df.iterrows():
            lines.append(f"  - `{row['suffix']}`: {row['count']}")
    lines.append("- Result: " + ("all resized image files are .png." if all_png else f"found {non_png_count} non-.png image files."))
    lines.append("")
    lines.append("## 5. Size check")
    lines.append(f"- CSV: `{size_path}`")
    lines.append(f"- Sampled images: {len(size_df)}")
    lines.append(
        "- Result: "
        + ("all sampled images have short edge 256." if size_pass else f"found {len(size_fail_df)} failures or bad short edges.")
    )
    lines.append("")
    lines.append("## 6. Pixel differences")
    lines.append(f"- Per-image CSV: `{diff_info['pixel_per_path']}`")
    lines.append(f"- Folder summary CSV: `{diff_info['pixel_summary_path']}`")
    lines.append(f"- Raw suffix summary CSV: `{diff_info['pixel_suffix_summary_path']}`")
    lines.append(f"- Valid samples: {len(ok_pixel)}")
    lines.append(f"- Size mismatches: {mismatch_count}")
    lines.append(f"- Overall MAE mean: {fmt_float(pixel_mae_mean)}")
    lines.append(f"- Overall MAE max: {fmt_float(pixel_mae_max)}")
    lines.append(f"- Overall max_abs max: {fmt_float(pixel_max_abs_max)}")
    lines.append("- Raw suffix groups:")
    for _, row in pixel_suffix.iterrows():
        lines.append(
            f"  - `{row['raw_suffix']}`: valid={int(row['valid_count'])}, "
            f"mae_mean={fmt_float(row['mae_mean'])}, mae_max={fmt_float(row['mae_max'])}, "
            f"max_abs_max={fmt_float(row['max_abs_max'])}, psnr_mean={fmt_float(row['psnr_mean'])}"
        )
    lines.append(
        "- Result: "
        + ("pixel differences are effectively zero." if pixel_near_zero else "pixel differences are not effectively zero.")
    )
    lines.append("")
    lines.append("## 7. High-frequency differences")
    lines.append(f"- Per-image CSV: `{diff_info['highfreq_per_path']}`")
    lines.append(f"- Folder summary CSV: `{diff_info['highfreq_summary_path']}`")
    lines.append(f"- Raw suffix summary CSV: `{diff_info['highfreq_suffix_summary_path']}`")
    lines.append(f"- Valid samples: {len(ok_highfreq)}")
    lines.append(f"- Overall Laplacian MAE mean: {fmt_float(lap_mean)}")
    lines.append(f"- Overall Laplacian MAE max: {fmt_float(lap_max)}")
    lines.append("- Raw suffix groups:")
    for _, row in high_suffix.iterrows():
        lines.append(
            f"  - `{row['raw_suffix']}`: valid={int(row['valid_count'])}, "
            f"laplacian_mae_mean={fmt_float(row['laplacian_mae_mean'])}, "
            f"laplacian_mae_max={fmt_float(row['laplacian_mae_max'])}"
        )
    lines.append(
        "- Result: "
        + ("high-frequency differences are effectively zero." if freq_near_zero else "high-frequency differences are not effectively zero.")
    )
    lines.append("")
    lines.append("## 8. Preprocess log")
    lines.append(f"- Log path: `{log_path}`")
    if log_data is None:
        lines.append("- Log data: unavailable.")
    else:
        lines.append(f"- output_format: {log_data.get('output_format')}")
        lines.append(f"- total_images: {log_data.get('total_images')}")
        lines.append(f"- success_count: {log_data.get('success_count')}")
        lines.append(f"- skipped_count: {log_data.get('skipped_count')}")
        lines.append(f"- failure_count: {log_data.get('failure_count')}")
        failed = log_data.get("failed_files", []) or []
        if failed:
            lines.append("- failed files:")
            for item in failed[:20]:
                lines.append(f"  - `{item.get('src')}`: {item.get('error')}")
        else:
            lines.append("- failed files: none")
    lines.append("")
    lines.append("## 9. Answers")
    lines.append(f"1. Correct raw-root used: {raw_root_correct}.")
    lines.append(f"2. All output images are PNG: {all_png}.")
    lines.append(f"3. Raw/resized key counts are identical: {key_info['keys_equal']}.")
    lines.append(f"4. Per-class train/val counts are identical: {counts_equal}.")
    lines.append(f"5. Train/val filename overlap exists: {overlap_info['has_overlap']}.")
    lines.append(f"6. Resize256 short edge check passes: {size_pass}.")
    lines.append(f"7. Online resize vs PNG pixel difference is near zero: {pixel_near_zero}.")
    lines.append(f"8. High-frequency difference is near zero: {freq_near_zero}.")
    lines.append("9. Real original JPEG and fake original PNG differences are eliminated in sampled pixel/frequency checks: " + str(pixel_near_zero and freq_near_zero) + ".")
    supports_equivalence = (
        raw_root_correct
        and all_png
        and key_info["common_count"] == key_info["resized_count"]
        and not overlap_info["has_overlap"]
        and size_pass
        and pixel_near_zero
        and freq_near_zero
    )
    if key_info["keys_equal"] and counts_equal:
        data_note = "full key/count equality holds."
    elif key_info["only_raw_count"] > 0 and key_info["only_resized_count"] == 0:
        data_note = "the generated dataset covers all readable/processed outputs, with raw-only failures recorded in the preprocess log."
    else:
        data_note = "there is a data construction mismatch beyond logged raw-only failures."
    lines.append(f"10. PNG resize256 supports advance-resize vs online-resize equivalence: {supports_equivalence}; {data_note}")
    lines.append("")

    summary_path = out_dir / "resize256_png_equivalence_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def main():
    args = parse_args()
    raw_root = args.raw_root.resolve()
    resized_root = args.resized_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_root.exists():
        raise FileNotFoundError(f"raw-root does not exist: {raw_root}")
    if not resized_root.exists():
        raise FileNotFoundError(f"resized-root does not exist: {resized_root}")

    rng = random.Random(RNG_SEED)

    print("Writing relative_key_check.csv ...")
    key_info = write_key_check(raw_root, resized_root, out_dir)

    print("Writing data_counts.csv ...")
    counts_df, counts_path = write_data_counts(raw_root, resized_root, out_dir)

    print("Writing train_val_filename_overlap.txt ...")
    overlap_info = write_train_val_overlap(resized_root, out_dir)

    print("Writing suffix_counts.csv ...")
    suffix_df, suffix_path, all_png, non_png_count = write_suffix_counts(resized_root, out_dir)

    print("Writing size_check.csv ...")
    size_df, size_path = write_size_check(
        resized_root,
        out_dir,
        args.sample_per_folder,
        args.size,
        rng,
    )

    print("Writing pixel/high-frequency difference CSVs ...")
    diff_info = write_pixel_and_highfreq_checks(
        raw_root,
        resized_root,
        out_dir,
        args.sample_per_folder,
        args.size,
        rng,
    )

    print("Writing resize256_png_equivalence_summary.md ...")
    summary_path = write_markdown_summary(
        raw_root,
        resized_root,
        out_dir,
        key_info,
        counts_df,
        counts_path,
        overlap_info,
        suffix_df,
        suffix_path,
        all_png,
        non_png_count,
        size_df,
        size_path,
        diff_info,
    )

    print("done")
    print(f"relative_key_check: {key_info['path']}")
    print(f"data_counts: {counts_path}")
    print(f"train_val_overlap: {overlap_info['path']}")
    print(f"suffix_counts: {suffix_path}")
    print(f"size_check: {size_path}")
    print(f"pixel_diff_summary: {diff_info['pixel_summary_path']}")
    print(f"pixel_diff_by_raw_suffix: {diff_info['pixel_suffix_summary_path']}")
    print(f"highfreq_diff_summary: {diff_info['highfreq_summary_path']}")
    print(f"highfreq_diff_by_raw_suffix: {diff_info['highfreq_suffix_summary_path']}")
    print(f"markdown_summary: {summary_path}")


if __name__ == "__main__":
    main()
