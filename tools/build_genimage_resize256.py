import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageFile


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a GenImage resize256 dataset with every output saved as PNG."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("/root/autodl-tmp/GenImage"),
        help="Source GenImage root.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("/root/autodl-tmp/GenImage_resize256"),
        help="Destination resize256 PNG root.",
    )
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def is_image(path):
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def scan_images(root):
    return sorted(
        (path for path in root.rglob("*") if is_image(path)),
        key=lambda path: path.as_posix().lower(),
    )


def is_inside(child, parent):
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def dst_has_files(dst):
    return dst.exists() and any(path.is_file() for path in dst.rglob("*"))


def output_path_for(src_path, src_root, dst_root):
    rel = src_path.relative_to(src_root)
    return dst_root / rel.parent / f"{rel.stem}.png"


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


def find_output_collisions(src_paths, src_root, dst_root):
    buckets = {}
    for src_path in src_paths:
        dst_path = output_path_for(src_path, src_root, dst_root)
        key = str(dst_path.resolve()).lower()
        buckets.setdefault(key, []).append(src_path)

    collisions = []
    for key, paths in buckets.items():
        if len(paths) > 1:
            collisions.append(
                {
                    "output_path": str(output_path_for(paths[0], src_root, dst_root)),
                    "source_paths": [str(path) for path in paths],
                }
            )
    return collisions


def top_level_counts(src_paths, src_root):
    counts = {}
    for path in src_paths:
        rel = path.relative_to(src_root)
        top = rel.parts[0] if rel.parts else "."
        counts[top] = counts.get(top, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0].lower()))


def process_one(item):
    src_path, dst_path, size = item
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src_path) as image:
            resized = resize_short_edge_rgb(image, size)
            resized.save(dst_path, format="PNG")
        return {
            "ok": True,
            "src": str(src_path),
            "dst": str(dst_path),
            "error": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "src": str(src_path),
            "dst": str(dst_path),
            "error": repr(exc),
        }


def print_collision_report(collisions, limit=50):
    print(f"ERROR: output path collisions found: {len(collisions)}")
    for collision in collisions[:limit]:
        print(f"output: {collision['output_path']}")
        for source_path in collision["source_paths"]:
            print(f"  source: {source_path}")
    if len(collisions) > limit:
        print(f"... {len(collisions) - limit} more collisions omitted")


def main():
    args = parse_args()
    src = args.src.resolve()
    dst = args.dst.resolve()
    size = args.size
    workers = max(1, args.workers)

    print(f"src: {src}")
    print(f"dst: {dst}")
    print(f"size: {size}")
    print(f"workers: {workers}")
    print(f"dry_run: {args.dry_run}")

    if not src.exists():
        raise FileNotFoundError(f"src does not exist: {src}")
    if not src.is_dir():
        raise NotADirectoryError(f"src is not a directory: {src}")
    if is_inside(dst, src):
        raise ValueError(f"dst must not be inside src: dst={dst}, src={src}")

    if dst_has_files(dst):
        message = (
            f"Destination already exists and contains files: {dst}. "
            "Refusing to overwrite. Move/rename/use an empty destination first."
        )
        if args.dry_run:
            print(f"WARNING: {message}")
        else:
            raise FileExistsError(message)

    start = time.time()
    start_time = datetime.now().isoformat(timespec="seconds")

    print("Scanning source images ...")
    src_paths = scan_images(src)
    counts = top_level_counts(src_paths, src)
    print(f"total_images: {len(src_paths)}")
    print("top_level_counts:")
    for name, count in counts.items():
        print(f"  {name}: {count}")

    print("Checking output path collisions ...")
    collisions = find_output_collisions(src_paths, src, dst)
    if collisions:
        print_collision_report(collisions)
        raise SystemExit(2)
    print("output_path_collisions: 0")

    if args.dry_run:
        elapsed = time.time() - start
        print("dry-run complete; no images were written.")
        print(f"elapsed_seconds: {elapsed:.3f}")
        return

    dst.mkdir(parents=True, exist_ok=True)
    items = [(path, output_path_for(path, src, dst), size) for path in src_paths]

    success_count = 0
    failed_files = []
    print("Processing images ...")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_one, item) for item in items]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result["ok"]:
                success_count += 1
            else:
                failed_files.append(
                    {
                        "src": result["src"],
                        "dst": result["dst"],
                        "error": result["error"],
                    }
                )
            if index % 5000 == 0 or index == len(futures):
                print(
                    f"processed={index}/{len(futures)} "
                    f"success={success_count} failed={len(failed_files)}"
                )

    end_time = datetime.now().isoformat(timespec="seconds")
    elapsed = time.time() - start
    log = {
        "src": str(src),
        "dst": str(dst),
        "size": size,
        "workers": workers,
        "output_format": "png",
        "total_images": len(src_paths),
        "success_count": success_count,
        "skipped_count": 0,
        "failure_count": len(failed_files),
        "failed_files": failed_files,
        "start_time": start_time,
        "end_time": end_time,
        "elapsed_seconds": round(elapsed, 3),
        "top_level_counts": counts,
    }
    log_path = dst / "preprocess_resize256_png_log.json"
    log_path.write_text(
        json.dumps(log, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"log_path: {log_path}")
    print(f"success_count: {success_count}")
    print(f"failure_count: {len(failed_files)}")
    if failed_files:
        print("failed_files:")
        for item in failed_files[:50]:
            print(f"  {item['src']} -> {item['error']}")
        if len(failed_files) > 50:
            print(f"  ... {len(failed_files) - 50} more failed files omitted")


if __name__ == "__main__":
    main()
