#!/usr/bin/env python3
"""Prepare or verify provenance-bound DDFSD all-source frequency statistics."""

import argparse
from pathlib import Path


DEFAULT_OUTPUT = (
    "/root/autodl-tmp/runs/transfer_ms_cocoai/train/"
    "ddfsd_allsource_full_step15000/freq_stats_allsource.pt"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="/root/autodl-tmp/data_fsd_full/GenImage")
    parser.add_argument("--output_path", default=DEFAULT_OUTPUT)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_sha_sidecar(stats_path, digest):
    sidecar = Path(f"{stats_path}.sha256")
    sidecar.write_text(f"{digest}  {Path(stats_path).name}\n", encoding="utf-8")
    return sidecar


def main():
    args = parse_args()
    import torch

    from util.ddfsd_frequency import (
        ALLSOURCE_CLASSES,
        compute_frequency_stats,
        load_frequency_stats,
        sha256_file,
        validate_allsource_frequency_metadata,
    )

    output_path = Path(args.output_path).resolve()
    if output_path.exists() and not args.overwrite:
        stats = load_frequency_stats(str(output_path))
        validate_allsource_frequency_metadata(stats.get("metadata"))
        digest = sha256_file(str(output_path))
        sidecar = Path(f"{output_path}.sha256")
        if sidecar.exists():
            recorded = sidecar.read_text(encoding="utf-8").split()[0]
            if recorded != digest:
                raise RuntimeError(
                    f"Existing SHA sidecar mismatch: recorded={recorded}, actual={digest}"
                )
        else:
            write_sha_sidecar(output_path, digest)
        print(f"Frequency stats already exist with valid provenance: {output_path}")
        print(f"sha256={digest}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    compute_frequency_stats(
        data_root=args.data_root,
        output_path=str(output_path),
        classes=ALLSOURCE_CLASSES,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    stats = load_frequency_stats(str(output_path))
    validate_allsource_frequency_metadata(stats.get("metadata"))
    digest = sha256_file(str(output_path))
    sidecar = write_sha_sidecar(output_path, digest)
    print(f"Saved all-source frequency stats: {output_path}")
    print(f"Saved SHA-256 sidecar: {sidecar}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
