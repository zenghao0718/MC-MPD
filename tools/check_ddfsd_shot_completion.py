"""Check whether a formal DDFSD shot result is safe to skip."""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from util.ddfsd_main_protocol_validation import (
    FORMAL_PROTOCOL,
    validate_completed_result,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--git_commit", required=True)
    parser.add_argument("--exclude_class", required=True)
    parser.add_argument("--shot", type=int, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--ckpt_step", type=int, required=True)
    parser.add_argument("--freq_stats_path", required=True)
    parser.add_argument("--checkpoint_model_mode", required=True)
    parser.add_argument("--model_mode", required=True)
    parser.add_argument("--branch_mode", required=True)
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--tau_r", type=float, required=True)
    parser.add_argument("--max_eval_query_per_class", type=int, required=True)
    parser.add_argument("--zero_shot_metadata_per_class", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    expected = {
        "protocol": FORMAL_PROTOCOL,
        "git_commit": args.git_commit,
        "exclude_class": args.exclude_class,
        "shot": args.shot,
        "seeds": [
            int(value.strip()) for value in args.seeds.split(",") if value.strip()
        ],
        "data_root": args.data_root,
        "ckpt_path": args.ckpt_path,
        "ckpt_step": args.ckpt_step,
        "freq_stats_path": args.freq_stats_path,
        "checkpoint_model_mode": args.checkpoint_model_mode,
        "model_mode": args.model_mode,
        "branch_mode": args.branch_mode,
        "tau": args.tau,
        "tau_r": args.tau_r,
        "max_eval_query_per_class": args.max_eval_query_per_class,
        "zero_shot_metadata_per_class": args.zero_shot_metadata_per_class,
        "strict_formal_eval_images": True,
    }
    errors = validate_completed_result(args.output_dir, expected)
    if errors:
        raise SystemExit(
            "Existing result is not safe to skip; no files were changed:\n- "
            + "\n- ".join(errors)
        )
    print(f"COMPLETE_AND_MATCHING: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
