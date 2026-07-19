"""Resolve the ADM parity reference CSV without guessing among existing files."""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from util.ddfsd_main_protocol_validation import resolve_reference_csv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--explicit")
    parser.add_argument("--candidate", action="append", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        resolved = resolve_reference_csv(args.explicit, args.candidate)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(resolved)


if __name__ == "__main__":
    main()
