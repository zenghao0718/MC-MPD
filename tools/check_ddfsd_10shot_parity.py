"""Audit 10-shot main-protocol results against the established main experiment."""

import argparse
import csv
import math
import os
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_SEEDS = (42, 101, 102, 103, 104)
EXACT_FIELDS = (
    "exclude_class",
    "ckpt_step",
    "checkpoint_model_mode",
    "branch_mode",
    "num_real_support",
    "num_fake_support",
    "num_real_query",
    "num_fake_query",
)
METRIC_FIELDS = ("acc", "ap", "auc")


def read_csv(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str, rows: Iterable[dict]) -> None:
    fields = [
        "exclude_class",
        "seed",
        "field",
        "new_value",
        "reference_value",
        "difference",
        "status",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _shot(row: dict) -> int:
    value = row.get("shot") or row.get("support_shot", "")
    if value == "":
        raise ValueError("Missing shot/support_shot field.")
    return int(value)


def _index_rows(rows: Sequence[dict], source: str) -> Tuple[Dict[int, dict], List[str]]:
    indexed = {}
    errors = []
    for row in rows:
        try:
            seed = int(row["seed"])
            shot = _shot(row)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{source} contains an invalid seed/shot row: {exc}")
            continue
        if shot != 10:
            errors.append(
                f"{source} contains non-10-shot row for seed {seed}: shot={shot}"
            )
        if seed in indexed:
            errors.append(f"{source} contains duplicate seed {seed}.")
        indexed[seed] = row
    return indexed, errors


def compare_parity_rows(
    new_rows: Sequence[dict],
    reference_rows: Sequence[dict],
    expected_seeds: Sequence[int] = DEFAULT_SEEDS,
    tolerance: float = 1e-6,
) -> Tuple[List[dict], List[str]]:
    new_by_seed, errors = _index_rows(new_rows, "new CSV")
    ref_by_seed, ref_errors = _index_rows(reference_rows, "reference CSV")
    errors.extend(ref_errors)
    expected = set(map(int, expected_seeds))
    for label, indexed in (("new CSV", new_by_seed), ("reference CSV", ref_by_seed)):
        missing = sorted(expected - set(indexed))
        unexpected = sorted(set(indexed) - expected)
        if missing:
            errors.append(f"{label} is missing seeds: {missing}")
        if unexpected:
            errors.append(f"{label} has unexpected seeds: {unexpected}")

    differences = []
    for seed in sorted(expected & set(new_by_seed) & set(ref_by_seed)):
        new = new_by_seed[seed]
        reference = ref_by_seed[seed]
        exclude_class = new.get("exclude_class", "")
        for field in EXACT_FIELDS:
            new_value = new.get(field, "")
            reference_value = reference.get(field, "")
            status = "PASS" if new_value == reference_value else "FAIL"
            differences.append(
                {
                    "exclude_class": exclude_class,
                    "seed": seed,
                    "field": field,
                    "new_value": new_value,
                    "reference_value": reference_value,
                    "difference": "" if status == "PASS" else "exact mismatch",
                    "status": status,
                }
            )
            if status == "FAIL":
                errors.append(
                    f"seed {seed} field {field} differs: new={new_value!r}, reference={reference_value!r}"
                )
        for field in METRIC_FIELDS:
            try:
                new_value = float(new[field])
                reference_value = float(reference[field])
                difference = abs(new_value - reference_value)
                status = (
                    "PASS"
                    if (
                        difference <= tolerance
                        or math.isclose(
                            difference, tolerance, rel_tol=1e-12, abs_tol=1e-15
                        )
                    )
                    else "FAIL"
                )
            except (KeyError, TypeError, ValueError) as exc:
                new_value = new.get(field, "")
                reference_value = reference.get(field, "")
                difference = "invalid numeric value"
                status = "FAIL"
                errors.append(f"seed {seed} field {field} is invalid: {exc}")
            differences.append(
                {
                    "exclude_class": exclude_class,
                    "seed": seed,
                    "field": field,
                    "new_value": new_value,
                    "reference_value": reference_value,
                    "difference": difference,
                    "status": status,
                }
            )
            if status == "FAIL" and isinstance(difference, float):
                errors.append(
                    f"seed {seed} field {field} difference {difference:.12g} exceeds {tolerance:.12g}"
                )
    return differences, errors


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check DDFSD main-protocol 10-shot parity"
    )
    parser.add_argument("--new_csv", required=True)
    parser.add_argument("--reference_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--expected_seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--tolerance", type=float, default=1e-6)
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = [
        int(value.strip()) for value in args.expected_seeds.split(",") if value.strip()
    ]
    differences, errors = compare_parity_rows(
        read_csv(args.new_csv), read_csv(args.reference_csv), seeds, args.tolerance
    )
    write_csv(args.output_csv, differences)
    if errors:
        raise SystemExit("10-shot parity FAILED:\n- " + "\n- ".join(errors))
    print(f"10-shot parity PASS: {args.new_csv} == {args.reference_csv}")


if __name__ == "__main__":
    main()
