"""Reduce one dataset_v2 class without splitting augmentation families."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "dataset_v2"
DEFAULT_LOG_ROOT = ROOT / "logs" / "dataset_balance"
DEFAULT_CLASS_KEY = "tomato__yellow_leaf_curl_virus"
SPLITS = ("train", "val", "test")


class BalanceError(RuntimeError):
    """Raised when a safe class reduction cannot be completed."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Strip an overrepresented class by complete augmentation families. "
            "Removed files are archived below logs/dataset_balance."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--class-key", default=DEFAULT_CLASS_KEY)
    parser.add_argument(
        "--target-families",
        type=int,
        help=(
            "Families to retain. Default: the largest family count among all "
            "other classes."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Move excess images, replace the manifest, and update the summary.",
    )
    return parser


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required = {"split", "class_key", "family_id", "relative_path"}
        missing = required - set(fieldnames)
        if missing:
            raise BalanceError(f"Manifest columns missing: {sorted(missing)}")
        return fieldnames, list(reader)


def write_manifest(
    path: Path, fieldnames: list[str], rows: list[dict[str, str]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def group_families(
    rows: list[dict[str, str]], class_key: str
) -> dict[str, dict[str, list[dict[str, str]]]]:
    groups: dict[str, dict[str, list[dict[str, str]]]] = {
        split: defaultdict(list) for split in SPLITS
    }
    for row in rows:
        if row["class_key"] != class_key:
            continue
        split = row["split"]
        if split not in groups:
            raise BalanceError(f"Unexpected split for {class_key}: {split}")
        groups[split][row["family_id"]].append(row)
    return groups


def family_counts_by_class(rows: list[dict[str, str]]) -> dict[str, int]:
    families: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        families[row["class_key"]].add(row["family_id"])
    return {key: len(values) for key, values in families.items()}


def allocate_families(target: int, ratios: dict[str, float]) -> dict[str, int]:
    ratio_total = sum(float(ratios.get(split, 0)) for split in SPLITS)
    if ratio_total <= 0:
        raise BalanceError("Split ratios must add up to a positive number")

    exact = {
        split: target * float(ratios.get(split, 0)) / ratio_total
        for split in SPLITS
    }
    allocation = {split: math.floor(exact[split]) for split in SPLITS}
    remaining = target - sum(allocation.values())
    priority = sorted(
        SPLITS,
        key=lambda split: (exact[split] - allocation[split], split == "test"),
        reverse=True,
    )
    for split in priority[:remaining]:
        allocation[split] += 1
    return allocation


def stable_family_score(seed: int, class_key: str, split: str, family_id: str) -> str:
    value = f"{seed}|{class_key}|{split}|{family_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def select_retained_families(
    groups: dict[str, dict[str, list[dict[str, str]]]],
    allocation: dict[str, int],
    seed: int,
    class_key: str,
) -> dict[str, set[str]]:
    retained: dict[str, set[str]] = {}
    for split in SPLITS:
        available = list(groups[split])
        requested = allocation[split]
        if requested > len(available):
            raise BalanceError(
                f"Cannot retain {requested} {split} families; only {len(available)} exist"
            )
        available.sort(
            key=lambda family_id: stable_family_score(
                seed, class_key, split, family_id
            )
        )
        retained[split] = set(available[:requested])
    return retained


def build_summary(
    dataset_root: Path,
    rows: list[dict[str, str]],
    labels: dict,
) -> dict:
    class_keys = [item["key"] for item in labels["classes"]]
    split_counts = {split: 0 for split in SPLITS}
    split_class_counts = {
        key: {split: 0 for split in SPLITS} for key in class_keys
    }
    families: dict[str, set[str]] = {key: set() for key in class_keys}
    family_splits: dict[tuple[str, str], set[str]] = defaultdict(set)

    for row in rows:
        split = row["split"]
        class_key = row["class_key"]
        family_id = row["family_id"]
        split_counts[split] += 1
        split_class_counts[class_key][split] += 1
        families[class_key].add(family_id)
        family_splits[(class_key, family_id)].add(split)

    leakage = sum(1 for values in family_splits.values() if len(values) > 1)
    return {
        "class_count": len(class_keys),
        "image_count": len(rows),
        "family_count": sum(len(values) for values in families.values()),
        "split_counts": split_counts,
        "family_counts_by_class": {
            key: len(families[key]) for key in class_keys
        },
        "split_class_counts": split_class_counts,
        "augmentation_family_leakage_count": leakage,
        "label_file": str((dataset_root / "label.txt").resolve()),
        "labels_json": str((dataset_root / "labels.json").resolve()),
        "manifest": str((dataset_root / "manifest.csv").resolve()),
    }


def main() -> int:
    args = build_parser().parse_args()
    dataset_root = args.dataset.resolve()
    manifest_path = dataset_root / "manifest.csv"
    labels_path = dataset_root / "labels.json"
    summary_path = dataset_root / "dataset_summary.json"

    try:
        labels = read_json(labels_path)
        fieldnames, rows = read_manifest(manifest_path)
    except (OSError, json.JSONDecodeError, BalanceError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    known_keys = {item["key"] for item in labels.get("classes", [])}
    if args.class_key not in known_keys:
        print(f"ERROR: Unknown class key: {args.class_key}", file=sys.stderr)
        return 1

    all_family_counts = family_counts_by_class(rows)
    other_family_counts = {
        key: count
        for key, count in all_family_counts.items()
        if key != args.class_key
    }
    if not other_family_counts:
        print("ERROR: No comparison classes found", file=sys.stderr)
        return 1

    reference_class = max(
        other_family_counts, key=lambda key: (other_family_counts[key], key)
    )
    target = args.target_families or other_family_counts[reference_class]
    current = all_family_counts[args.class_key]
    if target < 1:
        print("ERROR: --target-families must be at least 1", file=sys.stderr)
        return 1
    if target > current:
        print(
            f"ERROR: Target {target} exceeds current family count {current}",
            file=sys.stderr,
        )
        return 1

    if args.target_families is None:
        reference_groups = group_families(rows, reference_class)
        allocation = {
            split: len(reference_groups[split]) for split in SPLITS
        }
    else:
        ratios = labels.get(
            "split_ratios", {"train": 0.70, "val": 0.15, "test": 0.15}
        )
        allocation = allocate_families(target, ratios)
    groups = group_families(rows, args.class_key)
    retained = select_retained_families(
        groups, allocation, args.seed, args.class_key
    )

    removed_rows = [
        row
        for row in rows
        if row["class_key"] == args.class_key
        and row["family_id"] not in retained[row["split"]]
    ]
    removed_paths = {row["relative_path"] for row in removed_rows}
    if len(removed_paths) != len(removed_rows):
        print("ERROR: Duplicate relative paths found in removal plan", file=sys.stderr)
        return 1
    kept_rows = [row for row in rows if row["relative_path"] not in removed_paths]
    removed_families = {
        split: len(groups[split]) - len(retained[split]) for split in SPLITS
    }
    removed_images = {
        split: sum(1 for row in removed_rows if row["split"] == split)
        for split in SPLITS
    }

    print(f"Class                 : {args.class_key}")
    print(f"Current families      : {current:,}")
    print(f"Target families       : {target:,}")
    print(f"Images to remove      : {len(removed_rows):,}")
    for split in SPLITS:
        print(
            f"  {split:<5}: retain {allocation[split]:>4,} families; "
            f"remove {removed_families[split]:>4,} families / "
            f"{removed_images[split]:>6,} images"
        )

    if not args.apply:
        print("\nDry run only. Re-run with --apply to update dataset_v2.")
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = args.log_root.resolve() / f"{timestamp}_{args.class_key}"
    metadata_dir = log_dir / "metadata_before"
    archive_dir = log_dir / "removed_images"
    metadata_dir.mkdir(parents=True, exist_ok=False)
    archive_dir.mkdir(parents=True)

    manifest_backup = metadata_dir / "manifest.csv"
    summary_backup = metadata_dir / "dataset_summary.json"
    shutil.copy2(manifest_path, manifest_backup)
    if summary_path.is_file():
        shutil.copy2(summary_path, summary_backup)

    removed_csv = log_dir / "removed_images.csv"
    write_manifest(removed_csv, fieldnames, removed_rows)
    (log_dir / "retained_families.json").write_text(
        json.dumps(
            {split: sorted(retained[split]) for split in SPLITS}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )

    report = {
        "status": "running",
        "timestamp": timestamp,
        "dataset": str(dataset_root),
        "class_key": args.class_key,
        "selection_seed": args.seed,
        "target_rule": (
            "explicit" if args.target_families is not None else "largest_other_class"
        ),
        "target_reference_class": (
            None if args.target_families is not None else reference_class
        ),
        "families_before": current,
        "families_after": target,
        "retained_families_by_split": allocation,
        "removed_families_by_split": removed_families,
        "images_before": len(rows),
        "images_removed": len(removed_rows),
        "images_removed_by_split": removed_images,
        "images_after": len(kept_rows),
        "archive": str(archive_dir),
    }
    report_path = log_dir / "balance_report.json"
    write_json(report_path, report)

    moved: list[tuple[Path, Path]] = []
    manifest_temp = dataset_root / ".manifest.balance.tmp"
    summary_temp = dataset_root / ".dataset_summary.balance.tmp"
    metadata_replaced = False

    try:
        for row in removed_rows:
            relative_path = Path(row["relative_path"])
            source = dataset_root / relative_path
            destination = archive_dir / relative_path
            if not source.is_file():
                raise BalanceError(f"Manifest image missing: {source}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
            moved.append((source, destination))

        updated_summary = build_summary(dataset_root, kept_rows, labels)
        if updated_summary["augmentation_family_leakage_count"] != 0:
            raise BalanceError("Updated manifest contains augmentation-family leakage")

        write_manifest(manifest_temp, fieldnames, kept_rows)
        write_json(summary_temp, updated_summary)
        os.replace(manifest_temp, manifest_path)
        os.replace(summary_temp, summary_path)
        metadata_replaced = True

        report["status"] = "completed"
        report["summary_after"] = updated_summary
        write_json(report_path, report)
    except BaseException as exc:
        report["status"] = "rolled_back"
        report["error"] = str(exc)
        if metadata_replaced or manifest_backup.is_file():
            shutil.copy2(manifest_backup, manifest_path)
            if summary_backup.is_file():
                shutil.copy2(summary_backup, summary_path)
        for source, destination in reversed(moved):
            if destination.is_file():
                source.parent.mkdir(parents=True, exist_ok=True)
                os.replace(destination, source)
        for temp_path in (manifest_temp, summary_temp):
            if temp_path.exists():
                temp_path.unlink()
        write_json(report_path, report)
        raise

    print(f"\nCompleted. Log and recovery archive: {log_dir}")
    print(f"Dataset images after reduction: {len(kept_rows):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
