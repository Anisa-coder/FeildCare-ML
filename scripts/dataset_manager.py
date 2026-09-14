"""Manage the standardized FieldCare dataset safely.

The script operates on ``dataset/`` and keeps ``labels.json``, ``label.txt``,
``manifest.csv``, and ``dataset_summary.json`` consistent.

Examples:

    python scripts/dataset_manager.py list
    python scripts/dataset_manager.py validate
    python scripts/dataset_manager.py create --crop tomato --disease wilt \
        --display-name "Tomato Wilt" --apply
    python scripts/dataset_manager.py rename 17 --display-name "Tomato Early Blight"
    python scripts/dataset_manager.py delete 17 --apply
    python scripts/dataset_manager.py package
    python scripts/dataset_manager.py kaggle-upload --package-dir kaggle_upload/fieldcare_dataset_20260913_120000 \
        --owner YOUR_KAGGLE_USERNAME --slug fieldcare-plant-disease-dataset \
        --title "FieldCare Plant Disease Dataset"

Class creation, renaming, and deletion are dry runs unless ``--apply`` is
given. Deleted class files are moved to ``logs/_dataset_manager_archive`` so
they remain recoverable.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Callable
import uuid
import zipfile


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "dataset"
LOG_ROOT = ROOT / "logs" / "dataset_manager"
ARCHIVE_ROOT = ROOT / "logs" / "_dataset_manager_archive"
PACKAGE_ROOT = ROOT / "kaggle_upload"
MANIFEST_FIELDS = (
    "split",
    "class_id",
    "class_index",
    "class_key",
    "display_name",
    "crop",
    "disease",
    "family_id",
    "augmentation",
    "relative_path",
)


class DatasetManagerError(RuntimeError):
    """Raised when a requested dataset change is unsafe or invalid."""


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def slug(value: str, field: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    if not normalized:
        raise DatasetManagerError(f"{field} must contain letters or numbers")
    return normalized


def display_from(crop: str, disease: str) -> str:
    return f"{crop.replace('_', ' ').title()} {disease.replace('_', ' ').title()}"


def class_key(crop: str, disease: str) -> str:
    return f"{crop}__{disease}"


def relative_directory(crop: str, disease: str) -> str:
    return f"{crop}/{disease}"


def metadata_paths(dataset: Path) -> dict[str, Path]:
    return {
        "labels": dataset / "labels.json",
        "label_text": dataset / "label.txt",
        "manifest": dataset / "manifest.csv",
        "summary": dataset / "dataset_summary.json",
    }


def load_metadata(dataset: Path) -> tuple[dict, list[dict]]:
    paths = metadata_paths(dataset)
    if not dataset.is_dir():
        raise DatasetManagerError(f"Dataset directory not found: {dataset}")
    if not paths["labels"].is_file():
        raise DatasetManagerError(f"labels.json not found: {paths['labels']}")
    if not paths["manifest"].is_file():
        raise DatasetManagerError(f"manifest.csv not found: {paths['manifest']}")

    payload = json.loads(paths["labels"].read_text(encoding="utf-8"))
    classes = payload.get("classes")
    if not isinstance(classes, list) or not classes:
        raise DatasetManagerError("labels.json contains no classes")
    classes = [dict(item) for item in classes]
    validate_classes(classes)
    return payload, classes


def validate_classes(classes: list[dict]) -> None:
    ids = []
    indices = []
    keys = set()
    directories = set()
    for item in classes:
        required = {
            "id", "index", "key", "display_name", "crop", "disease",
            "relative_directory",
        }
        missing = required - set(item)
        if missing:
            raise DatasetManagerError(f"Class record missing fields: {sorted(missing)}")
        class_id = int(item["id"])
        index = int(item["index"])
        key = str(item["key"])
        directory = str(item["relative_directory"]).replace("\\", "/")
        if class_id < 1 or index < 0 or not key or not directory:
            raise DatasetManagerError(f"Invalid class record: {item!r}")
        if key in keys or directory in directories:
            raise DatasetManagerError(f"Duplicate class key or directory: {item!r}")
        ids.append(class_id)
        indices.append(index)
        keys.add(key)
        directories.add(directory)
    if sorted(ids) != list(range(1, len(classes) + 1)):
        raise DatasetManagerError("Class IDs must be contiguous and start at 1")
    if sorted(indices) != list(range(len(classes))):
        raise DatasetManagerError("Class indices must be contiguous and start at 0")


def get_class(classes: list[dict], class_id: int) -> dict:
    for item in classes:
        if int(item["id"]) == class_id:
            return item
    raise DatasetManagerError(f"Unknown class ID: {class_id}")


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def write_metadata(dataset: Path, payload: dict, classes: list[dict]) -> None:
    validate_classes(classes)
    paths = metadata_paths(dataset)
    payload = dict(payload)
    payload["class_count"] = len(classes)
    payload["classes"] = classes
    atomic_write_text(paths["labels"], json.dumps(payload, indent=2) + "\n")
    label_text = "".join(
        f"{item['id']} {item['display_name']}\n"
        for item in sorted(classes, key=lambda value: int(value["id"]))
    )
    atomic_write_text(paths["label_text"], label_text)


def validate_manifest_header(manifest: Path) -> None:
    with manifest.open("r", newline="", encoding="utf-8") as handle:
        fields = csv.DictReader(handle).fieldnames or []
    missing = set(MANIFEST_FIELDS) - set(fields)
    if missing:
        raise DatasetManagerError(
            f"Manifest has missing columns: {sorted(missing)}"
        )


def manifest_row_count(manifest: Path, class_id: int) -> int:
    count = 0
    with manifest.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["class_id"]) == class_id:
                count += 1
    return count


def rewrite_manifest(
    manifest: Path,
    transform: Callable[[dict], dict | None],
) -> None:
    temporary = manifest.with_name(f".{manifest.name}.{uuid.uuid4().hex}.tmp")
    with manifest.open("r", newline="", encoding="utf-8") as source, temporary.open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        reader = csv.DictReader(source)
        fields = reader.fieldnames or []
        missing = set(MANIFEST_FIELDS) - set(fields)
        if missing:
            raise DatasetManagerError(
                f"Manifest has missing columns: {sorted(missing)}"
            )
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            replacement = transform(dict(row))
            if replacement is not None:
                writer.writerow(replacement)
    os.replace(temporary, manifest)


def rebuild_summary(dataset: Path, classes: list[dict]) -> None:
    """Rebuild the small manifest-derived summary after a metadata change."""
    paths = metadata_paths(dataset)
    split_counts = {"train": 0, "val": 0, "test": 0}
    split_class_counts = {
        item["key"]: {"train": 0, "val": 0, "test": 0} for item in classes
    }
    family_ids: set[tuple[int, str]] = set()
    family_counts: dict[str, set[str]] = {item["key"]: set() for item in classes}
    total = 0
    class_by_id = {int(item["id"]): item for item in classes}
    with paths["manifest"].open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            class_id = int(row["class_id"])
            info = class_by_id.get(class_id)
            if info is None:
                raise DatasetManagerError(
                    f"Manifest contains class ID {class_id} absent from labels.json"
                )
            split = row["split"]
            if split not in split_counts:
                raise DatasetManagerError(f"Manifest has invalid split: {split!r}")
            total += 1
            split_counts[split] += 1
            split_class_counts[info["key"]][split] += 1
            family_id = row["family_id"]
            family_ids.add((class_id, family_id))
            family_counts[info["key"]].add(family_id)

    summary = {
        "class_count": len(classes),
        "image_count": total,
        "family_count": len(family_ids),
        "split_counts": split_counts,
        "family_counts_by_class": {
            key: len(values) for key, values in family_counts.items()
        },
        "split_class_counts": split_class_counts,
    }
    atomic_write_text(paths["summary"], json.dumps(summary, indent=2) + "\n")


def operation_log(command: str, details: dict) -> Path:
    path = LOG_ROOT / f"{timestamp()}_{command}.json"
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "command": command,
        **details,
    }
    atomic_write_text(path, json.dumps(payload, indent=2) + "\n")
    return path


def command_list(dataset: Path) -> int:
    _, classes = load_metadata(dataset)
    manifest = metadata_paths(dataset)["manifest"]
    counts = {int(item["id"]): 0 for item in classes}
    with manifest.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            class_id = int(row["class_id"])
            if class_id in counts:
                counts[class_id] += 1
    print(f"Dataset: {dataset}")
    print(f"{'ID':>3}  {'Class':<52} {'Manifest images':>15}")
    print("-" * 76)
    for item in classes:
        print(f"{int(item['id']):>3}  {item['display_name']:<52} {counts[int(item['id'])]:>15,}")
    print("-" * 76)
    print(f"Classes: {len(classes)} | Images: {sum(counts.values()):,}")
    return 0


def command_validate(dataset: Path) -> int:
    _, classes = load_metadata(dataset)
    paths = metadata_paths(dataset)
    validate_manifest_header(paths["manifest"])
    missing_directories = [
        item["relative_directory"]
        for item in classes
        if not (dataset / item["relative_directory"]).is_dir()
    ]
    if missing_directories:
        raise DatasetManagerError(
            f"Class directories missing: {missing_directories}"
        )

    checked = 0
    class_by_id = {int(item["id"]): item for item in classes}
    with paths["manifest"].open("r", newline="", encoding="utf-8") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            class_id = int(row["class_id"])
            info = class_by_id.get(class_id)
            if info is None:
                raise DatasetManagerError(f"Unknown class ID on manifest line {line_number}")
            expected = {
                "class_index": str(info["index"]),
                "class_key": str(info["key"]),
                "display_name": str(info["display_name"]),
                "crop": str(info["crop"]),
                "disease": str(info["disease"]),
            }
            if any(row[key] != value for key, value in expected.items()):
                raise DatasetManagerError(
                    f"Class metadata mismatch on manifest line {line_number}"
                )
            if not (dataset / row["relative_path"]).is_file():
                raise DatasetManagerError(
                    f"Missing manifest image on line {line_number}: {row['relative_path']}"
                )
            checked += 1
    print(f"Valid: {len(classes)} classes and {checked:,} manifest images")
    operation_log("validate", {"dataset": str(dataset), "checked_images": checked})
    return 0


def command_create(args: argparse.Namespace, dataset: Path) -> int:
    payload, classes = load_metadata(dataset)
    crop = slug(args.crop, "crop")
    disease = slug(args.disease, "disease")
    key = class_key(crop, disease)
    directory = relative_directory(crop, disease)
    if any(item["key"] == key for item in classes):
        raise DatasetManagerError(f"Class already exists: {key}")
    class_dir = dataset / directory
    if class_dir.exists():
        raise DatasetManagerError(f"Directory already exists: {class_dir}")
    new_class = {
        "id": len(classes) + 1,
        "index": len(classes),
        "key": key,
        "display_name": args.display_name.strip() if args.display_name else display_from(crop, disease),
        "crop": crop,
        "disease": disease,
        "relative_directory": directory,
    }
    details = {"dataset": str(dataset), "new_class": new_class, "applied": bool(args.apply)}
    if not args.apply:
        print("Dry run. Re-run with --apply to create this empty class:")
        print(json.dumps(new_class, indent=2))
        operation_log("create_dry_run", details)
        return 0
    class_dir.mkdir(parents=True)
    write_metadata(dataset, payload, [*classes, new_class])
    rebuild_summary(dataset, [*classes, new_class])
    log_path = operation_log("create", details)
    print(f"Created class {new_class['id']}: {new_class['display_name']}")
    print("The class is empty. Add images and rebuild the manifest before training.")
    print("Existing checkpoints no longer match this class list; train a new model.")
    print(f"Log: {log_path}")
    return 0


def command_rename(args: argparse.Namespace, dataset: Path) -> int:
    payload, classes = load_metadata(dataset)
    current = get_class(classes, args.class_id)
    crop = slug(args.crop, "crop") if args.crop else current["crop"]
    disease = slug(args.disease, "disease") if args.disease else current["disease"]
    key = class_key(crop, disease)
    directory = relative_directory(crop, disease)
    display_name = args.display_name.strip() if args.display_name else current["display_name"]
    if not display_name:
        raise DatasetManagerError("display name cannot be empty")
    if any(item["key"] == key and item is not current for item in classes):
        raise DatasetManagerError(f"Another class already has key: {key}")
    old_dir = dataset / current["relative_directory"]
    new_dir = dataset / directory
    if not old_dir.is_dir():
        raise DatasetManagerError(f"Class directory missing: {old_dir}")
    if old_dir != new_dir and new_dir.exists():
        raise DatasetManagerError(f"Target directory already exists: {new_dir}")
    replacement = {
        **current,
        "key": key,
        "display_name": display_name,
        "crop": crop,
        "disease": disease,
        "relative_directory": directory,
    }
    details = {
        "dataset": str(dataset), "old_class": current, "new_class": replacement,
        "manifest_rows": manifest_row_count(metadata_paths(dataset)["manifest"], args.class_id),
        "applied": bool(args.apply),
    }
    if not args.apply:
        print("Dry run. Re-run with --apply to rename this class:")
        print(json.dumps(details, indent=2))
        operation_log("rename_dry_run", details)
        return 0

    if old_dir != new_dir:
        new_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_dir), str(new_dir))
    updated_classes = [replacement if item is current else item for item in classes]
    def update_row(row: dict) -> dict:
        if int(row["class_id"]) == args.class_id:
            row.update({
                "class_index": str(replacement["index"]),
                "class_key": replacement["key"],
                "display_name": replacement["display_name"],
                "crop": replacement["crop"],
                "disease": replacement["disease"],
            })
            old_prefix = f"{current['relative_directory']}/"
            if not row["relative_path"].startswith(old_prefix):
                raise DatasetManagerError(
                    f"Manifest path does not match class directory: {row['relative_path']}"
                )
            row["relative_path"] = directory + row["relative_path"][len(current["relative_directory"]):]
        return row
    rewrite_manifest(metadata_paths(dataset)["manifest"], update_row)
    write_metadata(dataset, payload, updated_classes)
    rebuild_summary(dataset, updated_classes)
    log_path = operation_log("rename", details)
    print(f"Renamed class {args.class_id}: {replacement['display_name']}")
    print("Existing checkpoints may have stale class metadata; train a new model.")
    print(f"Log: {log_path}")
    return 0


def command_delete(args: argparse.Namespace, dataset: Path) -> int:
    payload, classes = load_metadata(dataset)
    current = get_class(classes, args.class_id)
    class_dir = dataset / current["relative_directory"]
    if not class_dir.is_dir():
        raise DatasetManagerError(f"Class directory missing: {class_dir}")
    paths = metadata_paths(dataset)
    manifest_rows = manifest_row_count(paths["manifest"], args.class_id)
    remaining_original = [item for item in classes if item is not current]
    if not remaining_original:
        raise DatasetManagerError("Refusing to delete the only class in the dataset")
    id_map = {
        int(item["id"]): index + 1
        for index, item in enumerate(remaining_original)
    }
    remaining = [
        {**item, "id": index + 1, "index": index}
        for index, item in enumerate(remaining_original)
    ]
    archive_dir = ARCHIVE_ROOT / f"{timestamp()}_delete" / current["relative_directory"]
    details = {
        "dataset": str(dataset), "deleted_class": current,
        "manifest_rows_removed": manifest_rows, "archive_directory": str(archive_dir),
        "id_remap": id_map, "applied": bool(args.apply),
    }
    if not args.apply:
        print("Dry run. Re-run with --apply to archive the class and update labels:")
        print(json.dumps(details, indent=2))
        operation_log("delete_dry_run", details)
        return 0

    archive_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(class_dir), str(archive_dir))
    def update_row(row: dict) -> dict | None:
        old_id = int(row["class_id"])
        if old_id == args.class_id:
            return None
        new_id = id_map[old_id]
        info = get_class(remaining, new_id)
        row.update({
            "class_id": str(new_id),
            "class_index": str(info["index"]),
            "class_key": info["key"],
            "display_name": info["display_name"],
            "crop": info["crop"],
            "disease": info["disease"],
        })
        return row
    rewrite_manifest(paths["manifest"], update_row)
    write_metadata(dataset, payload, remaining)
    rebuild_summary(dataset, remaining)
    log_path = operation_log("delete", details)
    print(f"Archived and removed class {args.class_id}: {current['display_name']}")
    print(f"Archive: {archive_dir}")
    print("Existing checkpoints no longer match the re-numbered class list; train a new model.")
    print(f"Log: {log_path}")
    return 0


def command_package(args: argparse.Namespace, dataset: Path) -> int:
    _, classes = load_metadata(dataset)
    package_dir = args.output_dir.resolve() if args.output_dir else (
        PACKAGE_ROOT / f"fieldcare_dataset_{timestamp()}"
    )
    if package_dir.exists():
        raise DatasetManagerError(f"Package directory already exists: {package_dir}")
    package_dir.mkdir(parents=True)
    archive_path = package_dir / "fieldcare_dataset.zip"
    files = [path for path in dataset.rglob("*") if path.is_file()]
    print(f"Packaging {len(files):,} files into {archive_path} ...")
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
    ) as archive:
        for index, path in enumerate(files, start=1):
            archive.write(path, Path("dataset") / path.relative_to(dataset))
            if index % 10_000 == 0:
                print(f"  Added {index:,}/{len(files):,} files")
    log_path = operation_log("package", {
        "dataset": str(dataset), "class_count": len(classes),
        "file_count": len(files), "archive": str(archive_path),
    })
    print(f"Created: {archive_path}")
    print(f"Log: {log_path}")
    return 0


def command_kaggle_upload(args: argparse.Namespace, dataset: Path) -> int:
    del dataset  # The package is the upload source for this operation.
    package_dir = args.package_dir.resolve()
    if not package_dir.is_dir():
        raise DatasetManagerError(f"Package directory not found: {package_dir}")
    archives = sorted(package_dir.glob("*.zip"))
    if len(archives) != 1:
        raise DatasetManagerError(
            f"Expected exactly one .zip archive in {package_dir}; found {len(archives)}"
        )
    owner = args.owner.strip()
    dataset_slug = slug(args.slug, "slug")
    title = args.title.strip()
    if not owner or not title:
        raise DatasetManagerError("--owner and --title are required")
    metadata_path = package_dir / "dataset-metadata.json"
    metadata = {
        "title": title,
        "id": f"{owner}/{dataset_slug}",
        "licenses": [{"name": "other"}],
        "isPrivate": not args.public,
    }
    atomic_write_text(metadata_path, json.dumps(metadata, indent=2) + "\n")
    command = ["kaggle", "datasets", "version" if args.version else "create", "-p", str(package_dir)]
    if args.version:
        command.extend(["-m", args.message])
    print("Running:", " ".join(command))
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise DatasetManagerError(
            "Kaggle CLI was not found. Install it with: python -m pip install kaggle"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise DatasetManagerError(f"Kaggle upload failed with exit code {exc.returncode}") from exc
    log_path = operation_log("kaggle_upload", {
        "package_directory": str(package_dir), "archive": str(archives[0]),
        "dataset_id": metadata["id"], "version": bool(args.version),
    })
    print(f"Kaggle upload completed: {metadata['id']}")
    print(f"Log: {log_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage the standardized FieldCare dataset.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List classes and manifest image counts")
    subparsers.add_parser("validate", help="Validate metadata, manifest, and image paths")

    create = subparsers.add_parser("create", help="Create an empty class")
    create.add_argument("--crop", required=True)
    create.add_argument("--disease", required=True)
    create.add_argument("--display-name")
    create.add_argument("--apply", action="store_true")

    rename = subparsers.add_parser("rename", help="Rename a class and its image directory")
    rename.add_argument("class_id", type=int)
    rename.add_argument("--crop")
    rename.add_argument("--disease")
    rename.add_argument("--display-name")
    rename.add_argument("--apply", action="store_true")

    delete = subparsers.add_parser("delete", help="Archive and remove a class")
    delete.add_argument("class_id", type=int)
    delete.add_argument("--apply", action="store_true")

    package = subparsers.add_parser("package", help="Create a ZIP archive for transfer or Kaggle")
    package.add_argument("--output-dir", type=Path)

    upload = subparsers.add_parser("kaggle-upload", help="Upload a package directory with the Kaggle CLI")
    upload.add_argument("--package-dir", type=Path, required=True)
    upload.add_argument("--owner", required=True)
    upload.add_argument("--slug", required=True)
    upload.add_argument("--title", required=True)
    upload.add_argument("--version", action="store_true", help="Create a new version instead of a dataset")
    upload.add_argument("--message", default="Update FieldCare dataset")
    upload.add_argument("--public", action="store_true", help="Make the Kaggle dataset public")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset = args.dataset.resolve()
    try:
        if args.command == "list":
            return command_list(dataset)
        if args.command == "validate":
            return command_validate(dataset)
        if args.command == "create":
            return command_create(args, dataset)
        if args.command == "rename":
            return command_rename(args, dataset)
        if args.command == "delete":
            return command_delete(args, dataset)
        if args.command == "package":
            return command_package(args, dataset)
        if args.command == "kaggle-upload":
            return command_kaggle_upload(args, dataset)
        raise DatasetManagerError(f"Unsupported command: {args.command}")
    except (DatasetManagerError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
