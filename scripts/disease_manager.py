"""Safely list, merge, or remove disease classes from the FieldCare dataset.

Examples (run from the repository root):

    python scripts/disease_manager.py list

    # Save a current dataset inventory under logs/disease_manager.
    python scripts/disease_manager.py snapshot

    # Backfill a log for the latest operation created by an older script run.
    python scripts/disease_manager.py backfill-log

    # Preview merging classes 2, 4, and 5 into class 2.
    python scripts/disease_manager.py merge 2 4 5 --target 2 \
        --name "Corn Leaf Spot"

    # Apply the merge after reviewing the preview.
    python scripts/disease_manager.py merge 2 4 5 --target 2 \
        --name "Corn Leaf Spot" --apply

    # Preview and then apply removal of class 5.
    python scripts/disease_manager.py delete 5
    python scripts/disease_manager.py delete 5 --apply

Changes are preview-only unless --apply is supplied. Deleted class folders are
moved to logs/_disease_manager_archive instead of being permanently erased.
Every applied operation also stores the old label file and an operation journal.
Human-readable and JSON logs are written to logs/disease_manager.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import uuid


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = ROOT_DIR / "dataset"
LOGS_ROOT = ROOT_DIR / "logs"
DEFAULT_LOGS_DIR = LOGS_ROOT / "disease_manager"
DEFAULT_ARCHIVE_DIR = LOGS_ROOT / "_disease_manager_archive"
LABEL_FILENAME = "label.txt"
ARCHIVE_DIRNAME = "_disease_manager_archive"
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


class DiseaseManagerError(RuntimeError):
    """Raised when an operation cannot be performed safely."""


def load_labels(label_file: Path) -> dict[int, str]:
    if not label_file.is_file():
        raise DiseaseManagerError(f"Label file not found: {label_file}")

    labels: dict[int, str] = {}
    for line_number, raw_line in enumerate(
        label_file.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not parts[0].isdigit():
            raise DiseaseManagerError(
                f"Invalid label line {line_number}: {raw_line!r}. "
                "Expected: <class_id> <class_name>"
            )

        class_id = int(parts[0])
        class_name = parts[1].strip()
        if class_id < 1 or not class_name:
            raise DiseaseManagerError(f"Invalid label line {line_number}: {raw_line!r}")
        if class_id in labels:
            raise DiseaseManagerError(f"Duplicate class ID in label file: {class_id}")
        labels[class_id] = class_name

    if not labels:
        raise DiseaseManagerError(f"No disease classes found in {label_file}")

    labels = dict(sorted(labels.items()))
    expected_ids = list(range(1, len(labels) + 1))
    if list(labels) != expected_ids:
        raise DiseaseManagerError(
            "Class IDs must be contiguous and start at 1 before running an "
            f"operation. Found {list(labels)}; expected {expected_ids}."
        )
    return labels


def image_count(class_dir: Path) -> int:
    if not class_dir.is_dir():
        return 0
    return sum(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        for path in class_dir.rglob("*")
    )


def all_file_count(class_dir: Path) -> int:
    if not class_dir.is_dir():
        return 0
    return sum(path.is_file() for path in class_dir.rglob("*"))


def dataset_snapshot(dataset_dir: Path, labels: dict[int, str]) -> dict:
    classes = []
    for class_id, class_name in labels.items():
        class_dir = dataset_dir / str(class_id)
        files = [path for path in class_dir.rglob("*") if path.is_file()]
        image_files = [path for path in files if path.suffix.lower() in IMAGE_EXTENSIONS]
        classes.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "folder": str(class_dir),
                "folder_exists": class_dir.is_dir(),
                "image_count": len(image_files),
                "file_count": len(files),
                "total_bytes": sum(path.stat().st_size for path in files),
            }
        )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": str(dataset_dir),
        "label_file": str(dataset_dir / LABEL_FILENAME),
        "class_count": len(classes),
        "image_count": sum(item["image_count"] for item in classes),
        "file_count": sum(item["file_count"] for item in classes),
        "total_bytes": sum(item["total_bytes"] for item in classes),
        "classes": classes,
    }


def unique_log_base(logs_dir: Path, suffix: str) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return logs_dir / f"{timestamp}_{suffix}"


def format_dataset_snapshot(snapshot: dict, title: str = "FIELDCARE DATASET LOG") -> str:
    lines = [
        title,
        "=" * 100,
        f"Generated at : {snapshot['generated_at']}",
        f"Dataset      : {snapshot['dataset_dir']}",
        f"Label file   : {snapshot['label_file']}",
        f"Classes      : {snapshot['class_count']:,}",
        f"Images       : {snapshot['image_count']:,}",
        f"All files    : {snapshot['file_count']:,}",
        f"Total bytes  : {snapshot['total_bytes']:,}",
        "",
        f"{'ID':>3}  {'Disease':<50} {'Images':>9} {'Files':>9} {'Bytes':>15}",
        "-" * 100,
    ]
    for item in snapshot["classes"]:
        lines.append(
            f"{item['class_id']:>3}  {item['class_name']:<50} "
            f"{item['image_count']:>9,} {item['file_count']:>9,} "
            f"{item['total_bytes']:>15,}"
        )
    lines.append("")
    return "\n".join(lines)


def write_dataset_snapshot_logs(logs_dir: Path, snapshot: dict) -> tuple[Path, Path]:
    base = unique_log_base(logs_dir, "dataset_snapshot")
    text_path = base.with_suffix(".log")
    json_path = base.with_suffix(".json")
    text_path.write_text(format_dataset_snapshot(snapshot), encoding="utf-8")
    json_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return text_path, json_path


def write_operation_logs(
    logs_dir: Path,
    operation: dict,
    before: dict,
    after: dict,
    operation_dir: Path,
) -> tuple[Path, Path]:
    operation_name = operation["operation"]
    base = unique_log_base(logs_dir, f"disease_{operation_name}")
    text_path = base.with_suffix(".log")
    json_path = base.with_suffix(".json")
    payload = {
        "operation": operation,
        "dataset_before": before,
        "dataset_after": after,
        "recovery_archive": str(operation_dir),
    }

    lines = [
        "FIELDCARE DISEASE OPERATION LOG",
        "=" * 100,
        f"Operation        : {operation_name}",
        f"Completed at     : {operation['created_at']}",
        f"Dataset          : {after['dataset_dir']}",
        f"Recovery archive : {operation_dir}",
        "",
    ]
    if operation_name == "merge":
        lines.extend(
            [
                f"Merged IDs       : {operation['merged_class_ids']}",
                f"Target ID before : {operation['target_class_id_before_renumbering']}",
                f"Target ID after  : {operation['target_class_id_after_renumbering']}",
                f"Merged name      : {operation['merged_name']}",
                f"Moved files      : {operation['moved_file_count']:,}",
            ]
        )
    else:
        lines.extend(
            [
                f"Deleted IDs      : {operation['deleted_class_ids']}",
                f"Deleted labels   : {operation['deleted_labels']}",
            ]
        )
    lines.extend(
        [
            f"ID mapping       : {operation['old_to_new_class_ids']}",
            "",
            "BEFORE",
            format_dataset_snapshot(before, title="DATASET BEFORE OPERATION"),
            "AFTER",
            format_dataset_snapshot(after, title="DATASET AFTER OPERATION"),
        ]
    )

    text_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return text_path, json_path


def expected_labels_after_operation(
    labels_before: dict[int, str], operation: dict
) -> dict[int, str]:
    mapping = {int(old): int(new) for old, new in operation["old_to_new_class_ids"].items()}
    if operation["operation"] == "merge":
        target = int(operation["target_class_id_before_renumbering"])
        merged_name = operation["merged_name"]
        return {
            new_id: merged_name if old_id == target else labels_before[old_id]
            for old_id, new_id in mapping.items()
        }
    return {new_id: labels_before[old_id] for old_id, new_id in mapping.items()}


def snapshot_class_entry(
    class_id: int,
    class_name: str,
    folder: Path,
    image_count_value: int,
    file_count_value: int,
    total_bytes: int,
) -> dict:
    return {
        "class_id": class_id,
        "class_name": class_name,
        "folder": str(folder),
        "folder_exists": True,
        "image_count": image_count_value,
        "file_count": file_count_value,
        "total_bytes": total_bytes,
    }


def finish_reconstructed_snapshot(
    dataset_dir: Path, classes: list[dict], generated_at: str
) -> dict:
    classes.sort(key=lambda item: item["class_id"])
    return {
        "generated_at": generated_at,
        "dataset_dir": str(dataset_dir),
        "label_file": str(dataset_dir / LABEL_FILENAME),
        "class_count": len(classes),
        "image_count": sum(item["image_count"] for item in classes),
        "file_count": sum(item["file_count"] for item in classes),
        "total_bytes": sum(item["total_bytes"] for item in classes),
        "classes": classes,
    }


def reconstruct_snapshot_before_operation(
    dataset_dir: Path,
    operation_dir: Path,
    operation: dict,
    labels_before: dict[int, str],
    after: dict,
) -> dict:
    """Reconstruct counts for a completed legacy operation journal."""
    mapping = {int(old): int(new) for old, new in operation["old_to_new_class_ids"].items()}
    after_by_id = {item["class_id"]: item for item in after["classes"]}
    classes: list[dict] = []

    if operation["operation"] == "merge":
        target = int(operation["target_class_id_before_renumbering"])
        merged_ids = {int(class_id) for class_id in operation["merged_class_ids"]}
        source_ids = merged_ids - {target}
        source_stats = {
            class_id: {"images": 0, "files": 0, "bytes": 0}
            for class_id in source_ids
        }

        for move in operation.get("moved_files", []):
            source_id = int(Path(move["from"]).parts[0])
            current_path = dataset_dir / Path(move["to_after_renumbering"])
            if source_id not in source_stats or not current_path.is_file():
                raise DiseaseManagerError(
                    f"Cannot reconstruct legacy merge log; missing moved file: {current_path}"
                )
            stats = source_stats[source_id]
            stats["files"] += 1
            stats["images"] += current_path.suffix.lower() in IMAGE_EXTENSIONS
            stats["bytes"] += current_path.stat().st_size

        for old_id, class_name in labels_before.items():
            old_folder = dataset_dir / str(old_id)
            if old_id in source_ids:
                stats = source_stats[old_id]
                classes.append(
                    snapshot_class_entry(
                        old_id,
                        class_name,
                        old_folder,
                        stats["images"],
                        stats["files"],
                        stats["bytes"],
                    )
                )
            elif old_id == target:
                current = after_by_id[mapping[old_id]]
                source_images = sum(item["images"] for item in source_stats.values())
                source_files = sum(item["files"] for item in source_stats.values())
                source_bytes = sum(item["bytes"] for item in source_stats.values())
                classes.append(
                    snapshot_class_entry(
                        old_id,
                        class_name,
                        old_folder,
                        current["image_count"] - source_images,
                        current["file_count"] - source_files,
                        current["total_bytes"] - source_bytes,
                    )
                )
            else:
                current = after_by_id[mapping[old_id]]
                classes.append(
                    snapshot_class_entry(
                        old_id,
                        class_name,
                        old_folder,
                        current["image_count"],
                        current["file_count"],
                        current["total_bytes"],
                    )
                )
    else:
        deleted_ids = {int(class_id) for class_id in operation["deleted_class_ids"]}
        for old_id, class_name in labels_before.items():
            old_folder = dataset_dir / str(old_id)
            if old_id in deleted_ids:
                archived = operation_dir / "classes" / str(old_id)
                files = [path for path in archived.rglob("*") if path.is_file()]
                classes.append(
                    snapshot_class_entry(
                        old_id,
                        class_name,
                        old_folder,
                        sum(path.suffix.lower() in IMAGE_EXTENSIONS for path in files),
                        len(files),
                        sum(path.stat().st_size for path in files),
                    )
                )
            else:
                current = after_by_id[mapping[old_id]]
                classes.append(
                    snapshot_class_entry(
                        old_id,
                        class_name,
                        old_folder,
                        current["image_count"],
                        current["file_count"],
                        current["total_bytes"],
                    )
                )

    return finish_reconstructed_snapshot(
        dataset_dir, classes, operation.get("created_at", after["generated_at"])
    )


def backfill_latest_operation_log(
    dataset_dir: Path,
    logs_dir: Path,
    archive_root: Path,
    current_labels: dict[int, str],
) -> None:
    journals = sorted(archive_root.glob("*/operation.json"), reverse=True)
    if not journals:
        raise DiseaseManagerError(f"No operation journals found under {archive_root}")

    already_logged: set[str] = set()
    if logs_dir.exists():
        for log_path in logs_dir.glob("*_disease_*.json"):
            try:
                payload = json.loads(log_path.read_text(encoding="utf-8"))
                already_logged.add(str(Path(payload["recovery_archive"]).resolve()))
            except (KeyError, OSError, ValueError, TypeError):
                continue

    after = dataset_snapshot(dataset_dir, current_labels)
    for journal_path in journals:
        operation_dir = journal_path.parent
        if str(operation_dir.resolve()) in already_logged:
            continue
        operation = json.loads(journal_path.read_text(encoding="utf-8"))
        labels_before = load_labels(operation_dir / "label_before.txt")
        if expected_labels_after_operation(labels_before, operation) != current_labels:
            continue

        before = reconstruct_snapshot_before_operation(
            dataset_dir, operation_dir, operation, labels_before, after
        )
        text_log, json_log = write_operation_logs(
            logs_dir, operation, before, after, operation_dir
        )
        print(f"Backfilled operation: {operation['operation']}")
        print(f"Text log: {text_log}")
        print(f"JSON log: {json_log}")
        return

    print("No unlogged operation matching the current dataset state was found.")


def validate_class_directories(dataset_dir: Path, class_ids: list[int]) -> None:
    missing = [class_id for class_id in class_ids if not (dataset_dir / str(class_id)).is_dir()]
    if missing:
        raise DiseaseManagerError(
            "Missing class folder(s): " + ", ".join(str(class_id) for class_id in missing)
        )


def write_labels_atomic(label_file: Path, labels: dict[int, str]) -> None:
    temporary = label_file.with_name(f".{label_file.name}.{uuid.uuid4().hex}.tmp")
    text = "".join(f"{class_id} {name}\n" for class_id, name in labels.items())
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, label_file)
    finally:
        if temporary.exists():
            temporary.unlink()


def unique_destination(destination: Path, source_class_id: int) -> Path:
    if not destination.exists():
        return destination

    candidate = destination.with_name(
        f"{destination.stem}__from_class_{source_class_id}{destination.suffix}"
    )
    counter = 2
    while candidate.exists():
        candidate = destination.with_name(
            f"{destination.stem}__from_class_{source_class_id}_{counter}"
            f"{destination.suffix}"
        )
        counter += 1
    return candidate


def remove_empty_directories(root: Path) -> None:
    if not root.exists():
        return
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        directory.rmdir()
    root.rmdir()


def build_survivor_mapping(
    labels: dict[int, str], removed_ids: set[int]
) -> tuple[list[int], dict[int, int]]:
    survivors = [class_id for class_id in labels if class_id not in removed_ids]
    mapping = {old_id: new_id for new_id, old_id in enumerate(survivors, start=1)}
    return survivors, mapping


def make_operation_dir(archive_dir: Path, operation: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    operation_dir = archive_dir / f"{timestamp}_{operation}"
    counter = 2
    while operation_dir.exists():
        operation_dir = archive_dir / f"{timestamp}_{operation}_{counter}"
        counter += 1
    operation_dir.mkdir(parents=True)
    return operation_dir


def confirm(operation_word: str, assume_yes: bool) -> None:
    if assume_yes:
        return
    print()
    answer = input(f"Type {operation_word} to continue: ").strip()
    if answer != operation_word:
        raise DiseaseManagerError("Operation cancelled; nothing was changed.")


def renumber_class_directories(
    dataset_dir: Path,
    mapping: dict[int, int],
    token: str,
    current_paths: dict[int, Path],
) -> None:
    """Rename all surviving folders through unique temporary names.

    ``current_paths`` is updated after every rename so rollback still has an
    accurate location if an exception occurs halfway through either phase.
    """
    for old_id in mapping:
        current = current_paths[old_id]
        temporary = dataset_dir / f"__disease_manager_{token}_{old_id}"
        if temporary.exists():
            raise DiseaseManagerError(f"Temporary path already exists: {temporary}")
        current.rename(temporary)
        current_paths[old_id] = temporary

    for old_id, new_id in mapping.items():
        current = current_paths[old_id]
        destination = dataset_dir / str(new_id)
        if destination.exists():
            raise DiseaseManagerError(f"Cannot renumber; path already exists: {destination}")
        current.rename(destination)
        current_paths[old_id] = destination


def rollback_renumbering(
    dataset_dir: Path,
    current_paths: dict[int, Path],
    token: str,
) -> None:
    rollback_paths: dict[int, Path] = {}

    for old_id, current in current_paths.items():
        original = dataset_dir / str(old_id)
        if current == original:
            continue
        if not current.exists():
            continue
        temporary = dataset_dir / f"__disease_manager_rollback_{token}_{old_id}"
        current.rename(temporary)
        rollback_paths[old_id] = temporary

    for old_id, temporary in rollback_paths.items():
        original = dataset_dir / str(old_id)
        if original.exists():
            raise DiseaseManagerError(
                f"Automatic rollback could not restore {original}; {temporary} is preserved."
            )
        temporary.rename(original)
        current_paths[old_id] = original


def print_class_table(dataset_dir: Path, labels: dict[int, str]) -> None:
    print()
    print(f"{'ID':>3}  {'Disease':<50} {'Images':>8} {'All files':>10}")
    print("-" * 76)
    for class_id, class_name in labels.items():
        class_dir = dataset_dir / str(class_id)
        marker = "" if class_dir.is_dir() else "  [MISSING FOLDER]"
        print(
            f"{class_id:>3}  {class_name:<50} "
            f"{image_count(class_dir):>8,} {all_file_count(class_dir):>10,}{marker}"
        )
    print()


def preview_final_classes(final_labels: dict[int, str]) -> None:
    print("Resulting class mapping:")
    for class_id, name in final_labels.items():
        print(f"  {class_id:>3}  {name}")


def merge_classes(args: argparse.Namespace, dataset_dir: Path, labels: dict[int, str]) -> None:
    class_ids = list(dict.fromkeys(args.class_ids))
    if len(class_ids) < 2:
        raise DiseaseManagerError("Merge requires at least two different class IDs.")

    unknown = [class_id for class_id in class_ids if class_id not in labels]
    if unknown:
        raise DiseaseManagerError(f"Unknown class ID(s): {unknown}")

    target_id = args.target if args.target is not None else class_ids[0]
    if target_id not in class_ids:
        raise DiseaseManagerError("--target must be one of the class IDs being merged.")

    validate_class_directories(dataset_dir, class_ids)
    source_ids = [class_id for class_id in class_ids if class_id != target_id]
    removed_ids = set(source_ids)
    survivors, mapping = build_survivor_mapping(labels, removed_ids)
    merged_name = args.name.strip() if args.name else labels[target_id]
    if not merged_name:
        raise DiseaseManagerError("The merged disease name cannot be empty.")

    final_labels = {
        mapping[old_id]: merged_name if old_id == target_id else labels[old_id]
        for old_id in survivors
    }

    print("MERGE PREVIEW")
    print("-" * 76)
    for class_id in class_ids:
        print(
            f"  {class_id:>3}  {labels[class_id]:<50} "
            f"{image_count(dataset_dir / str(class_id)):>8,} images"
        )
    print(f"Target class: {target_id}")
    print(f"Merged name : {merged_name}")
    print()
    preview_final_classes(final_labels)

    if not args.apply:
        print()
        print("Preview only: nothing was changed.")
        print("Run the same command with --apply to perform this merge.")
        return

    confirm("MERGE", args.yes)

    label_file = dataset_dir / LABEL_FILENAME
    before_snapshot = dataset_snapshot(dataset_dir, labels)
    operation_dir = make_operation_dir(args.archive_dir, "merge")
    shutil.copy2(label_file, operation_dir / "label_before.txt")
    token = uuid.uuid4().hex
    target_dir = dataset_dir / str(target_id)
    moved_files: list[tuple[Path, Path]] = []
    current_paths = {old_id: dataset_dir / str(old_id) for old_id in mapping}

    try:
        for source_id in source_ids:
            source_dir = dataset_dir / str(source_id)
            files = sorted(path for path in source_dir.rglob("*") if path.is_file())
            for source_path in files:
                relative_path = source_path.relative_to(source_dir)
                destination = unique_destination(target_dir / relative_path, source_id)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source_path), str(destination))
                moved_files.append((source_path, destination))
            remove_empty_directories(source_dir)

        renumber_class_directories(dataset_dir, mapping, token, current_paths)
        write_labels_atomic(label_file, final_labels)

        journal = {
            "operation": "merge",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "merged_class_ids": class_ids,
            "target_class_id_before_renumbering": target_id,
            "target_class_id_after_renumbering": mapping[target_id],
            "merged_name": merged_name,
            "old_to_new_class_ids": mapping,
            "moved_file_count": len(moved_files),
            "moved_files": [
                {
                    "from": str(source.relative_to(dataset_dir)),
                    "to_before_renumbering": str(destination.relative_to(dataset_dir)),
                    "to_after_renumbering": str(
                        Path(str(mapping[target_id])) / destination.relative_to(target_dir)
                    ),
                }
                for source, destination in moved_files
            ],
        }
        (operation_dir / "operation.json").write_text(
            json.dumps(journal, indent=2), encoding="utf-8"
        )
        after_snapshot = dataset_snapshot(dataset_dir, final_labels)
    except Exception:
        rollback_renumbering(dataset_dir, current_paths, token)
        for source, destination in reversed(moved_files):
            if destination.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(destination), str(source))
        shutil.copy2(operation_dir / "label_before.txt", label_file)
        raise

    try:
        text_log, json_log = write_operation_logs(
            args.logs_dir, journal, before_snapshot, after_snapshot, operation_dir
        )
    except OSError as error:
        text_log = json_log = None
        print(f"WARNING: Dataset changed, but separate log creation failed: {error}")

    print()
    print(f"Merge complete. Moved {len(moved_files):,} file(s).")
    print(f"New target class ID: {mapping[target_id]}")
    print(f"Operation journal: {operation_dir / 'operation.json'}")
    if text_log and json_log:
        print(f"Text log        : {text_log}")
        print(f"JSON log        : {json_log}")
    print("Retrain the model after changing the dataset classes.")


def delete_classes(args: argparse.Namespace, dataset_dir: Path, labels: dict[int, str]) -> None:
    delete_ids = list(dict.fromkeys(args.class_ids))
    unknown = [class_id for class_id in delete_ids if class_id not in labels]
    if unknown:
        raise DiseaseManagerError(f"Unknown class ID(s): {unknown}")
    if len(delete_ids) >= len(labels):
        raise DiseaseManagerError("Refusing to delete every disease class.")

    validate_class_directories(dataset_dir, delete_ids)
    removed_ids = set(delete_ids)
    survivors, mapping = build_survivor_mapping(labels, removed_ids)
    final_labels = {mapping[old_id]: labels[old_id] for old_id in survivors}

    print("DELETE PREVIEW")
    print("-" * 76)
    for class_id in delete_ids:
        print(
            f"  {class_id:>3}  {labels[class_id]:<50} "
            f"{image_count(dataset_dir / str(class_id)):>8,} images"
        )
    print()
    print("Deleted class folders will be archived, not permanently erased.")
    print()
    preview_final_classes(final_labels)

    if not args.apply:
        print()
        print("Preview only: nothing was changed.")
        print("Run the same command with --apply to perform this deletion.")
        return

    confirm("DELETE", args.yes)

    label_file = dataset_dir / LABEL_FILENAME
    before_snapshot = dataset_snapshot(dataset_dir, labels)
    operation_dir = make_operation_dir(args.archive_dir, "delete")
    archived_classes_dir = operation_dir / "classes"
    archived_classes_dir.mkdir()
    shutil.copy2(label_file, operation_dir / "label_before.txt")
    token = uuid.uuid4().hex
    archived_paths: list[tuple[Path, Path]] = []
    current_paths = {old_id: dataset_dir / str(old_id) for old_id in mapping}

    try:
        for class_id in delete_ids:
            source = dataset_dir / str(class_id)
            destination = archived_classes_dir / str(class_id)
            shutil.move(str(source), str(destination))
            archived_paths.append((source, destination))

        renumber_class_directories(dataset_dir, mapping, token, current_paths)
        write_labels_atomic(label_file, final_labels)

        journal = {
            "operation": "delete",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "deleted_class_ids": delete_ids,
            "deleted_labels": {class_id: labels[class_id] for class_id in delete_ids},
            "old_to_new_class_ids": mapping,
            "archived_class_folders": [
                str(destination)
                for _, destination in archived_paths
            ],
        }
        (operation_dir / "operation.json").write_text(
            json.dumps(journal, indent=2), encoding="utf-8"
        )
        after_snapshot = dataset_snapshot(dataset_dir, final_labels)
    except Exception:
        rollback_renumbering(dataset_dir, current_paths, token)
        for source, destination in reversed(archived_paths):
            if destination.exists():
                shutil.move(str(destination), str(source))
        shutil.copy2(operation_dir / "label_before.txt", label_file)
        raise

    try:
        text_log, json_log = write_operation_logs(
            args.logs_dir, journal, before_snapshot, after_snapshot, operation_dir
        )
    except OSError as error:
        text_log = json_log = None
        print(f"WARNING: Dataset changed, but separate log creation failed: {error}")

    print()
    print(f"Deleted {len(delete_ids)} disease class(es) from the active dataset.")
    print(f"Recoverable archive: {operation_dir}")
    if text_log and json_log:
        print(f"Text log          : {text_log}")
        print(f"JSON log          : {json_log}")
    print("Retrain the model after changing the dataset classes.")


def save_dataset_snapshot(dataset_dir: Path, logs_dir: Path, labels: dict[int, str]) -> None:
    snapshot = dataset_snapshot(dataset_dir, labels)
    text_log, json_log = write_dataset_snapshot_logs(logs_dir, snapshot)
    print(format_dataset_snapshot(snapshot), end="")
    print(f"Text log: {text_log}")
    print(f"JSON log: {json_log}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List, merge, or safely delete FieldCare disease classes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Dataset directory (default: {DEFAULT_DATASET_DIR})",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=DEFAULT_LOGS_DIR,
        help=f"Separate log directory (default: {DEFAULT_LOGS_DIR})",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=DEFAULT_ARCHIVE_DIR,
        help=f"Recovery archive directory (default: {DEFAULT_ARCHIVE_DIR})",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List disease IDs, names, and image counts.")
    subparsers.add_parser(
        "snapshot", help="Save current dataset counts as text and JSON logs."
    )
    subparsers.add_parser(
        "backfill-log",
        help="Create logs for the latest matching legacy operation journal.",
    )

    merge_parser = subparsers.add_parser(
        "merge", help="Merge two or more disease classes into one class."
    )
    merge_parser.add_argument(
        "class_ids", type=int, nargs="+", help="Two or more class IDs to merge."
    )
    merge_parser.add_argument(
        "--target",
        type=int,
        help="Class ID to keep as the target (default: first supplied ID).",
    )
    merge_parser.add_argument(
        "--name", help="New merged disease name (default: target's current name)."
    )
    merge_parser.add_argument(
        "--apply", action="store_true", help="Apply the operation; otherwise preview only."
    )
    merge_parser.add_argument(
        "--yes", action="store_true", help="Skip the typed confirmation (requires --apply)."
    )

    delete_parser = subparsers.add_parser(
        "delete", help="Remove one or more disease classes from the active dataset."
    )
    delete_parser.add_argument(
        "class_ids", type=int, nargs="+", help="Class ID(s) to remove."
    )
    delete_parser.add_argument(
        "--apply", action="store_true", help="Apply the operation; otherwise preview only."
    )
    delete_parser.add_argument(
        "--yes", action="store_true", help="Skip the typed confirmation (requires --apply)."
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    dataset_dir = args.dataset_dir.resolve()
    args.logs_dir = args.logs_dir.resolve()
    args.archive_dir = args.archive_dir.resolve()
    label_file = dataset_dir / LABEL_FILENAME

    try:
        labels = load_labels(label_file)
        if args.command == "list":
            print_class_table(dataset_dir, labels)
        elif args.command == "snapshot":
            save_dataset_snapshot(dataset_dir, args.logs_dir, labels)
        elif args.command == "backfill-log":
            backfill_latest_operation_log(
                dataset_dir, args.logs_dir, args.archive_dir, labels
            )
        elif args.command == "merge":
            merge_classes(args, dataset_dir, labels)
        elif args.command == "delete":
            delete_classes(args, dataset_dir, labels)
        else:
            parser.error(f"Unknown command: {args.command}")
    except DiseaseManagerError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    except (OSError, shutil.Error) as error:
        print(f"FILESYSTEM ERROR: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
