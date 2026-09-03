"""Evaluate a FieldCare v2 checkpoint on manifest or real-world images.

Examples:

    # Leakage-safe held-out test split from dataset_v2/manifest.csv
    python notebooks/evaluate_v2.py

    # Real-world images using real_world/label.txt ("filename class_id")
    python notebooks/evaluate_v2.py --source real-world

The newest v2 best checkpoint is selected from output/, outputs/, or
models/train_v2 unless --checkpoint is supplied. Results are written under
outputs/evaluate_v2 and include CSV/JSON/text metrics and confusion matrices.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    # Plotting is optional. Evaluation metrics and machine-readable confusion
    # matrices can still be produced in lightweight training environments.
    plt = None
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm

from predict_v2 import (
    PredictionError,
    VALID_EXTENSIONS,
    build_transform,
    find_latest_checkpoint,
    load_checkpoint,
    select_device,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "dataset_v2"
DEFAULT_REAL_WORLD = ROOT / "real_world"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "evaluate_v2"


class EvaluationError(RuntimeError):
    """Raised when an evaluation dataset is invalid."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class EvaluationRecord:
    path: Path
    class_index: int
    identifier: str


class ImageEvaluationDataset(Dataset):
    def __init__(self, records: list[EvaluationRecord], transform) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        try:
            with Image.open(record.path) as image:
                tensor = self.transform(image.convert("RGB"))
        except Exception as exc:
            raise EvaluationError(f"Could not read image: {record.path}") from exc
        return tensor, record.class_index, record.identifier


def manifest_records(
    dataset_root: Path,
    split: str,
    classes: list[dict],
    expected_manifest_sha256: str | None = None,
) -> list[EvaluationRecord]:
    manifest = dataset_root / "manifest.csv"
    if not manifest.is_file():
        raise EvaluationError(f"Manifest not found: {manifest}")
    if expected_manifest_sha256:
        actual_manifest_sha256 = file_sha256(manifest)
        if actual_manifest_sha256 != expected_manifest_sha256:
            raise EvaluationError(
                "The dataset manifest does not match the checkpoint used for "
                "training. This could invalidate the held-out evaluation.\n"
                f"Checkpoint manifest: {expected_manifest_sha256}\n"
                f"Dataset manifest   : {actual_manifest_sha256}"
            )
    class_keys = {int(item["index"]): item["key"] for item in classes}
    records = []
    with manifest.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"split", "class_index", "class_key", "relative_path"}
        if not required.issubset(reader.fieldnames or []):
            raise EvaluationError("dataset_v2 manifest has missing columns")
        for line_number, row in enumerate(reader, start=2):
            if row["split"] != split:
                continue
            try:
                class_index = int(row["class_index"])
            except ValueError as exc:
                raise EvaluationError(
                    f"Invalid class index on manifest line {line_number}"
                ) from exc
            if class_keys.get(class_index) != row["class_key"]:
                raise EvaluationError(
                    f"Checkpoint/manifest class mismatch on line {line_number}"
                )
            image_path = dataset_root / Path(row["relative_path"])
            if not image_path.is_file():
                raise EvaluationError(f"Manifest image missing: {image_path}")
            records.append(
                EvaluationRecord(image_path, class_index, row["relative_path"])
            )
    if not records:
        raise EvaluationError(f"No images found in manifest split: {split}")
    return records


def load_real_world_labels(path: Path, classes: list[dict]) -> dict[str, int]:
    if not path.is_file():
        raise EvaluationError(
            f"Real-world labels not found: {path}\n"
            "Expected each line to contain: relative/image.jpg CLASS_ID"
        )
    valid_ids = {int(item["id"]): int(item["index"]) for item in classes}
    labels = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(maxsplit=1)
        if len(parts) != 2:
            raise EvaluationError(
                f"Invalid real-world label line {line_number}: {raw_line}"
            )
        identifier, raw_id = parts
        try:
            class_id = int(raw_id)
        except ValueError as exc:
            raise EvaluationError(
                f"Invalid class ID on real-world label line {line_number}"
            ) from exc
        if class_id not in valid_ids:
            raise EvaluationError(
                f"Unknown v2 class ID {class_id} on label line {line_number}"
            )
        key = Path(identifier).as_posix().casefold()
        if key in labels:
            raise EvaluationError(f"Duplicate real-world label: {identifier}")
        labels[key] = valid_ids[class_id]
    if not labels:
        raise EvaluationError(f"No labels found in {path}")
    return labels


def real_world_records(
    real_world_root: Path,
    classes: list[dict],
    labels_path: Path | None = None,
    allow_unlabeled: bool = False,
) -> list[EvaluationRecord]:
    image_root = real_world_root / "images"
    labels_path = labels_path or (real_world_root / "label.txt")
    labels = load_real_world_labels(labels_path, classes)
    if not image_root.is_dir():
        raise EvaluationError(f"Real-world image directory not found: {image_root}")
    images = sorted(
        path for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )
    if not images:
        raise EvaluationError(f"No real-world images found under {image_root}")

    records = []
    used_labels = set()
    for image_path in images:
        relative = image_path.relative_to(image_root).as_posix()
        relative_key = relative.casefold()
        name_key = image_path.name.casefold()
        if relative_key in labels:
            label_key = relative_key
        elif name_key in labels:
            label_key = name_key
        else:
            if allow_unlabeled:
                print(f"WARNING: skipping image without a v2 label: {relative}")
                continue
            raise EvaluationError(f"No ground-truth label for: {relative}")
        used_labels.add(label_key)
        records.append(
            EvaluationRecord(image_path, labels[label_key], relative)
        )
    unused = sorted(set(labels) - used_labels)
    if unused:
        print(f"WARNING: {len(unused)} labels do not have matching images")
    if not records:
        raise EvaluationError(
            f"No labelled evaluation images were found using {labels_path}"
        )
    return records


def make_loader(
    records: list[EvaluationRecord],
    transform,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> DataLoader:
    kwargs = {
        "dataset": ImageEvaluationDataset(records, transform),
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def run_evaluation(model, loader, device: torch.device):
    predictions: list[int] = []
    targets: list[int] = []
    confidences: list[float] = []
    identifiers: list[str] = []
    total_seconds = 0.0

    model.eval()
    with torch.inference_mode():
        for images, batch_targets, batch_identifiers in tqdm(
            loader, desc="Evaluating"
        ):
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                probabilities = torch.softmax(model(images), dim=1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_seconds += time.perf_counter() - started

            confidence, prediction = probabilities.max(dim=1)
            predictions.extend(prediction.cpu().tolist())
            targets.extend(batch_targets.tolist())
            confidences.extend(confidence.cpu().tolist())
            identifiers.extend(batch_identifiers)
    return predictions, targets, confidences, identifiers, total_seconds


def save_results(
    output_dir: Path,
    checkpoint_path: Path,
    source: str,
    split: str,
    classes: list[dict],
    predictions: list[int],
    targets: list[int],
    confidences: list[float],
    identifiers: list[str],
    total_seconds: float,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    names = [item["display_name"] for item in classes]
    indices = list(range(len(classes)))
    accuracy = accuracy_score(targets, predictions)
    macro_f1 = f1_score(targets, predictions, average="macro", zero_division=0)
    weighted_f1 = f1_score(
        targets, predictions, average="weighted", zero_division=0
    )
    balanced = balanced_accuracy_score(targets, predictions)
    report = classification_report(
        targets,
        predictions,
        labels=indices,
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(targets, predictions, labels=indices)
    normalized = np.divide(
        matrix,
        matrix.sum(axis=1, keepdims=True),
        out=np.zeros_like(matrix, dtype=float),
        where=matrix.sum(axis=1, keepdims=True) != 0,
    )

    with (output_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = [
            "image", "actual_class_id", "actual_name", "predicted_class_id",
            "predicted_name", "confidence", "correct",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for identifier, target, prediction, confidence in zip(
            identifiers, targets, predictions, confidences
        ):
            writer.writerow(
                {
                    "image": identifier,
                    "actual_class_id": int(classes[target]["id"]),
                    "actual_name": names[target],
                    "predicted_class_id": int(classes[prediction]["id"]),
                    "predicted_name": names[prediction],
                    "confidence": round(confidence, 6),
                    "correct": target == prediction,
                }
            )

    with (output_dir / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *names])
        for name, row in zip(names, matrix):
            writer.writerow([name, *row.tolist()])
    np.save(output_dir / "confusion_matrix.npy", matrix)
    np.save(output_dir / "confusion_matrix_normalized.npy", normalized)
    (output_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    report_text = classification_report(
        targets,
        predictions,
        labels=indices,
        target_names=names,
        zero_division=0,
    )
    (output_dir / "classification_report.txt").write_text(
        report_text + "\n", encoding="utf-8"
    )
    with (output_dir / "classification_report.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])
        for name, values in report.items():
            if isinstance(values, dict):
                writer.writerow(
                    [
                        name,
                        values.get("precision", ""),
                        values.get("recall", ""),
                        values.get("f1-score", ""),
                        values.get("support", ""),
                    ]
                )

    plot_created = plt is not None
    if plot_created:
        figure, axes = plt.subplots(1, 2, figsize=(24, 10))
        count_image = axes[0].imshow(np.log1p(matrix), cmap="magma")
        figure.colorbar(count_image, ax=axes[0], fraction=0.046)
        normalized_image = axes[1].imshow(
            normalized, cmap="Blues", vmin=0, vmax=1
        )
        figure.colorbar(normalized_image, ax=axes[1], fraction=0.046)
        label = "Test" if source == "manifest" and split == "test" else "Evaluation"
        axes[0].set_title(f"{label} confusion matrix — log(1 + count)")
        axes[1].set_title(f"{label} confusion matrix — row normalized")
        for axis in axes:
            axis.set_xticks(indices)
            axis.set_yticks(indices)
            axis.set_xticklabels(names, rotation=90, fontsize=7)
            axis.set_yticklabels(names, fontsize=7)
            axis.set_xlabel("Predicted")
            axis.set_ylabel("Actual")
        figure.tight_layout()
        figure.savefig(
            output_dir / "confusion_matrix.png", dpi=180, bbox_inches="tight"
        )
        plt.close(figure)
    else:
        print(
            "WARNING: matplotlib is not installed; confusion_matrix.png was "
            "skipped. Install it with: python -m pip install matplotlib",
            file=sys.stderr,
        )

    summary = {
        "checkpoint": str(checkpoint_path),
        "source": source,
        "split": split if source == "manifest" else None,
        "image_count": len(targets),
        "correct": int(sum(a == b for a, b in zip(targets, predictions))),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "balanced_accuracy": float(balanced),
        "total_inference_seconds": total_seconds,
        "average_inference_ms": total_seconds / len(targets) * 1000,
        "confusion_matrix_plot_created": plot_created,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a FieldCare v2 checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source", choices=("manifest", "real-world"), default="manifest")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--real-world-dir", type=Path, default=DEFAULT_REAL_WORLD)
    parser.add_argument(
        "--real-world-labels",
        type=Path,
        help="Label file for real-world evaluation (default: <real-world-dir>/label.txt)",
    )
    parser.add_argument(
        "--allow-unlabeled",
        action="store_true",
        help="Skip real-world images that are absent from the selected label file",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-samples", type=int, help="Optional diagnostic limit")
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.batch_size < 1 or args.workers < 0:
            raise EvaluationError("Invalid batch size or worker count")
        if args.max_samples is not None and args.max_samples < 1:
            raise EvaluationError("--max-samples must be positive")
        device = select_device(args.device)
        checkpoint_path = (
            args.checkpoint.resolve() if args.checkpoint else find_latest_checkpoint()
        )
        model, checkpoint, classes = load_checkpoint(checkpoint_path, device)
        transform = build_transform(checkpoint)
        if args.source == "manifest":
            records = manifest_records(
                args.dataset.resolve(),
                args.split,
                classes,
                checkpoint.get("manifest_sha256"),
            )
        else:
            labels_path = (
                args.real_world_labels.resolve() if args.real_world_labels else None
            )
            records = real_world_records(
                args.real_world_dir.resolve(),
                classes,
                labels_path=labels_path,
                allow_unlabeled=args.allow_unlabeled,
            )
        if args.max_samples:
            records = records[: args.max_samples]

        print(f"Checkpoint: {checkpoint_path}")
        print(f"Device    : {device}")
        print(f"Source    : {args.source}")
        print(f"Images    : {len(records):,}")
        loader = make_loader(
            records, transform, args.batch_size, args.workers, device
        )
        predictions, targets, confidences, identifiers, seconds = run_evaluation(
            model, loader, device
        )
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
        summary = save_results(
            output_dir.resolve(), checkpoint_path, args.source, args.split,
            classes, predictions, targets, confidences, identifiers, seconds,
        )
        print("\n" + "=" * 72)
        print(f"Accuracy          : {summary['accuracy'] * 100:.2f}%")
        print(f"Macro F1          : {summary['macro_f1']:.4f}")
        print(f"Weighted F1       : {summary['weighted_f1']:.4f}")
        print(f"Balanced accuracy : {summary['balanced_accuracy'] * 100:.2f}%")
        print(f"Average inference : {summary['average_inference_ms']:.2f} ms/image")
        print(f"Results           : {output_dir.resolve()}")
        return 0
    except (
        OSError, RuntimeError, ValueError, PredictionError, EvaluationError,
        json.JSONDecodeError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    raise SystemExit(main())
