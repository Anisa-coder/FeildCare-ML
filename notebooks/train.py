"""Train EfficientNet-B3 using standardized dataset metadata and manifest."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "dataset"
DEFAULT_MODEL_ROOT = ROOT / "models" / "train"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "train"
DEFAULT_LOG_ROOT = ROOT / "logs" / "train"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class ClassInfo:
    id: int
    index: int
    key: str
    display_name: str
    crop: str
    disease: str
    relative_directory: str


@dataclass(frozen=True)
class Sample:
    path: Path
    class_index: int
    family_id: str


class TrainingError(RuntimeError):
    """Raised when training inputs or configuration are invalid."""


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fieldcare.train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def load_classes(dataset_root: Path) -> list[ClassInfo]:
    labels_path = dataset_root / "labels.json"
    if not labels_path.is_file():
        raise TrainingError(
            f"Missing {labels_path}. Build or restore the standardized dataset first."
        )
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    raw_classes = payload.get("classes")
    if not isinstance(raw_classes, list) or not raw_classes:
        raise TrainingError(f"No classes found in {labels_path}")

    classes = [ClassInfo(**item) for item in raw_classes]
    expected_ids = list(range(1, len(classes) + 1))
    expected_indices = list(range(len(classes)))
    if [item.id for item in classes] != expected_ids:
        raise TrainingError("Class IDs must be contiguous and start at 1")
    if [item.index for item in classes] != expected_indices:
        raise TrainingError("Class indices must be contiguous and start at 0")
    if len({item.key for item in classes}) != len(classes):
        raise TrainingError("Duplicate class keys in labels.json")

    for item in classes:
        class_dir = dataset_root / Path(item.relative_directory)
        if not class_dir.is_dir():
            raise TrainingError(f"Class directory missing: {class_dir}")
    return classes


def load_manifest(
    dataset_root: Path, classes: list[ClassInfo]
) -> tuple[dict[str, list[Sample]], dict]:
    manifest_path = dataset_root / "manifest.csv"
    if not manifest_path.is_file():
        raise TrainingError(
            f"Missing {manifest_path}. Build or restore the standardized dataset first."
        )

    class_by_id = {item.id: item for item in classes}
    samples = {split: [] for split in SPLITS}
    per_class_split = defaultdict(Counter)
    family_splits: dict[tuple[int, str], set[str]] = defaultdict(set)
    missing_paths: list[str] = []

    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "split", "class_id", "class_index", "class_key", "family_id",
            "relative_path",
        }
        if not required.issubset(reader.fieldnames or []):
            raise TrainingError(
                f"Manifest columns missing: {sorted(required - set(reader.fieldnames or []))}"
            )

        for line_number, row in enumerate(reader, start=2):
            split = row["split"]
            if split not in samples:
                raise TrainingError(f"Invalid split on manifest line {line_number}: {split}")
            try:
                class_id = int(row["class_id"])
                class_index = int(row["class_index"])
            except ValueError as exc:
                raise TrainingError(
                    f"Invalid class number on manifest line {line_number}"
                ) from exc

            class_info = class_by_id.get(class_id)
            if class_info is None:
                raise TrainingError(f"Unknown class ID on line {line_number}: {class_id}")
            if class_index != class_info.index or row["class_key"] != class_info.key:
                raise TrainingError(f"Class metadata mismatch on manifest line {line_number}")

            relative_path = Path(row["relative_path"])
            image_path = dataset_root / relative_path
            if image_path.suffix.lower() not in VALID_EXTENSIONS:
                raise TrainingError(f"Unsupported image extension: {image_path}")
            if not image_path.is_file() and len(missing_paths) < 20:
                missing_paths.append(relative_path.as_posix())

            family_id = row["family_id"]
            samples[split].append(Sample(image_path, class_index, family_id))
            per_class_split[class_info.key][split] += 1
            family_splits[(class_id, family_id)].add(split)

    if missing_paths:
        raise TrainingError(f"Manifest references missing images: {missing_paths}")
    for split in SPLITS:
        if not samples[split]:
            raise TrainingError(f"Manifest split is empty: {split}")

    leakage = [key for key, split_set in family_splits.items() if len(split_set) > 1]
    if leakage:
        raise TrainingError(
            f"Augmentation-family leakage found in manifest: {len(leakage)} groups"
        )

    for item in classes:
        for split in SPLITS:
            if per_class_split[item.key][split] == 0:
                raise TrainingError(f"{item.display_name} has no {split} samples")

    summary = {
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "total_images": sum(len(items) for items in samples.values()),
        "split_counts": {split: len(samples[split]) for split in SPLITS},
        "family_count": len(family_splits),
        "family_leakage_count": len(leakage),
        "per_class_split": {
            item.key: {
                split: per_class_split[item.key][split] for split in SPLITS
            }
            for item in classes
        },
    }
    return samples, summary


class FieldCareV2Dataset(Dataset):
    def __init__(self, samples: list[Sample], transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        try:
            with Image.open(sample.path) as image:
                image = image.convert("RGB")
                image = self.transform(image)
        except Exception as exc:
            raise TrainingError(f"Could not read image: {sample.path}") from exc
        return image, sample.class_index


def build_transforms(image_size: int):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(
                brightness=0.12, contrast=0.12, saturation=0.12, hue=0.02
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


def make_loader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    device: torch.device,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def effective_class_weights(
    train_samples: list[Sample], num_classes: int, beta: float
) -> tuple[torch.Tensor, list[int]]:
    counts = np.bincount(
        [sample.class_index for sample in train_samples], minlength=num_classes
    ).astype(np.float64)
    if np.any(counts == 0):
        raise TrainingError(f"Empty training classes: {np.where(counts == 0)[0].tolist()}")
    effective = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / effective
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32), counts.astype(int).tolist()


def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    loss_sum = 0.0
    count = 0
    predictions: list[int] = []
    targets_all: list[int] = []
    with torch.inference_mode():
        for images, targets in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
                loss = criterion(logits, targets)
            loss_sum += loss.item() * targets.size(0)
            count += targets.size(0)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets_all.extend(targets.cpu().tolist())

    prediction_array = np.asarray(predictions)
    target_array = np.asarray(targets_all)
    return {
        "loss": loss_sum / count,
        "accuracy": float((prediction_array == target_array).mean()),
        "macro_f1": float(
            f1_score(target_array, prediction_array, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(target_array, prediction_array)
        ),
        "predictions": predictions,
        "targets": targets_all,
    }


def train_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    loss_sum = 0.0
    correct = 0
    count = 0
    progress = tqdm(loader, desc="Training", leave=False)
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        count += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return {"loss": loss_sum / count, "accuracy": correct / count}


def checkpoint_payload(
    model,
    classes,
    args,
    manifest_summary,
    epoch,
    val_metrics,
):
    return {
        "model_state_dict": model.state_dict(),
        "architecture": "efficientnet_b3",
        "num_classes": len(classes),
        "classes": [asdict(item) for item in classes],
        "labels": {item.id: item.display_name for item in classes},
        "image_size": args.image_size,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "epoch": epoch,
        "validation_metrics": val_metrics,
        "manifest_sha256": manifest_summary["manifest_sha256"],
        "seed": args.seed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train EfficientNet-B3 on the standardized FieldCare dataset."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--run-name", help="Optional run directory name")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--image-size", type=int, default=300)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--class-balance-beta", type=float, default=0.9999)
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--allow-cpu", action="store_true",
        help="Allow training without CUDA (not recommended for this dataset)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate metadata and print split counts without creating a model",
    )
    return parser


def validate_args(args) -> None:
    if args.epochs < 1:
        raise TrainingError("--epochs must be at least 1")
    if not 0 <= args.warmup_epochs < args.epochs:
        raise TrainingError("--warmup-epochs must be >= 0 and below --epochs")
    if args.batch_size < 1 or args.workers < 0 or args.image_size < 32:
        raise TrainingError("Invalid batch size, worker count, or image size")
    if not 0 < args.class_balance_beta < 1:
        raise TrainingError("--class-balance-beta must be between 0 and 1")


def main() -> int:
    args = build_parser().parse_args()
    try:
        validate_args(args)
        dataset_root = args.dataset.resolve()
        classes = load_classes(dataset_root)
        samples, manifest_summary = load_manifest(dataset_root, classes)
    except (OSError, ValueError, TrainingError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("=" * 72)
    print("FIELDCARE DATASET_V2")
    print("=" * 72)
    print(f"Dataset       : {dataset_root}")
    print(f"Classes       : {len(classes)}")
    print(f"Images        : {manifest_summary['total_images']:,}")
    print(f"Source groups : {manifest_summary['family_count']:,}")
    print(f"Family leakage: {manifest_summary['family_leakage_count']}")
    for split in SPLITS:
        print(f"{split.capitalize():<14}: {manifest_summary['split_counts'][split]:,}")
    print()
    for item in classes:
        counts = manifest_summary["per_class_split"][item.key]
        print(
            f"{item.id:2d} {item.display_name:<48} "
            f"{counts['train']:>6,} / {counts['val']:>5,} / {counts['test']:>5,}"
        )

    if args.dry_run:
        print("\nDry run complete: labels, paths, splits, and leakage checks passed.")
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        print(
            "ERROR: CUDA is unavailable. Use an OVH GPU instance or pass "
            "--allow-cpu explicitly.",
            file=sys.stderr,
        )
        return 1

    run_id = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    if not re_safe_run_name(run_id):
        print("ERROR: --run-name may contain only letters, numbers, dot, _ and -", file=sys.stderr)
        return 1
    model_dir = DEFAULT_MODEL_ROOT / run_id
    output_dir = DEFAULT_OUTPUT_ROOT / run_id
    log_dir = DEFAULT_LOG_ROOT / run_id
    if model_dir.exists() or output_dir.exists() or log_dir.exists():
        print(f"ERROR: Run directory already exists: {run_id}", file=sys.stderr)
        return 1
    model_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    logger = setup_logger(log_dir / "training.log")

    set_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
    else:
        logger.warning("Training on CPU will be very slow")

    config = vars(args).copy()
    config["dataset"] = str(dataset_root)
    config.update(
        {
            "run_id": run_id,
            "device": str(device),
            "class_count": len(classes),
            "manifest_summary": manifest_summary,
            "classes": [asdict(item) for item in classes],
        }
    )
    write_json(output_dir / "config.json", config)

    train_transform, eval_transform = build_transforms(args.image_size)
    datasets = {
        "train": FieldCareV2Dataset(samples["train"], train_transform),
        "val": FieldCareV2Dataset(samples["val"], eval_transform),
        "test": FieldCareV2Dataset(samples["test"], eval_transform),
    }
    loaders = {
        split: make_loader(
            datasets[split], args.batch_size, args.workers,
            split == "train", device, args.seed
        )
        for split in SPLITS
    }

    class_weights, train_class_counts = effective_class_weights(
        samples["train"], len(classes), args.class_balance_beta
    )
    logger.info("Train class counts: %s", train_class_counts)
    logger.info("Class weights: %s", [round(value, 4) for value in class_weights.tolist()])

    weights = None if args.no_pretrained else EfficientNet_B3_Weights.DEFAULT
    model = efficientnet_b3(weights=weights)
    input_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=args.dropout, inplace=True),
        nn.Linear(input_features, len(classes)),
    )
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=args.label_smoothing,
    )
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_path = model_dir / "fieldcare_efficientnet_b3_best.pth"
    final_path = model_dir / "fieldcare_efficientnet_b3_final.pth"
    best_macro_f1 = -math.inf
    no_improvement = 0
    history: list[dict] = []
    training_started = time.time()

    for parameter in model.features.parameters():
        parameter.requires_grad = args.warmup_epochs == 0

    if args.warmup_epochs:
        optimizer = torch.optim.AdamW(
            model.classifier.parameters(),
            lr=args.head_lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": model.features.parameters(), "lr": args.backbone_lr},
                {"params": model.classifier.parameters(), "lr": args.head_lr},
            ],
            weight_decay=args.weight_decay,
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.warmup_epochs or args.epochs)
    )

    try:
        for epoch in range(1, args.epochs + 1):
            if epoch == args.warmup_epochs + 1 and args.warmup_epochs:
                for parameter in model.features.parameters():
                    parameter.requires_grad = True
                optimizer = torch.optim.AdamW(
                    [
                        {"params": model.features.parameters(), "lr": args.backbone_lr},
                        {"params": model.classifier.parameters(), "lr": args.head_lr},
                    ],
                    weight_decay=args.weight_decay,
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, args.epochs - args.warmup_epochs)
                )
                logger.info("Backbone unfrozen at epoch %d", epoch)

            phase = "warmup" if epoch <= args.warmup_epochs else "fine_tune"
            logger.info("Epoch %d/%d (%s)", epoch, args.epochs, phase)
            train_metrics = train_epoch(
                model, loaders["train"], criterion, optimizer,
                scaler, device, use_amp
            )
            val_metrics = evaluate(
                model, loaders["val"], criterion, device, use_amp
            )
            scheduler.step()

            record = {
                "epoch": epoch,
                "phase": phase,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "learning_rates": [group["lr"] for group in optimizer.param_groups],
            }
            history.append(record)
            write_json(output_dir / "history.json", history)
            logger.info(
                "train loss=%.4f acc=%.2f%% | val loss=%.4f acc=%.2f%% "
                "macro_f1=%.4f balanced_acc=%.2f%%",
                train_metrics["loss"], train_metrics["accuracy"] * 100,
                val_metrics["loss"], val_metrics["accuracy"] * 100,
                val_metrics["macro_f1"], val_metrics["balanced_accuracy"] * 100,
            )

            if val_metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = val_metrics["macro_f1"]
                no_improvement = 0
                torch.save(
                    checkpoint_payload(
                        model, classes, args, manifest_summary, epoch,
                        {key: value for key, value in val_metrics.items()
                         if key not in {"predictions", "targets"}},
                    ),
                    best_path,
                )
                logger.info("Saved new best checkpoint: %s", best_path)
            else:
                no_improvement += 1
                if no_improvement >= args.early_stop_patience:
                    logger.info("Early stopping after %d epochs without improvement", no_improvement)
                    break

        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = evaluate(model, loaders["test"], criterion, device, use_amp)
        test_report = classification_report(
            test_metrics["targets"],
            test_metrics["predictions"],
            labels=list(range(len(classes))),
            target_names=[item.display_name for item in classes],
            output_dict=True,
            zero_division=0,
        )
        matrix = confusion_matrix(
            test_metrics["targets"],
            test_metrics["predictions"],
            labels=list(range(len(classes))),
        )
        np.save(output_dir / "confusion_matrix.npy", matrix)
        write_json(output_dir / "classification_report.json", test_report)

        final_metrics = {
            key: value for key, value in test_metrics.items()
            if key not in {"predictions", "targets"}
        }
        summary = {
            "run_id": run_id,
            "best_validation_macro_f1": best_macro_f1,
            "test_metrics": final_metrics,
            "epochs_completed": len(history),
            "training_seconds": time.time() - training_started,
            "best_checkpoint": str(best_path),
            "final_checkpoint": str(final_path),
            "manifest_sha256": manifest_summary["manifest_sha256"],
        }
        torch.save(
            {
                **checkpoint_payload(
                    model, classes, args, manifest_summary,
                    checkpoint["epoch"], final_metrics
                ),
                "test_metrics": final_metrics,
            },
            final_path,
        )
        write_json(output_dir / "summary.json", summary)
        logger.info(
            "Test accuracy=%.2f%% macro_f1=%.4f balanced_accuracy=%.2f%%",
            final_metrics["accuracy"] * 100,
            final_metrics["macro_f1"],
            final_metrics["balanced_accuracy"] * 100,
        )
        logger.info("Training complete. Best=%s Final=%s", best_path, final_path)
        return 0
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        return 130
    except Exception:
        logger.exception("Training failed")
        return 1


def re_safe_run_name(value: str) -> bool:
    return bool(value) and all(character.isalnum() or character in "._-" for character in value)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    raise SystemExit(main())
