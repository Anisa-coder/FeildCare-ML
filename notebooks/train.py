

from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

from PIL import Image

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
)

from tqdm import tqdm

try:
    import imagehash
    HAVE_IMAGEHASH = True
except ImportError:
    HAVE_IMAGEHASH = False


# ============================================================
# CONFIG
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
LABEL_FILE = DATASET_DIR / "label.txt"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"

IMAGE_SIZE = 300
BATCH_SIZE = 32
RANDOM_SEED = 42
NUM_WORKERS = 4

# Two-phase training
FREEZE_EPOCHS = 3          # backbone frozen, head-only warmup
FINE_TUNE_EPOCHS = 27      # full network unfrozen after warmup
HEAD_LR = 1e-4
BACKBONE_LR = 1e-5         # lower LR for pretrained backbone in phase 2
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.4         # slightly higher than default b3 (0.3) - more
                            # regularization given several very small classes
LABEL_SMOOTHING = 0.1

EARLY_STOP_PATIENCE = 7    # epochs without macro-F1 improvement before stop

# Class-balanced loss (effective number of samples, Cui et al. 2019)
CB_BETA = 0.999

# Leakage-aware split
USE_DEDUP_SPLIT = True     # requires `pip install imagehash`

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LABELS
# ============================================================

def load_labels():
    if not LABEL_FILE.exists():
        raise FileNotFoundError(f"label.txt not found: {LABEL_FILE}")

    labels = {}
    for line_number, raw_line in enumerate(
        LABEL_FILE.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not parts[0].isdigit():
            raise ValueError(
                f"Invalid label.txt line {line_number}: {raw_line!r}\n"
                "Expected: <class_id> <class_name>"
            )

        class_id = int(parts[0])
        if class_id <= 0:
            raise ValueError(f"Invalid class ID {class_id} on line {line_number}.")
        if class_id in labels:
            raise ValueError(f"Duplicate class ID {class_id} in label.txt.")

        labels[class_id] = parts[1].strip()

    if not labels:
        raise ValueError(f"No classes found in {LABEL_FILE}")

    return dict(sorted(labels.items()))


LABELS = load_labels()
CLASS_IDS = sorted(LABELS)
NUM_CLASSES = len(CLASS_IDS)

expected_ids = list(range(1, NUM_CLASSES + 1))
if CLASS_IDS != expected_ids:
    raise ValueError(
        "Class IDs in label.txt must be contiguous starting at 1.\n"
        f"Found: {CLASS_IDS}\nExpected: {expected_ids}"
    )


# ============================================================
# DEVICE / GPU SETTINGS
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

print("=" * 70)
print("              FIELDCARE TRAINING v2")
print("=" * 70)
print()
print("Device:", DEVICE)
print("Classes:", NUM_CLASSES)
print("Dedup-aware split:", USE_DEDUP_SPLIT and HAVE_IMAGEHASH)
if USE_DEDUP_SPLIT and not HAVE_IMAGEHASH:
    print("  -> imagehash not installed, falling back to stratified split")
    print("     (run: pip install imagehash)")
print()


# ============================================================
# SCAN DATASET
# ============================================================

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

samples = []
for folder_id in CLASS_IDS:
    folder = DATASET_DIR / str(folder_id)
    if not folder.exists():
        print(f"WARNING: Folder {folder_id} missing ({LABELS[folder_id]})")
        continue

    class_index = folder_id - 1
    image_files = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]

    print(f"Class {folder_id:2d}: {LABELS[folder_id]:<45} {len(image_files):>6,} images")
    samples.extend((p, class_index) for p in image_files)

print()
print(f"Total images: {len(samples):,}")
print()

empty_classes = [
    fid for fid in CLASS_IDS
    if sum(lbl == fid - 1 for _, lbl in samples) == 0
]
if empty_classes:
    print("ERROR: These classes contain no images:")
    for cid in empty_classes:
        print(f"  {cid}: {LABELS[cid]}")
    raise SystemExit(1)

paths = [s[0] for s in samples]
labels = [s[1] for s in samples]


# ============================================================
# LEAKAGE-AWARE TRAIN / VAL / TEST SPLIT
# ============================================================

def compute_groups(image_paths):
    """Perceptual-hash each image so near-duplicates share a group id and
    can't be split across train/val/test."""
    groups = []
    for p in tqdm(image_paths, desc="Hashing images for dedup-aware split"):
        try:
            with Image.open(p) as img:
                h = imagehash.phash(img.convert("RGB"))
            groups.append(str(h))
        except Exception:
            # Unreadable file: give it a unique group so it doesn't
            # accidentally collide with real duplicates.
            groups.append(f"unreadable::{p}")
    return groups


print("=" * 70)
print("CREATING DATASET SPLIT")
print("=" * 70)
print()

if USE_DEDUP_SPLIT and HAVE_IMAGEHASH:
    groups = compute_groups(paths)
    n_unique_groups = len(set(groups))
    print(f"Unique perceptual hashes: {n_unique_groups:,} / {len(paths):,} images")
    if n_unique_groups < len(paths):
        print(f"  -> found {len(paths) - n_unique_groups:,} likely near-duplicate images")
    print()

    indices = np.arange(len(paths))
    labels_arr = np.array(labels)
    groups_arr = np.array(groups)

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_SEED)
    train_idx, temp_idx = next(gss1.split(indices, labels_arr, groups_arr))

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_SEED)
    val_rel_idx, test_rel_idx = next(
        gss2.split(temp_idx, labels_arr[temp_idx], groups_arr[temp_idx])
    )
    val_idx = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]

    train_paths = [paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_paths = [paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    test_paths = [paths[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    # Group-based splitting can skew per-class proportions slightly -
    # print the resulting distribution so it's easy to sanity check.
    print("Per-class counts after grouped split (train / val / test):")
    for cid in CLASS_IDS:
        ci = cid - 1
        tr = sum(l == ci for l in train_labels)
        va = sum(l == ci for l in val_labels)
        te = sum(l == ci for l in test_labels)
        print(f"  {cid:2d} {LABELS[cid]:<45} {tr:>6,} / {va:>5,} / {te:>5,}")
    print()
else:
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=0.30, random_state=RANDOM_SEED, stratify=labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, random_state=RANDOM_SEED,
        stratify=temp_labels,
    )

print(f"Training   : {len(train_paths):,}")
print(f"Validation : {len(val_paths):,}")
print(f"Testing    : {len(test_paths):,}")
print()


# ============================================================
# CLASS-BALANCED LOSS WEIGHTS (effective number of samples)
# ============================================================

def compute_class_weights(class_labels, num_classes, beta=CB_BETA):
    counts = np.bincount(class_labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1)  # guard against divide-by-zero
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


class_weights = compute_class_weights(train_labels, NUM_CLASSES)

print("Class weights (effective-number-of-samples re-weighting):")
for cid in CLASS_IDS:
    print(f"  {cid:2d} {LABELS[cid]:<45} weight={class_weights[cid - 1]:.3f}")
print()


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ============================================================
# DATASET
# ============================================================

class FieldCareDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as error:
            print(f"\nCould not read: {image_path}")
            raise error
        if self.transform:
            image = self.transform(image)
        return image, label


def make_loader(dataset, shuffle):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )


# ============================================================
# EVAL HELPERS
# ============================================================

def run_eval(model, loader, criterion):
    model.eval()
    total_loss, total_count = 0.0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            if DEVICE.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)

            total_loss += loss.item() * images.size(0)
            total_count += targets.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / total_count
    accuracy = np.mean(np.array(all_preds) == np.array(all_targets))
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    balanced_acc = balanced_accuracy_score(all_targets, all_preds)
    return avg_loss, accuracy, macro_f1, balanced_acc, all_preds, all_targets


# ============================================================
# MAIN
# ============================================================

def main():
    train_dataset = FieldCareDataset(train_paths, train_labels, train_transform)
    val_dataset = FieldCareDataset(val_paths, val_labels, eval_transform)
    test_dataset = FieldCareDataset(test_paths, test_labels, eval_transform)

    train_loader = make_loader(train_dataset, shuffle=True)
    val_loader = make_loader(val_dataset, shuffle=False)
    test_loader = make_loader(test_dataset, shuffle=False)

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------
    print("=" * 70)
    print("LOADING EFFICIENTNET-B3")
    print("=" * 70)
    print()

    weights = EfficientNet_B3_Weights.DEFAULT
    model = efficientnet_b3(weights=weights)

    input_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT_RATE, inplace=True),
        nn.Linear(input_features, NUM_CLASSES),
    )
    model = model.to(DEVICE)
    if DEVICE.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(DEVICE), label_smoothing=LABEL_SMOOTHING
    )

    best_model_path = MODEL_DIR / "fieldcare_efficientnet_b3_best.pth"
    best_macro_f1 = 0.0
    epochs_without_improvement = 0
    total_epochs = FREEZE_EPOCHS + FINE_TUNE_EPOCHS

    # --------------------------------------------------------
    # PHASE 1: freeze backbone, warm up classifier head
    # --------------------------------------------------------
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FREEZE_EPOCHS)

    global_epoch = 0
    for phase_name, phase_epochs in (
        ("WARMUP (backbone frozen)", FREEZE_EPOCHS),
        ("FINE-TUNE (backbone unfrozen)", FINE_TUNE_EPOCHS),
    ):
        if phase_name.startswith("FINE-TUNE"):
            # unfreeze backbone, rebuild optimizer with differential LRs
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.features.parameters(), "lr": BACKBONE_LR},
                    {"params": model.classifier.parameters(), "lr": HEAD_LR},
                ],
                weight_decay=WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=phase_epochs
            )

        for _ in range(phase_epochs):
            global_epoch += 1
            print()
            print("=" * 70)
            print(f"Epoch {global_epoch}/{total_epochs}  [{phase_name}]")
            print("=" * 70)

            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            progress = tqdm(train_loader, desc="Training")

            for images, targets in progress:
                images = images.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)
                if DEVICE.type == "cuda":
                    images = images.contiguous(memory_format=torch.channels_last)

                optimizer.zero_grad(set_to_none=True)

                if DEVICE.type == "cuda":
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = model(images)
                        loss = criterion(outputs, targets)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                train_correct += (preds == targets).sum().item()
                train_total += targets.size(0)
                progress.set_postfix(loss=f"{loss.item():.4f}")

            train_loss /= train_total
            train_accuracy = train_correct / train_total
            scheduler.step()

            val_loss, val_accuracy, val_macro_f1, val_balanced_acc, val_preds, val_targets = (
                run_eval(model, val_loader, criterion)
            )

            print()
            print(f"Train Loss     : {train_loss:.4f}   Train Acc : {train_accuracy * 100:.2f}%")
            print(f"Val   Loss     : {val_loss:.4f}   Val   Acc : {val_accuracy * 100:.2f}%")
            print(f"Val Macro-F1   : {val_macro_f1:.4f}")
            print(f"Val Balanced Acc: {val_balanced_acc * 100:.2f}%")

            # Print full per-class report every 5 epochs so you can see
            # which classes are still struggling without flooding the log.
            if global_epoch % 5 == 0 or global_epoch == total_epochs:
                print()
                print(classification_report(
                    val_targets, val_preds,
                    labels=list(range(NUM_CLASSES)),
                    target_names=[LABELS[c] for c in CLASS_IDS],
                    digits=3, zero_division=0,
                ))

            # Model selection + early stopping on macro-F1, not accuracy
            if val_macro_f1 > best_macro_f1:
                best_macro_f1 = val_macro_f1
                epochs_without_improvement = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "num_classes": NUM_CLASSES,
                    "class_ids": CLASS_IDS,
                    "labels": LABELS,
                    "image_size": IMAGE_SIZE,
                    "best_val_macro_f1": best_macro_f1,
                    "best_val_accuracy": val_accuracy,
                    "best_val_balanced_accuracy": val_balanced_acc,
                }, best_model_path)
                print()
                print(f"New best model saved (macro-F1={best_macro_f1:.4f}).")
            else:
                epochs_without_improvement += 1
                print()
                print(f"No macro-F1 improvement for {epochs_without_improvement} epoch(s).")
                if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                    print(f"Early stopping (patience={EARLY_STOP_PATIENCE}).")
                    break
        else:
            continue
        break  # early stop broke the inner loop -> also break the outer phase loop

    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------
    print()
    print("=" * 70)
    print("FINAL TEST")
    print("=" * 70)

    checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_accuracy, test_macro_f1, test_balanced_acc, test_preds, test_targets = (
        run_eval(model, test_loader, criterion)
    )

    print()
    print(f"Test Loss        : {test_loss:.4f}")
    print(f"Test Accuracy    : {test_accuracy * 100:.2f}%")
    print(f"Test Macro-F1    : {test_macro_f1:.4f}")
    print(f"Test Balanced Acc: {test_balanced_acc * 100:.2f}%")
    print()

    target_names = [LABELS[c] for c in CLASS_IDS]
    print(classification_report(
        test_targets, test_preds,
        labels=list(range(NUM_CLASSES)),
        target_names=target_names, digits=4, zero_division=0,
    ))

    matrix = confusion_matrix(test_targets, test_preds)
    torch.save(matrix, OUTPUT_DIR / "confusion_matrix.pt")

    final_model_path = MODEL_DIR / "fieldcare_efficientnet_b3_final.pth"
    mapping_path = OUTPUT_DIR / "trained_label_mapping.txt"
    with open(mapping_path, "w", encoding="utf-8") as f:
        for cid in CLASS_IDS:
            f.write(f"{cid} {LABELS[cid]}\n")

    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes": NUM_CLASSES,
        "class_ids": CLASS_IDS,
        "labels": LABELS,
        "image_size": IMAGE_SIZE,
    }, final_model_path)

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print()
    print(f"Best val macro-F1: {best_macro_f1:.4f}")
    print(f"Best model : {best_model_path}")
    print(f"Final model: {final_model_path}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()