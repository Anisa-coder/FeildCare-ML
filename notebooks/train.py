from pathlib import Path
import random

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import (
    efficientnet_b3,
    EfficientNet_B3_Weights
)

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from tqdm import tqdm


# ============================================================
# FIELDCARE CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT_DIR / "dataset"
LABEL_FILE = DATASET_DIR / "label.txt"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"


def load_labels():
    """Load the authoritative class mapping from dataset/label.txt."""
    if not LABEL_FILE.exists():
        raise FileNotFoundError(f"label.txt not found: {LABEL_FILE}")

    labels = {}

    for line_number, raw_line in enumerate(
        LABEL_FILE.read_text(encoding="utf-8").splitlines(),
        start=1,
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
            raise ValueError(
                f"Invalid class ID {class_id} on line {line_number}."
            )

        if class_id in labels:
            raise ValueError(
                f"Duplicate class ID {class_id} in label.txt."
            )

        labels[class_id] = parts[1].strip()

    if not labels:
        raise ValueError(f"No classes found in {LABEL_FILE}")

    return dict(sorted(labels.items()))


LABELS = load_labels()
CLASS_IDS = sorted(LABELS)
NUM_CLASSES = len(CLASS_IDS)

# This training pipeline maps folder 1 -> model class 0,
# folder 2 -> model class 1, etc. Therefore IDs must be contiguous.
expected_ids = list(range(1, NUM_CLASSES + 1))

if CLASS_IDS != expected_ids:
    raise ValueError(
        "Class IDs in label.txt must be contiguous starting at 1.\n"
        f"Found: {CLASS_IDS}\n"
        f"Expected: {expected_ids}\n"
        "Run your dataset renumbering/merge script first."
    )


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TRAINING SETTINGS
# ============================================================


IMAGE_SIZE = 300

BATCH_SIZE = 32

EPOCHS = 30

LEARNING_RATE = 1e-4

RANDOM_SEED = 42


# Windows:
#
# 4 workers keeps image loading/augmentation off the main
# training process. If CPU/RAM usage is high, try 2.


NUM_WORKERS = 4


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

else:

    DEVICE = torch.device("cpu")


# ============================================================
# GPU PERFORMANCE SETTINGS
# ============================================================

if DEVICE.type == "cuda":

    # Use optimized convolution algorithms for fixed 300x300 inputs.
    torch.backends.cudnn.benchmark = True

    # Allows TF32 on Ampere GPUs such as RTX 3060.
    # This speeds up matrix multiplications/convolutions with
    # negligible practical impact on image-classification quality.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Use high-precision matmul kernels where available.
    torch.set_float32_matmul_precision("high")


print("=" * 70)
print("              FIELDCARE TRAINING")
print("=" * 70)

print()

print("Device:", DEVICE)
print("Classes:", NUM_CLASSES)
print("Label file:", LABEL_FILE)

if DEVICE.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA:",
        torch.version.cuda
    )

if DEVICE.type == "cuda":
    print(
        "Mixed precision: FP16"
    )
    print(
        "CUDA workers:",
        NUM_WORKERS
    )
    print(
        "Batch size:",
        BATCH_SIZE
    )

print()


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(
    RANDOM_SEED
)

torch.manual_seed(
    RANDOM_SEED
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )


# ============================================================
# FIND IMAGES
# ============================================================

print("=" * 70)
print("SCANNING DATASET")
print("=" * 70)

print()

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


samples = []


for folder_id in CLASS_IDS:

    folder = DATASET_DIR / str(folder_id)

    if not folder.exists():

        print(
            f"WARNING: Folder {folder_id} missing "
            f"({LABELS[folder_id]})"
        )

        continue

    # Folder IDs are 1-based; PyTorch targets are 0-based.
    class_index = folder_id - 1

    image_files = []

    for image_path in folder.rglob("*"):

        if (
            image_path.is_file()
            and
            image_path.suffix.lower()
            in VALID_EXTENSIONS
        ):

            image_files.append(image_path)

    print(
        f"Class {folder_id:2d}: "
        f"{LABELS[folder_id]:<55} "
        f"{len(image_files):>6,} images"
    )

    for image_path in image_files:

        samples.append(
            (
                image_path,
                class_index
            )
        )


print()
print(
    f"Total images: {len(samples):,}"
)

print()


# ============================================================
# CHECK EMPTY CLASSES
# ============================================================

empty_classes = []


for folder_id in CLASS_IDS:

    class_index = folder_id - 1

    count = sum(
        label == class_index
        for _, label in samples
    )

    if count == 0:

        empty_classes.append(
            folder_id
        )


if empty_classes:

    print(
        "ERROR: These classes contain"
        " no images:"
    )

    for class_id in empty_classes:
        print(
            f"  {class_id}: {LABELS[class_id]}"
        )

    print()

    print(
        "Fix the dataset before training."
    )

    raise SystemExit(1)


# ============================================================
# EXTRACT PATHS / LABELS
# ============================================================

paths = [
    item[0]
    for item in samples
]


labels = [
    item[1]
    for item in samples
]


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("=" * 70)
print("CREATING DATASET SPLIT")
print("=" * 70)

print()

#
# 70% TRAIN
# 15% VALIDATION
# 15% TEST
#

train_paths, temp_paths, train_labels, temp_labels = (
    train_test_split(
        paths,
        labels,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=labels
    )
)


val_paths, test_paths, val_labels, test_labels = (
    train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_labels
    )
)


print(
    f"Training   : {len(train_paths):,}"
)

print(
    f"Validation : {len(val_paths):,}"
)

print(
    f"Testing    : {len(test_paths):,}"
)

print()


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.75, 1.0)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomVerticalFlip(
        p=0.2
    ),

    transforms.RandomRotation(
        degrees=20
    ),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


eval_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# DATASET CLASS
# ============================================================

class FieldCareDataset(Dataset):

    def __init__(
        self,
        image_paths,
        labels,
        transform=None
    ):

        self.image_paths = image_paths

        self.labels = labels

        self.transform = transform


    def __len__(self):

        return len(
            self.image_paths
        )


    def __getitem__(
        self,
        index
    ):

        image_path = (
            self.image_paths[index]
        )

        label = (
            self.labels[index]
        )


        try:

            image = Image.open(
                image_path
            ).convert("RGB")


        except Exception as error:

            print(
                f"\nCould not read:"
                f" {image_path}"
            )

            raise error


        if self.transform:

            image = self.transform(
                image
            )


        return image, label


# ============================================================

def main():

    # CREATE DATASETS
    # ============================================================

    train_dataset = FieldCareDataset(
        train_paths,
        train_labels,
        train_transform
    )


    val_dataset = FieldCareDataset(
        val_paths,
        val_labels,
        eval_transform
    )


    test_dataset = FieldCareDataset(
        test_paths,
        test_labels,
        eval_transform
    )


    # ============================================================
    # CREATE DATALOADERS
    # ============================================================

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS,

        pin_memory=(
            DEVICE.type == "cuda"
        ),

        persistent_workers=(
            NUM_WORKERS > 0
        ),

        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=(
            DEVICE.type == "cuda"
        ),

        persistent_workers=(
            NUM_WORKERS > 0
        ),

        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )


    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=(
            DEVICE.type == "cuda"
        ),

        persistent_workers=(
            NUM_WORKERS > 0
        ),

        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )


    # ============================================================
    # LOAD PRETRAINED MODEL
    # ============================================================

    print("=" * 70)
    print("LOADING EFFICIENTNET-B3")
    print("=" * 70)

    print()

    weights = (
        EfficientNet_B3_Weights.DEFAULT
    )


    model = efficientnet_b3(
        weights=weights
    )


    # ============================================================
    # REPLACE CLASSIFIER
    # ============================================================

    input_features = (
        model.classifier[1].in_features
    )


    model.classifier[1] = nn.Linear(
        input_features,
        NUM_CLASSES
    )


    model = model.to(
        DEVICE
    )

    # Channels-last is well suited to convolutional models and
    # can improve throughput on NVIDIA GPUs without changing
    # the model architecture or image resolution.
    if DEVICE.type == "cuda":
        model = model.to(
            memory_format=torch.channels_last
        )


    # ============================================================
    # MIXED PRECISION
    # ============================================================

    # FP16 autocasting + GradScaler uses the RTX 3060 Tensor Cores
    # to accelerate training while keeping sensitive operations
    # in FP32. It does not reduce IMAGE_SIZE or model capacity.
    if DEVICE.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = None


    # ============================================================
    # LOSS
    # ============================================================

    criterion = nn.CrossEntropyLoss()


    # ============================================================
    # OPTIMIZER
    # ============================================================

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=1e-4
    )


    # ============================================================
    # LEARNING RATE SCHEDULER
    # ============================================================

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS
        )
    )


    # ============================================================
    # TRAINING VARIABLES
    # ============================================================

    best_val_accuracy = 0.0


    best_model_path = (
        MODEL_DIR /
        "fieldcare_efficientnet_b3_best.pth"
    )


    # ============================================================
    # TRAINING LOOP
    # ============================================================

    for epoch in range(
        EPOCHS
    ):

        print()
        print("=" * 70)

        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        print("=" * 70)


        # ========================================================
        # TRAIN
        # ========================================================

        model.train()


        train_loss = 0.0

        train_correct = 0

        train_total = 0


        progress = tqdm(
            train_loader,
            desc="Training"
        )


        for images, targets in progress:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            if DEVICE.type == "cuda":
                images = images.contiguous(
                    memory_format=torch.channels_last
                )

            targets = targets.to(
                DEVICE,
                non_blocking=True
            )


            optimizer.zero_grad(
                set_to_none=True
            )


            if DEVICE.type == "cuda":

                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                ):

                    outputs = model(
                        images
                    )

                    loss = criterion(
                        outputs,
                        targets
                    )

                scaler.scale(
                    loss
                ).backward()

                scaler.step(
                    optimizer
                )

                scaler.update()

            else:

                outputs = model(
                    images
                )

                loss = criterion(
                    outputs,
                    targets
                )

                loss.backward()

                optimizer.step()


            train_loss += (
                loss.item()
                * images.size(0)
            )


            predictions = (
                outputs.argmax(
                    dim=1
                )
            )


            train_correct += (
                predictions == targets
            ).sum().item()


            train_total += (
                targets.size(0)
            )


            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )


        train_loss /= train_total

        train_accuracy = (
            train_correct /
            train_total
        )


        # ========================================================
        # VALIDATION
        # ========================================================

        model.eval()


        val_loss = 0.0

        val_correct = 0

        val_total = 0


        with torch.no_grad():

            for images, targets in val_loader:

                images = images.to(
                    DEVICE,
                    non_blocking=True
                )

                targets = targets.to(
                    DEVICE,
                    non_blocking=True
                )


                if DEVICE.type == "cuda":

                    images = images.contiguous(
                        memory_format=torch.channels_last
                    )

                    with torch.amp.autocast(
                        device_type="cuda",
                        dtype=torch.float16
                    ):

                        outputs = model(
                            images
                        )

                        loss = criterion(
                            outputs,
                            targets
                        )

                else:

                    outputs = model(
                        images
                    )

                    loss = criterion(
                        outputs,
                        targets
                    )


                val_loss += (
                    loss.item()
                    * images.size(0)
                )


                predictions = (
                    outputs.argmax(
                        dim=1
                    )
                )


                val_correct += (
                    predictions == targets
                ).sum().item()


                val_total += (
                    targets.size(0)
                )


        val_loss /= val_total

        val_accuracy = (
            val_correct /
            val_total
        )


        scheduler.step()


        # ========================================================
        # RESULTS
        # ========================================================

        print()

        print(
            f"Train Loss : "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Acc  : "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"Val Loss   : "
            f"{val_loss:.4f}"
        )

        print(
            f"Val Acc    : "
            f"{val_accuracy * 100:.2f}%"
        )


        # ========================================================
        # SAVE BEST MODEL
        # ========================================================

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = (
                val_accuracy
            )


            torch.save(

                {
                    "model_state_dict":
                        model.state_dict(),

                    "num_classes":
                        NUM_CLASSES,

                    "class_ids":
                        CLASS_IDS,

                    "labels":
                        LABELS,

                    "image_size":
                        IMAGE_SIZE,

                    "best_val_accuracy":
                        best_val_accuracy
                },

                best_model_path
            )


            print()
            print(
                "✓ New best model saved."
            )


    # ============================================================
    # TEST BEST MODEL
    # ============================================================

    print()
    print("=" * 70)
    print("FINAL TEST")
    print("=" * 70)


    checkpoint = torch.load(

        best_model_path,

        map_location=DEVICE,

        weights_only=False
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    model.eval()


    all_predictions = []

    all_targets = []


    with torch.no_grad():

        for images, targets in tqdm(
            test_loader,
            desc="Testing"
        ):

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            if DEVICE.type == "cuda":
                images = images.contiguous(
                    memory_format=torch.channels_last
                )


            if DEVICE.type == "cuda":

                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                ):

                    outputs = model(
                        images
                    )

            else:

                outputs = model(
                    images
                )


            predictions = (
                outputs.argmax(
                    dim=1
                )
            )


            all_predictions.extend(
                predictions.cpu().numpy()
            )


            all_targets.extend(
                targets.numpy()
            )


    # ============================================================
    # CLASSIFICATION REPORT
    # ============================================================

    print()
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print()

    target_names = [
        LABELS[class_id]
        for class_id in CLASS_IDS
    ]

    print(
        classification_report(
            all_targets,
            all_predictions,
            labels=list(range(NUM_CLASSES)),
            target_names=target_names,
            digits=4,
            zero_division=0
        )
    )


    # ============================================================
    # CONFUSION MATRIX
    # ============================================================

    matrix = confusion_matrix(
        all_targets,
        all_predictions
    )


    torch.save(
        matrix,
        OUTPUT_DIR /
        "confusion_matrix.pt"
    )


    # ============================================================
    # SAVE FINAL MODEL
    # ============================================================

    final_model_path = (
        MODEL_DIR /
        "fieldcare_efficientnet_b3_final.pth"
    )

    mapping_path = OUTPUT_DIR / "trained_label_mapping.txt"

    with open(
        mapping_path,
        "w",
        encoding="utf-8"
    ) as mapping_file:

        for class_id in CLASS_IDS:
            mapping_file.write(
                f"{class_id} {LABELS[class_id]}\\n"
            )


    torch.save(

        {
            "model_state_dict":
                model.state_dict(),

            "num_classes":
                NUM_CLASSES,

            "class_ids":
                CLASS_IDS,

            "labels":
                LABELS,

            "image_size":
                IMAGE_SIZE
        },

        final_model_path
    )


    # ============================================================
    # COMPLETE
    # ============================================================

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print()

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print()

    print(
        f"Best model:"
    )

    print(
        best_model_path
    )

    print()

    print(
        f"Final model:"
    )

    print(
        final_model_path
    )


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()