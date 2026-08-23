from pathlib import Path
import sys

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT_DIR / "dataset"
LABEL_FILE = DATASET_DIR / "label.txt"

MODEL_FILE = (
    ROOT_DIR
    / "models"
    / "fieldcare_efficientnet_b3_best.pth"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 300

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD LABELS
# ============================================================

def load_labels():

    if not LABEL_FILE.exists():

        raise FileNotFoundError(
            f"label.txt not found:\n{LABEL_FILE}"
        )

    labels = {}

    for line in LABEL_FILE.read_text(
        encoding="utf-8"
    ).splitlines():

        line = line.strip()

        if not line:
            continue

        parts = line.split(
            maxsplit=1
        )

        if len(parts) != 2:
            continue

        class_id = int(parts[0])

        labels[class_id] = parts[1]

    return dict(
        sorted(labels.items())
    )


LABELS = load_labels()

CLASS_IDS = sorted(
    LABELS.keys()
)

NUM_CLASSES = len(
    CLASS_IDS
)


# ============================================================
# MODEL
# ============================================================

print("=" * 70)
print("FIELDCARE IMAGE PREDICTION")
print("=" * 70)

print()
print("Device:", DEVICE)
print("Classes:", NUM_CLASSES)
print("Model:", MODEL_FILE)
print()


model = models.efficientnet_b3(
    weights=None
)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    NUM_CLASSES
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE
)


if isinstance(
    checkpoint,
    dict
) and "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

else:

    # Supports checkpoints that contain
    # only state_dict.
    model.load_state_dict(
        checkpoint
    )


model = model.to(
    DEVICE
)

model.eval()


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([

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
# PREDICT
# ============================================================

def predict(image_path):

    image_path = Path(
        image_path
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    image = Image.open(
        image_path
    ).convert("RGB")

    tensor = transform(
        image
    )

    tensor = tensor.unsqueeze(
        0
    )

    tensor = tensor.to(
        DEVICE
    )


    with torch.no_grad():

        outputs = model(
            tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]


    # Top predictions
    top_count = min(
        5,
        NUM_CLASSES
    )

    values, indices = torch.topk(
        probabilities,
        top_count
    )


    print()
    print("=" * 70)
    print("PREDICTION")
    print("=" * 70)

    print()
    print(
        "Image:",
        image_path
    )

    print()

    for rank, (
        probability,
        index
    ) in enumerate(
        zip(values, indices),
        start=1
    ):

        class_index = (
            index.item()
        )

        # Model index 0 corresponds
        # to folder/class ID 1.
        class_id = (
            class_index + 1
        )

        class_name = LABELS.get(
            class_id,
            f"Unknown class {class_id}"
        )

        confidence = (
            probability.item()
            * 100
        )

        print(
            f"{rank}. "
            f"{class_name:<55} "
            f"{confidence:6.2f}%"
        )

    print()

    best_index = (
        indices[0].item()
    )

    best_class_id = (
        best_index + 1
    )

    best_name = LABELS[
        best_class_id
    ]

    best_confidence = (
        values[0].item()
        * 100
    )

    print("=" * 70)

    print(
        f"RESULT: {best_name}"
    )

    print(
        f"CONFIDENCE: {best_confidence:.2f}%"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print()

        print(
            "python scripts/predict.py "
            "\"path\\to\\plant_image.jpg\""
        )

        print()

        sys.exit(1)


    predict(
        sys.argv[1]
    )