from __future__ import annotations

from pathlib import Path
import csv
import time

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT_DIR / "dataset"
DATASET_LABEL_FILE = DATASET_DIR / "label.txt"

REAL_WORLD_DIR = ROOT_DIR / "real_world"
IMAGE_DIR = REAL_WORLD_DIR / "images"
REAL_LABEL_FILE = REAL_WORLD_DIR / "label.txt"

RESULT_DIR = REAL_WORLD_DIR / "results"

MODEL_FILE = (
    ROOT_DIR
    / "models"
    / "fieldcare_efficientnet_b3_best.pth"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 300

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

REAL_WORLD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LABEL LOADER
# ============================================================

def load_labels(path: Path):

    if not path.exists():

        raise FileNotFoundError(
            f"Label file not found:\n{path}"
        )

    labels = {}

    for line_number, raw_line in enumerate(
        path.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):

        line = raw_line.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        parts = line.split(
            maxsplit=1
        )

        if len(parts) != 2:

            raise ValueError(
                f"Invalid line {line_number} "
                f"in {path}:\n{raw_line}"
            )

        class_id = int(
            parts[0]
        )

        class_name = parts[1].strip()

        labels[class_id] = class_name

    if not labels:

        raise ValueError(
            f"No labels found in {path}"
        )

    return dict(
        sorted(labels.items())
    )


# ============================================================
# LOAD FIELDCARE LABELS
# ============================================================

LABELS = load_labels(
    DATASET_LABEL_FILE
)

CLASS_IDS = sorted(
    LABELS.keys()
)

NUM_CLASSES = len(
    CLASS_IDS
)


# ============================================================
# LOAD REAL-WORLD LABEL FILE
# ============================================================

def load_ground_truth():

    if not REAL_LABEL_FILE.exists():

        raise FileNotFoundError(
            f"""
Real-world label file does not exist:

{REAL_LABEL_FILE}

Create it using:

image001.jpg 1
image002.jpg 14
image003.jpg 15
"""
        )

    ground_truth = {}

    for line_number, raw_line in enumerate(
        REAL_LABEL_FILE.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):

        line = raw_line.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        parts = line.split()

        if len(parts) != 2:

            raise ValueError(
                f"Invalid label.txt line "
                f"{line_number}:\n"
                f"{raw_line}\n\n"
                f"Expected:\n"
                f"image.jpg CLASS_ID"
            )

        filename = parts[0]

        class_id = int(
            parts[1]
        )

        if class_id not in LABELS:

            raise ValueError(
                f"Unknown class {class_id} "
                f"on line {line_number}."
            )

        ground_truth[filename] = (
            class_id
        )

    return ground_truth


# ============================================================
# MODEL
# ============================================================

def create_model():

    print()
    print("=" * 70)
    print("LOADING FIELDCARE MODEL")
    print("=" * 70)

    print()
    print(
        f"Model : {MODEL_FILE}"
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Classes: {NUM_CLASSES}"
    )

    model = models.efficientnet_b3(
        weights=None
    )

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        NUM_CLASSES
    )

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=DEVICE
    )

    if (
        isinstance(
            checkpoint,
            dict
        )
        and
        "model_state_dict"
        in checkpoint
    ):

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model = model.to(
        DEVICE
    )

    model.eval()

    return model


# ============================================================
# TRANSFORM
# ============================================================

TRANSFORM = transforms.Compose([

    transforms.Resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406,
        ],

        std=[
            0.229,
            0.224,
            0.225,
        ],
    ),
])


# ============================================================
# FIND IMAGES
# ============================================================

def find_images():

    images = []

    for path in IMAGE_DIR.rglob("*"):

        if (
            path.is_file()
            and
            path.suffix.lower()
            in VALID_EXTENSIONS
        ):

            images.append(
                path
            )

    return sorted(
        images,
        key=lambda p: p.name.lower()
    )


# ============================================================
# PREDICT ONE IMAGE
# ============================================================

def predict_image(
    model,
    image_path: Path,
):

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    tensor = TRANSFORM(
        image
    )

    tensor = tensor.unsqueeze(
        0
    )

    tensor = tensor.to(
        DEVICE
    )

    start = time.perf_counter()

    with torch.no_grad():

        output = model(
            tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    confidence, prediction = (
        probabilities[0].max(
            dim=0
        )
    )

    model_index = (
        prediction.item()
    )

    predicted_class = (
        model_index + 1
    )

    confidence = (
        confidence.item()
        * 100
    )

    return (
        predicted_class,
        confidence,
        elapsed,
    )


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 70)
    print("FIELDCARE REAL-WORLD MODEL EVALUATION")
    print("=" * 70)

    print()
    print(
        f"Dataset labels : "
        f"{DATASET_LABEL_FILE}"
    )

    print(
        f"Real-world dir  : "
        f"{REAL_WORLD_DIR}"
    )

    print(
        f"Model           : "
        f"{MODEL_FILE}"
    )

    # --------------------------------------------------------
    # Load ground truth
    # --------------------------------------------------------

    ground_truth = (
        load_ground_truth()
    )

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    images = find_images()

    if not images:

        print()
        print(
            "ERROR: No images found."
        )

        print()
        print(
            f"Put images inside:"
        )

        print(
            IMAGE_DIR
        )

        return

    print()
    print(
        f"Images found: "
        f"{len(images)}"
    )

    # --------------------------------------------------------
    # Check labels
    # --------------------------------------------------------

    image_names = {
        image.name
        for image in images
    }

    missing_labels = [
        image.name
        for image in images
        if image.name
        not in ground_truth
    ]

    extra_labels = [
        filename
        for filename in ground_truth
        if filename
        not in image_names
    ]

    if missing_labels:

        print()
        print(
            "ERROR: These images "
            "have no ground-truth label:"
        )

        for filename in missing_labels:

            print(
                f"  {filename}"
            )

        print()
        print(
            "Add them to:"
        )

        print(
            REAL_LABEL_FILE
        )

        return

    if extra_labels:

        print()
        print(
            "WARNING: label.txt contains "
            "images that don't exist:"
        )

        for filename in extra_labels:

            print(
                f"  {filename}"
            )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = create_model()

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    results = []

    y_true = []
    y_pred = []

    total_inference_time = 0.0

    print()
    print("=" * 70)
    print("TESTING REAL-WORLD IMAGES")
    print("=" * 70)

    for index, image_path in enumerate(
        images,
        start=1,
    ):

        actual_class = (
            ground_truth[
                image_path.name
            ]
        )

        try:

            (
                predicted_class,
                confidence,
                inference_time,
            ) = predict_image(
                model,
                image_path,
            )

        except Exception as error:

            print()
            print(
                f"ERROR: {image_path.name}"
            )

            print(
                error
            )

            continue

        total_inference_time += (
            inference_time
        )

        correct = (
            actual_class
            ==
            predicted_class
        )

        y_true.append(
            actual_class - 1
        )

        y_pred.append(
            predicted_class - 1
        )

        actual_name = LABELS[
            actual_class
        ]

        predicted_name = LABELS.get(
            predicted_class,
            f"Unknown {predicted_class}"
        )

        status = (
            "CORRECT"
            if correct
            else "WRONG"
        )

        print(
            f"[{index:>4}/{len(images):<4}] "
            f"{status:<7} "
            f"{image_path.name:<35} "
            f"{confidence:6.2f}%"
        )

        if not correct:

            print(
                f"       Actual   : "
                f"{actual_class} - "
                f"{actual_name}"
            )

            print(
                f"       Predicted: "
                f"{predicted_class} - "
                f"{predicted_name}"
            )

        results.append({

            "image": image_path.name,

            "actual_class":
                actual_class,

            "actual_name":
                actual_name,

            "predicted_class":
                predicted_class,

            "predicted_name":
                predicted_name,

            "confidence":
                round(
                    confidence,
                    4
                ),

            "correct":
                correct,

            "inference_ms":
                round(
                    inference_time
                    * 1000,
                    3
                ),
        })

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    if not y_true:

        print(
            "No images could be evaluated."
        )

        return

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    correct_count = sum(
        result["correct"]
        for result in results
    )

    total_count = len(
        results
    )

    average_ms = (
        total_inference_time
        / total_count
        * 1000
    )

    # --------------------------------------------------------
    # Overall result
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("REAL-WORLD RESULTS")
    print("=" * 70)

    print()
    print(
        f"Images tested      : "
        f"{total_count:,}"
    )

    print(
        f"Correct            : "
        f"{correct_count:,}"
    )

    print(
        f"Incorrect          : "
        f"{total_count - correct_count:,}"
    )

    print()

    print(
        f"Accuracy           : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Macro F1           : "
        f"{macro_f1 * 100:.2f}%"
    )

    print(
        f"Weighted F1        : "
        f"{weighted_f1 * 100:.2f}%"
    )

    print(
        f"Average inference  : "
        f"{average_ms:.2f} ms/image"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    target_names = [
        LABELS[class_id]
        for class_id in CLASS_IDS
    ]

    print()

    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(
                range(NUM_CLASSES)
            ),
            target_names=target_names,
            digits=4,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(
            range(NUM_CLASSES)
        ),
    )

    matrix_file = (
        RESULT_DIR
        / "confusion_matrix.csv"
    )

    with open(
        matrix_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "Actual \\ Predicted"
            ]
            + target_names
        )

        for row_index, row in enumerate(
            matrix
        ):

            writer.writerow(
                [
                    target_names[
                        row_index
                    ]
                ]
                + row.tolist()
            )

    # --------------------------------------------------------
    # Per-image results
    # --------------------------------------------------------

    results_file = (
        RESULT_DIR
        / "real_world_predictions.csv"
    )

    with open(
        results_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        fieldnames = [
            "image",
            "actual_class",
            "actual_name",
            "predicted_class",
            "predicted_name",
            "confidence",
            "correct",
            "inference_ms",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    # --------------------------------------------------------
    # Incorrect predictions
    # --------------------------------------------------------

    wrong_results = [
        result
        for result in results
        if not result["correct"]
    ]

    wrong_file = (
        RESULT_DIR
        / "incorrect_predictions.csv"
    )

    with open(
        wrong_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        fieldnames = [
            "image",
            "actual_class",
            "actual_name",
            "predicted_class",
            "predicted_name",
            "confidence",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in wrong_results:

            writer.writerow({

                key: result[key]
                for key in fieldnames

            })

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Prediction results:"
    )

    print(
        results_file
    )

    print()
    print(
        f"Incorrect predictions:"
    )

    print(
        wrong_file
    )

    print()
    print(
        f"Confusion matrix:"
    )

    print(
        matrix_file
    )

    print()


if __name__ == "__main__":

    main()