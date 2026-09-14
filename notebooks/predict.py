"""Predict one or more plant images with a FieldCare checkpoint.

Examples:

    python notebooks/predict.py leaf.jpg
    python notebooks/predict.py leaf1.jpg leaf2.jpg --top-k 3
    python notebooks/predict.py leaf.jpg --checkpoint models/train/RUN/fieldcare_efficientnet_b3_best.pth

When --checkpoint is omitted, the newest best checkpoint is selected.
Labels, image size, and normalization are loaded from the checkpoint so the
prediction mapping cannot drift from training.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b3


ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = ROOT / "models" / "train"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "predict"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class PredictionError(RuntimeError):
    """Raised when a checkpoint or input image is invalid."""


def find_latest_checkpoint() -> Path:
    # Training bundles are commonly extracted into output/ or outputs/, while
    # local training runs store checkpoints under models/train/<run>/.
    # Also include the standard absolute Jupyter/OVH workspace locations.
    roots = [
        ROOT / "output",
        ROOT / "outputs",
        Path.cwd() / "output",
        Path.cwd() / "outputs",
        MODEL_ROOT,
        ROOT / "models" / "train_v2",  # Backward compatibility with old runs.
        Path("/output"),
        Path("/outputs"),
        Path("/workspace/output"),
        Path("/workspace/outputs"),
    ]
    unique_roots: list[Path] = []
    seen_roots: set[str] = set()
    for root in roots:
        key = str(root.resolve())
        if key not in seen_roots:
            seen_roots.add(key)
            unique_roots.append(root)

    candidates: list[Path] = []
    for root in unique_roots:
        if root.is_dir():
            candidates.extend(root.rglob("fieldcare_efficientnet_b3*.pth"))
            candidates.extend(root.rglob("fieldcare_v2_efficientnet_b3*.pth"))

    # A best checkpoint must win over a final checkpoint even when the final
    # file was written later. Among checkpoints of the same kind, use newest.
    best = [path for path in candidates if "best" in path.stem.casefold()]
    final = [path for path in candidates if "final" in path.stem.casefold()]
    preferred = best or final or candidates
    if not preferred:
        searched = "\n  - ".join(str(path.resolve()) for path in unique_roots)
        raise PredictionError(
            "No FieldCare .pth checkpoint was found. Searched:\n"
            f"  - {searched}\n"
            "Place the checkpoint in output/ or outputs/, or pass --checkpoint."
        )
    return max(preferred, key=lambda path: path.stat().st_mtime).resolve()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise PredictionError("CUDA was requested but is unavailable")
    return torch.device(requested)


def load_checkpoint(path: Path, device: torch.device):
    if not path.is_file():
        raise PredictionError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    required = {"model_state_dict", "num_classes", "classes", "image_size"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise PredictionError(
            f"Checkpoint is not a FieldCare checkpoint; missing "
            f"{sorted(required - set(checkpoint if isinstance(checkpoint, dict) else []))}"
        )

    architecture = checkpoint.get("architecture", "efficientnet_b3")
    if architecture != "efficientnet_b3":
        raise PredictionError(
            f"Unsupported checkpoint architecture: {architecture!r}; "
            "expected 'efficientnet_b3'"
        )

    classes = checkpoint["classes"]
    num_classes = int(checkpoint["num_classes"])
    if len(classes) != num_classes:
        raise PredictionError("Checkpoint class metadata length is inconsistent")
    if [int(item["index"]) for item in classes] != list(range(num_classes)):
        raise PredictionError("Checkpoint class indices are not contiguous from zero")

    model = efficientnet_b3(weights=None)
    input_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(input_features, num_classes),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    return model, checkpoint, classes


def build_transform(checkpoint: dict):
    image_size = int(checkpoint["image_size"])
    normalization = checkpoint.get(
        "normalization",
        {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=normalization["mean"], std=normalization["std"]
            ),
        ]
    )


def predict_image(
    model,
    image_path: Path,
    transform,
    classes: list[dict],
    device: torch.device,
    top_k: int,
) -> dict:
    if not image_path.is_file():
        raise PredictionError(f"Image not found: {image_path}")
    if image_path.suffix.lower() not in VALID_EXTENSIONS:
        raise PredictionError(f"Unsupported image extension: {image_path}")

    try:
        with Image.open(image_path) as image:
            return predict_pil_image(
                model,
                image,
                transform,
                classes,
                device,
                top_k,
                image_name=str(image_path.resolve()),
            )
    except Exception as exc:
        if isinstance(exc, PredictionError):
            raise
        raise PredictionError(f"Could not read image: {image_path}") from exc


def predict_pil_image(
    model,
    image: Image.Image,
    transform,
    classes: list[dict],
    device: torch.device,
    top_k: int = 5,
    image_name: str = "uploaded-image",
) -> dict:
    """Predict an already decoded image without writing it to disk."""
    if top_k < 1:
        raise PredictionError("top_k must be at least 1")

    try:
        tensor = transform(image.convert("RGB")).unsqueeze(0)
    except Exception as exc:
        raise PredictionError(f"Could not process image: {image_name}") from exc

    tensor = tensor.to(device)
    if device.type == "cuda":
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    elapsed_ms = (time.perf_counter() - started) * 1000

    count = min(top_k, len(classes))
    values, indices = torch.topk(probabilities, count)
    predictions = []
    for rank, (value, index_tensor) in enumerate(zip(values, indices), start=1):
        class_info = classes[index_tensor.item()]
        predictions.append(
            {
                "rank": rank,
                "class_id": int(class_info["id"]),
                "class_index": int(class_info["index"]),
                "class_key": class_info["key"],
                "display_name": class_info["display_name"],
                "crop": class_info["crop"],
                "disease": class_info["disease"],
                "confidence": float(value.item()),
                "confidence_percent": float(value.item() * 100),
            }
        )
    return {
        "image": image_name,
        "inference_ms": elapsed_ms,
        "predictions": predictions,
    }


def print_result(result: dict, minimum_confidence: float) -> None:
    print("\n" + "=" * 72)
    print(result["image"])
    print("=" * 72)
    for prediction in result["predictions"]:
        print(
            f"{prediction['rank']}. {prediction['display_name']:<52} "
            f"{prediction['confidence_percent']:6.2f}%"
        )
    best = result["predictions"][0]
    print(f"Inference: {result['inference_ms']:.2f} ms")
    if best["confidence"] < minimum_confidence:
        print(
            f"WARNING: confidence is below {minimum_confidence * 100:.1f}%; "
            "treat this prediction as uncertain."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict plant diseases with a FieldCare checkpoint."
    )
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--minimum-confidence", type=float, default=0.50,
        help="Print an uncertainty warning below this probability",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.top_k < 1:
            raise PredictionError("--top-k must be at least 1")
        if not 0 <= args.minimum_confidence <= 1:
            raise PredictionError("--minimum-confidence must be between 0 and 1")
        device = select_device(args.device)
        checkpoint_path = (
            args.checkpoint.resolve() if args.checkpoint else find_latest_checkpoint()
        )
        model, checkpoint, classes = load_checkpoint(checkpoint_path, device)
        transform = build_transform(checkpoint)

        print(f"Checkpoint: {checkpoint_path}")
        print(f"Device    : {device}")
        print(f"Classes   : {len(classes)}")
        results = [
            predict_image(
                model, image.resolve(), transform, classes, device, args.top_k
            )
            for image in args.images
        ]
        for result in results:
            print_result(result, args.minimum_confidence)

        output = args.output
        if output is None and len(results) > 1:
            output = DEFAULT_OUTPUT_ROOT / (
                datetime.now().strftime("%Y%m%d_%H%M%S") + "_predictions.json"
            )
        if output:
            output = output.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "checkpoint": str(checkpoint_path),
                "device": str(device),
                "results": results,
            }
            output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(f"\nSaved: {output}")
        return 0
    except (OSError, PredictionError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
