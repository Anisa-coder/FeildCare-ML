from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import combinations
from pathlib import Path
from statistics import mean, median

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow is required.")
    print("Install with: pip install pillow")
    raise

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9.1
    RESAMPLE = Image.LANCZOS


# ============================================================================
# PATHS  (unchanged — do not move these)
# ============================================================================

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"

LOGS_ROOT = ROOT / "logs"
REPORT_DIR = LOGS_ROOT / "_dataset_reports"
REPORT_JSON = REPORT_DIR / "dataset_audit.json"

# New in v3: a small, training-script-ready config dropped next to the
# full audit. Your train.py can load this directly instead of
# hard-coding input size / normalization / class weights.
TRAINING_CONFIG_JSON = REPORT_DIR / "effnet_b3_training_config.json"

DEFAULT_LOG_DIR = LOGS_ROOT / "dataset_audit"

LABEL_FILE = DATASET / "label.txt"


def load_labels() -> dict[int, str]:
    """Load class IDs and names directly from dataset/label.txt."""
    if not LABEL_FILE.exists():
        raise FileNotFoundError(f"label.txt not found: {LABEL_FILE}")

    labels: dict[int, str] = {}

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
                "Expected: <number> <class name>"
            )

        class_id = int(parts[0])
        class_name = parts[1].strip()

        if class_id <= 0:
            raise ValueError(f"Invalid class ID {class_id} on line {line_number}.")

        if class_id in labels:
            raise ValueError(f"Duplicate class ID {class_id} in label.txt.")

        labels[class_id] = class_name

    if not labels:
        raise ValueError(f"No classes found in {LABEL_FILE}")

    return dict(sorted(labels.items()))


LABELS = load_labels()


# ============================================================================
# EFFICIENTNET-B3 MODEL CONFIG
# ============================================================================
# Source: Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs" (2019).
# B3 is compound-scaled to a native 300x300 input; params ~12M; ImageNet
# top-1 ~81.6%. These numbers don't change — they're fixed by the published
# architecture, not by which framework/library you load it from.

MODEL_NAME = "EfficientNet-B3"
EFFNET_INPUT_SIZE = 300          # native training resolution
EFFNET_RESIZE_SIZE = 320         # common "resize-then-crop" convention (resize
                                  # shorter side to this, then crop to 300x300)
EFFNET_PARAMS_MILLIONS = 12
EFFNET_IMAGENET_TOP1 = 81.6

# Standard ImageNet normalization — correct for torchvision.models and timm
# pretrained weights. NOTE: if you're using tf.keras.applications.EfficientNetB3,
# its preprocess_input is a pass-through (rescaling is baked into the model as
# a Rescaling layer) — don't apply these stats on top of that, or you'll
# double-normalize. Check whichever library you actually train with.
EFFNET_NORM_MEAN = (0.485, 0.456, 0.406)
EFFNET_NORM_STD = (0.229, 0.224, 0.225)

# Resolution tiers, all relative to EFFNET_INPUT_SIZE.
MIN_ACCEPTABLE_SIZE = EFFNET_INPUT_SIZE          # 300 — below this, every
                                                  # image needs upscaling
IDEAL_SOURCE_SIZE = int(EFFNET_INPUT_SIZE * 1.5)  # 450 — comfortable margin
                                                   # for RandomResizedCrop
                                                   # augmentation without
                                                   # magnifying artifacts
SEVERELY_UNDERSIZED = EFFNET_INPUT_SIZE // 2      # 150 — needs >2x upscale,
                                                   # flag for manual review
MIN_PIXELS = MIN_ACCEPTABLE_SIZE * MIN_ACCEPTABLE_SIZE

# Aspect ratios outside this range lose significant content on a
# center/random square crop and are flagged (model-agnostic).
MIN_ASPECT_RATIO = 0.35
MAX_ASPECT_RATIO = 2.85

# Per-class sample-size rules of thumb for fine-tuning a pretrained B3.
MIN_RECOMMENDED_PER_CLASS = 150   # below this, expect high variance / risk
                                   # of overfitting that class
IDEAL_PER_CLASS = 400             # comfortable for fine-tuning without
                                   # heavy augmentation reliance

# File-level sanity thresholds.
LARGE_FILE_MB = 10
TINY_FILE_KB = 5

# Perceptual (visual) duplicate detection.
PHASH_SIZE = 8                     # -> 64-bit dHash
NEAR_DUP_HAMMING_THRESHOLD = 5      # <=5 bits different = "looks the same"
NEAR_DUP_MAX_CLASS_SIZE = 2500       # skip pairwise near-dup scan for a class
                                      # larger than this (cost is O(n^2));
                                      # exact-dHash matching still runs for it

MAX_EXAMPLES = 15
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ============================================================================
# HELPERS
# ============================================================================

def label_for(class_id: int) -> str:
    return LABELS.get(class_id, f"Unknown Class {class_id}")


def pct(value: float, total: float) -> str:
    if total <= 0:
        return "0.00%"
    return f"{(value / total) * 100:.2f}%"


def ratio_text(a: float, b: float) -> str:
    if b <= 0:
        return "N/A"
    return f"{a / b:.2f}:1"


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_subsection(title: str) -> None:
    print()
    print("-" * 80)
    print(title)
    print("-" * 80)


def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in EXTENSIONS


def iter_class_images():
    if not DATASET.exists():
        return

    for folder in DATASET.iterdir():
        if not folder.is_dir() or not folder.name.isdigit():
            continue

        class_id = int(folder.name)

        for image in folder.rglob("*"):
            if is_image(image):
                yield class_id, image


def human_size(size_bytes: float) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(size_bytes)

    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size_bytes} B"


def safe_mean(values):
    return mean(values) if values else 0


def safe_median(values):
    return median(values) if values else 0


def percentile(values, p: float):
    if not values:
        return 0

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def compute_dhash(image: Image.Image, hash_size: int = PHASH_SIZE) -> int:
    """64-bit difference hash. Robust to resizing/recompression, which is
    exactly the kind of near-duplicate a byte-exact SHA-256 check misses
    (e.g. the same leaf photo pulled from two different aggregated
    datasets at different JPEG quality levels)."""
    small = image.convert("L").resize((hash_size + 1, hash_size), RESAMPLE)
    # .tobytes() on an "L" image gives one byte (0-255) per pixel, row-major —
    # equivalent to getdata() here but without the Pillow 14 deprecation and
    # without depending on a Pillow version new enough to have get_flattened_data.
    pixels = list(small.tobytes())

    bits = 0
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        row_pixels = pixels[offset : offset + hash_size + 1]
        for col in range(hash_size):
            bits = (bits << 1) | (1 if row_pixels[col] > row_pixels[col + 1] else 0)

    return bits


# ============================================================================
# DATA COLLECTION
# ============================================================================

def collect_files():
    class_files = defaultdict(list)

    for class_id, path in iter_class_images():
        class_files[class_id].append(path)

    for class_id in class_files:
        class_files[class_id].sort()

    return dict(class_files)


def inspect_image(path: Path) -> dict:
    """Single-pass image inspection.

    v2 opened every file up to 4 times (stat, full read for sha256,
    Image.open+verify, Image.open+decode). This reads the file once,
    hashes the in-memory bytes, and reuses the same bytes for both the
    PIL verify pass and the real decode — no repeat disk I/O.
    """

    result = {
        "path": str(path),
        "relative_path": safe_relative(path),
        "valid": False,
        "width": None,
        "height": None,
        "pixels": None,
        "aspect_ratio": None,
        "format": path.suffix.lower().lstrip("."),
        "mode": None,
        "channels": None,
        "file_size": 0,
        "sha256": None,
        "dhash": None,
        "error": None,
        "small": False,               # below EFFNET_INPUT_SIZE (300x300)
        "below_ideal_resolution": False,  # below IDEAL_SOURCE_SIZE (450x450)
        "severely_undersized": False,     # below SEVERELY_UNDERSIZED (150x150)
        "extreme_aspect": False,
        "tiny_file": False,
        "large_file": False,
        "grayscale": False,
        "has_alpha": False,
    }

    try:
        data = path.read_bytes()

        result["file_size"] = len(data)
        result["tiny_file"] = len(data) < TINY_FILE_KB * 1024
        result["large_file"] = len(data) > LARGE_FILE_MB * 1024 * 1024
        result["sha256"] = hashlib.sha256(data).hexdigest()

        # verify() checks the encoded stream without fully decoding pixels;
        # it invalidates the file object afterwards so we reopen from the
        # same in-memory bytes (no extra disk read either way).
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()

        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size

            result["width"] = width
            result["height"] = height
            result["pixels"] = width * height
            result["aspect_ratio"] = width / height if height else None
            result["format"] = (image.format or path.suffix.lstrip(".")).lower()
            result["mode"] = image.mode

            channels = {
                "1": 1, "L": 1, "LA": 2, "P": 1, "RGB": 3,
                "RGBA": 4, "CMYK": 4, "YCbCr": 3, "I": 1, "F": 1,
            }.get(image.mode)
            result["channels"] = channels

            result["grayscale"] = image.mode in {"1", "L", "LA"}
            result["has_alpha"] = (
                "A" in image.mode
                or (image.mode == "P" and "transparency" in image.info)
            )

            result["dhash"] = compute_dhash(image)

        result["small"] = (
            width < MIN_ACCEPTABLE_SIZE
            or height < MIN_ACCEPTABLE_SIZE
            or result["pixels"] < MIN_PIXELS
        )
        result["below_ideal_resolution"] = (
            width < IDEAL_SOURCE_SIZE or height < IDEAL_SOURCE_SIZE
        )
        result["severely_undersized"] = (
            width < SEVERELY_UNDERSIZED or height < SEVERELY_UNDERSIZED
        )

        ratio = result["aspect_ratio"]
        result["extreme_aspect"] = ratio is not None and (
            ratio < MIN_ASPECT_RATIO or ratio > MAX_ASPECT_RATIO
        )

        result["valid"] = True

    except Exception as error:
        result["error"] = str(error)

    return result


# ============================================================================
# DISTRIBUTION
# ============================================================================

def audit_distribution(class_files: dict) -> dict:
    counts = {class_id: len(class_files.get(class_id, [])) for class_id in LABELS}

    for class_id in class_files:
        if class_id not in counts:
            counts[class_id] = len(class_files[class_id])

    present = [count for count in counts.values() if count > 0]
    total = sum(counts.values())
    minimum = min(present) if present else 0
    maximum = max(present) if present else 0

    return {
        "counts": counts,
        "total": total,
        "minimum": minimum,
        "maximum": maximum,
        "ratio": (maximum / minimum) if minimum else None,
    }


# ============================================================================
# IMAGE QUALITY (parallelized)
# ============================================================================

def audit_image_quality(class_files: dict, workers: int) -> tuple[dict, list[dict]]:
    all_items = [
        (class_id, path)
        for class_id, images in class_files.items()
        for path in images
    ]

    print_subsection("IMAGE VALIDATION / QUALITY SCAN")
    print(f"Images to inspect : {len(all_items):,}")
    print(f"Worker processes  : {workers}")

    records: list[dict] = []

    if not all_items:
        return {}, records

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(inspect_image, path): class_id
            for class_id, path in all_items
        }

        completed = 0
        for future in as_completed(futures):
            record = future.result()
            record["class_id"] = futures[future]
            records.append(record)

            completed += 1
            if completed % 1000 == 0 or completed == len(futures):
                print(f"  Processed {completed:,}/{len(futures):,} images...")

    stats_by_class = {}

    for class_id in class_files:
        class_records = [r for r in records if r["class_id"] == class_id]
        valid = [r for r in class_records if r["valid"]]

        widths = [r["width"] for r in valid if r["width"] is not None]
        heights = [r["height"] for r in valid if r["height"] is not None]
        pixels = [r["pixels"] for r in valid if r["pixels"] is not None]
        file_sizes = [r["file_size"] for r in valid]

        stats_by_class[class_id] = {
            "total": len(class_records),
            "valid": len(valid),
            "corrupt": sum(not r["valid"] for r in class_records),
            "small": sum(r["small"] for r in valid),
            "below_ideal_resolution": sum(r["below_ideal_resolution"] for r in valid),
            "severely_undersized": sum(r["severely_undersized"] for r in valid),
            "extreme_aspect": sum(r["extreme_aspect"] for r in valid),
            "grayscale": sum(r["grayscale"] for r in valid),
            "alpha": sum(r["has_alpha"] for r in valid),
            "tiny_files": sum(r["tiny_file"] for r in valid),
            "large_files": sum(r["large_file"] for r in valid),
            "width": {
                "min": min(widths) if widths else 0,
                "median": safe_median(widths),
                "mean": safe_mean(widths),
                "p10": percentile(widths, 0.10),
                "p90": percentile(widths, 0.90),
                "max": max(widths) if widths else 0,
            },
            "height": {
                "min": min(heights) if heights else 0,
                "median": safe_median(heights),
                "mean": safe_mean(heights),
                "p10": percentile(heights, 0.10),
                "p90": percentile(heights, 0.90),
                "max": max(heights) if heights else 0,
            },
            "pixels": {
                "min": min(pixels) if pixels else 0,
                "median": safe_median(pixels),
                "mean": safe_mean(pixels),
                "max": max(pixels) if pixels else 0,
            },
            "file_size": {
                "min": min(file_sizes) if file_sizes else 0,
                "median": safe_median(file_sizes),
                "mean": safe_mean(file_sizes),
                "max": max(file_sizes) if file_sizes else 0,
            },
            "formats": dict(Counter(r["format"] for r in valid)),
            "modes": dict(Counter(r["mode"] for r in valid)),
        }

    return stats_by_class, records


# ============================================================================
# DUPLICATES  (exact SHA-256 + perceptual dHash, both O(n))
# ============================================================================

def audit_duplicates(records: list[dict]) -> dict:
    print_subsection("DUPLICATE ANALYSIS (exact + visual)")

    valid_records = [r for r in records if r["valid"]]

    # ---- Exact byte-for-byte duplicates -----------------------------------
    hash_map = defaultdict(list)
    for record in valid_records:
        digest = record.get("sha256")
        if digest:
            hash_map[digest].append(record)

    exact_groups = {d: items for d, items in hash_map.items() if len(items) > 1}
    exact_cross_class = [
        (d, items)
        for d, items in exact_groups.items()
        if len({item["class_id"] for item in items}) > 1
    ]
    exact_within_class = [
        (d, items)
        for d, items in exact_groups.items()
        if len({item["class_id"] for item in items}) == 1
    ]
    exact_duplicate_files = sum(len(items) - 1 for items in exact_groups.values())

    # ---- Visual (perceptual) duplicates — exact dHash match, O(n) ---------
    dhash_map = defaultdict(list)
    for record in valid_records:
        digest = record.get("dhash")
        if digest is not None:
            dhash_map[digest].append(record)

    visual_groups = {d: items for d, items in dhash_map.items() if len(items) > 1}
    visual_cross_class = [
        (d, items)
        for d, items in visual_groups.items()
        if len({item["class_id"] for item in items}) > 1
    ]
    visual_within_class = [
        (d, items)
        for d, items in visual_groups.items()
        if len({item["class_id"] for item in items}) == 1
    ]
    visual_duplicate_files = sum(len(items) - 1 for items in visual_groups.values())

    # ---- Near-duplicates (Hamming distance <= threshold), bounded per class
    # Exact-dHash grouping above is O(n). Fuzzy matching is O(n^2) per class,
    # so it's only run within each class (not across the whole dataset) and
    # skipped for any class larger than NEAR_DUP_MAX_CLASS_SIZE.
    already_grouped = {
        id(item) for items in visual_groups.values() for item in items
    }

    by_class = defaultdict(list)
    for record in valid_records:
        if id(record) not in already_grouped and record.get("dhash") is not None:
            by_class[record["class_id"]].append(record)

    near_dup_pairs = []
    skipped_classes = []

    for class_id, items in by_class.items():
        if len(items) > NEAR_DUP_MAX_CLASS_SIZE:
            skipped_classes.append(class_id)
            continue

        for a, b in combinations(items, 2):
            distance = hamming_distance(a["dhash"], b["dhash"])
            if distance <= NEAR_DUP_HAMMING_THRESHOLD:
                near_dup_pairs.append((class_id, a, b, distance))

    print(f"Exact duplicate groups    : {len(exact_groups):,} "
          f"({exact_duplicate_files:,} redundant files)")
    print(f"  Cross-class             : {len(exact_cross_class):,}")
    print(f"  Within-class            : {len(exact_within_class):,}")
    print(f"Visual duplicate groups   : {len(visual_groups):,} "
          f"({visual_duplicate_files:,} redundant files)")
    print(f"  Cross-class             : {len(visual_cross_class):,}")
    print(f"  Within-class            : {len(visual_within_class):,}")
    print(f"Near-duplicate pairs      : {len(near_dup_pairs):,} "
          f"(Hamming <= {NEAR_DUP_HAMMING_THRESHOLD}, within-class only)")

    if skipped_classes:
        print(f"  Skipped near-dup scan for {len(skipped_classes)} class(es) "
              f"over {NEAR_DUP_MAX_CLASS_SIZE:,} images (exact-dHash still ran)")

    if exact_cross_class or visual_cross_class:
        print()
        print("WARNING: cross-class duplicates found — the same image exists")
        print("under two different labels. This is a labeling-correctness bug,")
        print("not just wasted disk space, and should be fixed before training.")

    return {
        "exact": {
            "unique_hashes": len(hash_map),
            "duplicate_groups": len(exact_groups),
            "duplicate_files": exact_duplicate_files,
            "cross_class_groups": len(exact_cross_class),
            "within_class_groups": len(exact_within_class),
            "groups": exact_groups,
            "cross_class_examples": [
                {
                    "sha256": digest,
                    "files": [
                        {"class_id": i["class_id"], "path": i["relative_path"]}
                        for i in items
                    ],
                }
                for digest, items in exact_cross_class[:MAX_EXAMPLES]
            ],
        },
        "visual": {
            "unique_hashes": len(dhash_map),
            "duplicate_groups": len(visual_groups),
            "duplicate_files": visual_duplicate_files,
            "cross_class_groups": len(visual_cross_class),
            "within_class_groups": len(visual_within_class),
            "groups": visual_groups,
            "cross_class_examples": [
                {
                    "dhash": f"{digest:016x}",
                    "files": [
                        {"class_id": i["class_id"], "path": i["relative_path"]}
                        for i in items
                    ],
                }
                for digest, items in visual_cross_class[:MAX_EXAMPLES]
            ],
        },
        "near_duplicates": {
            "pair_count": len(near_dup_pairs),
            "skipped_classes": skipped_classes,
            "examples": [
                {
                    "class_id": class_id,
                    "class_name": label_for(class_id),
                    "hamming_distance": distance,
                    "file_a": a["relative_path"],
                    "file_b": b["relative_path"],
                }
                for class_id, a, b, distance in near_dup_pairs[:MAX_EXAMPLES]
            ],
        },
    }


# ============================================================================
# GLOBAL QUALITY SUMMARY
# ============================================================================

def global_quality_summary(records: list[dict]) -> dict:
    valid = [r for r in records if r["valid"]]
    corrupt = [r for r in records if not r["valid"]]

    dimensions = [
        (r["width"], r["height"]) for r in valid if r["width"] and r["height"]
    ]
    ratios = [r["aspect_ratio"] for r in valid if r["aspect_ratio"]]
    sizes = [r["file_size"] for r in valid]

    return {
        "total_records": len(records),
        "valid": len(valid),
        "corrupt": len(corrupt),
        "small": sum(r["small"] for r in valid),
        "below_ideal_resolution": sum(r["below_ideal_resolution"] for r in valid),
        "severely_undersized": sum(r["severely_undersized"] for r in valid),
        "extreme_aspect": sum(r["extreme_aspect"] for r in valid),
        "grayscale": sum(r["grayscale"] for r in valid),
        "alpha": sum(r["has_alpha"] for r in valid),
        "tiny_files": sum(r["tiny_file"] for r in valid),
        "large_files": sum(r["large_file"] for r in valid),
        "formats": dict(Counter(r["format"] for r in valid)),
        "modes": dict(Counter(r["mode"] for r in valid)),
        "dimensions": {
            "min_width": min((x[0] for x in dimensions), default=0),
            "median_width": safe_median([x[0] for x in dimensions]),
            "min_height": min((x[1] for x in dimensions), default=0),
            "median_height": safe_median([x[1] for x in dimensions]),
            "min_pixels": min((r["pixels"] for r in valid if r["pixels"]), default=0),
            "median_pixels": safe_median([r["pixels"] for r in valid if r["pixels"]]),
        },
        "aspect_ratio": {
            "minimum": min(ratios, default=0),
            "median": safe_median(ratios),
            "maximum": max(ratios, default=0),
        },
        "file_size": {
            "total_bytes": sum(sizes),
            "median_bytes": safe_median(sizes),
            "mean_bytes": safe_mean(sizes),
            "maximum_bytes": max(sizes, default=0),
        },
        "corrupt_examples": [r["relative_path"] for r in corrupt[:MAX_EXAMPLES]],
        "small_examples": [
            r["relative_path"] for r in valid if r["small"]
        ][:MAX_EXAMPLES],
        "severely_undersized_examples": [
            r["relative_path"] for r in valid if r["severely_undersized"]
        ][:MAX_EXAMPLES],
        "aspect_examples": [
            r["relative_path"] for r in valid if r["extreme_aspect"]
        ][:MAX_EXAMPLES],
        "tiny_file_examples": [
            r["relative_path"] for r in valid if r["tiny_file"]
        ][:MAX_EXAMPLES],
    }


# ============================================================================
# IMBALANCE / COMPLETENESS
# ============================================================================

def imbalance_analysis(counts: dict) -> dict:
    nonzero = {cid: c for cid, c in counts.items() if c > 0}

    if not nonzero:
        return {
            "median": 0, "mean": 0, "max": 0, "min": 0, "max_to_min": None,
            "classes_below_25pct_median": [], "classes_below_50pct_median": [],
            "classes_above_2x_median": [],
        }

    values = list(nonzero.values())
    med = median(values)

    return {
        "median": med,
        "mean": mean(values),
        "max": max(values),
        "min": min(values),
        "max_to_min": (max(values) / min(values)) if min(values) else None,
        "classes_below_25pct_median": [c for c, n in nonzero.items() if n < med * 0.25],
        "classes_below_50pct_median": [c for c, n in nonzero.items() if n < med * 0.50],
        "classes_above_2x_median": [c for c, n in nonzero.items() if n > med * 2],
    }


def completeness_analysis(counts: dict) -> dict:
    expected = set(LABELS)
    actual = set(counts)

    return {
        "expected_classes": len(expected),
        "classes_with_images": sum(counts.get(c, 0) > 0 for c in expected),
        "empty_classes": sorted(c for c in expected if counts.get(c, 0) == 0),
        "unexpected_numeric_folders": sorted(c for c in actual if c not in expected),
    }


def compute_class_weights(counts: dict) -> dict:
    """sklearn-style 'balanced' weights: n_samples / (n_classes * count).
    Feed this straight into nn.CrossEntropyLoss(weight=...) (PyTorch) or
    class_weight=... (Keras/scikit-learn)."""
    nonzero = {cid: c for cid, c in counts.items() if c > 0}
    if not nonzero:
        return {}

    n_samples = sum(nonzero.values())
    n_classes = len(nonzero)

    return {cid: round(n_samples / (n_classes * count), 4) for cid, count in nonzero.items()}


# ============================================================================
# EFFICIENTNET-B3 TRAINING READINESS
# ============================================================================
# Full batch-size / fine-tuning detail lives in the JSON config
# (TRAINING_CONFIG_JSON) for the training script to consume directly.
# The console report only prints short, aggregate stats — no per-image
# detail, no walls of prose.

BATCH_SIZE_GUIDANCE = [
    ("~6 GB  (e.g. GTX 1660, RTX 2060)", 8),
    ("~8 GB  (e.g. RTX 3050/3060 8GB)", 16),
    ("~12 GB (e.g. RTX 3060 12GB, RTX 4070)", 24),
    ("~16 GB (e.g. T4, RTX 4060 Ti 16GB)", 32),
    ("~24 GB (e.g. RTX 3090/4090, A10, A5000)", 48),
    ("~40 GB+ (e.g. A100)", 64),
]

FINE_TUNING_NOTES = [
    "Start from ImageNet-pretrained weights, replace the classifier head "
    "with num_classes outputs.",
    "Phase 1: freeze backbone, train head only, LR ~1e-3.",
    "Phase 2: unfreeze all, fine-tune at LR ~1e-4 to 1e-5, cosine/step schedule.",
    "Use label smoothing (~0.1) for multi-source aggregated label noise.",
    "Use mixed precision (AMP).",
]

SUGGESTED_SPLIT_SHORT = "70/15/15 or 80/10/10, stratified by class"


def effnet_readiness_report(
    quality: dict, distribution: dict, imbalance: dict
) -> dict:
    total_valid = max(1, quality["valid"])
    below_target = quality["small"]
    below_ideal = quality["below_ideal_resolution"]
    severe = quality["severely_undersized"]
    at_or_above_ideal = total_valid - below_ideal

    counts = distribution["counts"]
    below_min_class = sorted(
        [c for c, n in counts.items() if 0 < n < MIN_RECOMMENDED_PER_CLASS]
    )
    below_ideal_class = sorted(
        [c for c, n in counts.items() if 0 < n < IDEAL_PER_CLASS]
    )

    class_weights = compute_class_weights(counts)

    return {
        "model": MODEL_NAME,
        "input_size": EFFNET_INPUT_SIZE,
        "resize_size": EFFNET_RESIZE_SIZE,
        "normalization_mean": EFFNET_NORM_MEAN,
        "normalization_std": EFFNET_NORM_STD,
        "normalization_note": (
            "Standard ImageNet stats — correct for torchvision.models / timm "
            "pretrained weights. tf.keras.applications.EfficientNetB3 bakes "
            "rescaling into the model itself; don't double-normalize if using Keras."
        ),
        "resolution": {
            "at_or_above_ideal_450px": at_or_above_ideal,
            "at_or_above_ideal_450px_pct": pct(at_or_above_ideal, total_valid),
            "below_target_300px": below_target,
            "below_target_300px_pct": pct(below_target, total_valid),
            "severely_undersized_150px": severe,
            "severely_undersized_150px_pct": pct(severe, total_valid),
        },
        "class_sample_size": {
            "min_recommended_per_class": MIN_RECOMMENDED_PER_CLASS,
            "ideal_per_class": IDEAL_PER_CLASS,
            "classes_below_minimum": below_min_class,
            "classes_below_ideal": below_ideal_class,
        },
        "class_weights": class_weights,
        "imbalance_ratio": ratio_text(imbalance["max"], imbalance["min"]),
        "batch_size_guidance": [
            {"gpu_memory": label, "suggested_batch_size": size}
            for label, size in BATCH_SIZE_GUIDANCE
        ],
        "fine_tuning_notes": FINE_TUNING_NOTES,
        "suggested_split": SUGGESTED_SPLIT_SHORT,
    }


def print_effnet_readiness(readiness: dict) -> None:
    print_section(f"6. {readiness['model'].upper()} TRAINING READINESS")

    print(f"Target input resolution : {readiness['input_size']}x{readiness['input_size']}")
    print(f"Resize-then-crop        : resize shorter side to "
          f"{readiness['resize_size']}, crop to {readiness['input_size']}")
    print(f"Normalization mean      : {readiness['normalization_mean']}")
    print(f"Normalization std       : {readiness['normalization_std']}")

    res = readiness["resolution"]
    print()
    print("Resolution readiness:")
    print(f"  >= ideal (450px)  : {res['at_or_above_ideal_450px']:,} ({res['at_or_above_ideal_450px_pct']})")
    print(f"  < target (300px)  : {res['below_target_300px']:,} ({res['below_target_300px_pct']})")
    print(f"  severely undersized (<150px) : {res['severely_undersized_150px']:,} "
          f"({res['severely_undersized_150px_pct']})")

    cls = readiness["class_sample_size"]
    below_min = cls["classes_below_minimum"]
    small_class_text = (
        ", ".join(f"{c} ({label_for(c)})" for c in below_min) if below_min else "none"
    )

    print()
    print("Training notes:")
    print(f"  Small class size (< {cls['min_recommended_per_class']}) : {small_class_text}")
    print(f"  Imbalance ratio (max:min)            : {readiness['imbalance_ratio']}")
    print(f"  Suggested split                      : {readiness['suggested_split']}")

    print()
    print(f"Class weights, batch-size guidance, and fine-tuning notes written to:\n  {TRAINING_CONFIG_JSON}")


# ============================================================================
# REPORT OUTPUT
# ============================================================================

def print_distribution_report(distribution: dict) -> None:
    print_section("1. CLASS DISTRIBUTION")
    print(f"{'ID':>3}  {'Class':<60} {'Images':>10}")
    print("-" * 85)

    for class_id in sorted(LABELS):
        count = distribution["counts"].get(class_id, 0)
        status = "EMPTY" if count == 0 else ""
        print(f"{class_id:>3}  {label_for(class_id):<60} {count:>10,} {status}")

    unexpected = [c for c in distribution["counts"] if c not in LABELS]
    if unexpected:
        print()
        print("NUMERIC FOLDERS NOT DEFINED IN label.txt:")
        for class_id in sorted(unexpected):
            print(f"  {class_id}: {distribution['counts'][class_id]:,} images")

    print("-" * 85)
    print(f"TOTAL IMAGES: {distribution['total']:,}")


def print_imbalance_report(distribution: dict) -> None:
    analysis = imbalance_analysis(distribution["counts"])

    print_section("2. CLASS IMBALANCE ANALYSIS")
    print(f"Smallest non-empty class : {analysis['min']:,}")
    print(f"Largest class            : {analysis['max']:,}")
    print(f"Median class size        : {analysis['median']:,.0f}")
    print(f"Mean class size          : {analysis['mean']:,.0f}")

    if analysis["max_to_min"] is not None:
        print(f"Max / Min ratio          : {ratio_text(analysis['max'], analysis['min'])}")

    if analysis["classes_below_50pct_median"]:
        print()
        print("Classes below 50% of median:")
        for class_id in analysis["classes_below_50pct_median"]:
            print(f"  {class_id}: {label_for(class_id)} "
                  f"({distribution['counts'][class_id]:,})")

    if analysis["classes_above_2x_median"]:
        print()
        print("Classes above 2x median:")
        for class_id in analysis["classes_above_2x_median"]:
            print(f"  {class_id}: {label_for(class_id)} "
                  f"({distribution['counts'][class_id]:,})")


def print_quality_report(quality: dict) -> None:
    print_section("3. IMAGE QUALITY / INTEGRITY")
    total = max(1, quality["total_records"])

    print(f"Valid images            : {quality['valid']:,} ({pct(quality['valid'], total)})")
    print(f"Corrupt/unreadable      : {quality['corrupt']:,} ({pct(quality['corrupt'], total)})")
    print(f"Below B3 target (300px) : {quality['small']:,} ({pct(quality['small'], total)})")
    print(f"Below ideal (450px)     : {quality['below_ideal_resolution']:,} "
          f"({pct(quality['below_ideal_resolution'], total)})")
    print(f"Severely undersized     : {quality['severely_undersized']:,} "
          f"({pct(quality['severely_undersized'], total)})")
    print(f"Extreme aspect ratio    : {quality['extreme_aspect']:,} "
          f"({pct(quality['extreme_aspect'], total)})")
    print(f"Grayscale               : {quality['grayscale']:,} ({pct(quality['grayscale'], total)})")
    print(f"Images with alpha       : {quality['alpha']:,} ({pct(quality['alpha'], total)})")
    print(f"Tiny files              : {quality['tiny_files']:,}")
    print(f"Large files > {LARGE_FILE_MB} MB    : {quality['large_files']:,}")
    print(f"Total dataset size      : {human_size(quality['file_size']['total_bytes'])}")

    print()
    print("Dimensions:")
    print(f"  Minimum width  : {quality['dimensions']['min_width']}")
    print(f"  Median width   : {quality['dimensions']['median_width']:,.0f}")
    print(f"  Minimum height : {quality['dimensions']['min_height']}")
    print(f"  Median height  : {quality['dimensions']['median_height']:,.0f}")
    print(f"  Median pixels  : {quality['dimensions']['median_pixels']:,.0f}")

    print()
    print("Aspect ratio:")
    print(f"  Minimum : {quality['aspect_ratio']['minimum']:.3f}")
    print(f"  Median  : {quality['aspect_ratio']['median']:.3f}")
    print(f"  Maximum : {quality['aspect_ratio']['maximum']:.3f}")

    print()
    print("Formats:")
    for fmt, count in sorted(quality["formats"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {fmt:<10} {count:>8,} ({pct(count, quality['valid'])})")

    print()
    print("Image modes:")
    for mode, count in sorted(quality["modes"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {str(mode):<10} {count:>8,} ({pct(count, quality['valid'])})")


def print_class_quality_report(stats_by_class: dict) -> None:
    print_section("4. PER-CLASS QUALITY PROFILE")
    print(f"{'ID':>3} {'Class':<40} {'Valid':>8} {'Bad':>6} {'<300px':>7} "
          f"{'Gray':>7} {'Aspect':>8}")
    print("-" * 95)

    for class_id in sorted(LABELS):
        stats = stats_by_class.get(class_id, {
            "valid": 0, "corrupt": 0, "small": 0, "grayscale": 0, "extreme_aspect": 0,
        })

        print(f"{class_id:>3} {label_for(class_id):<40} {stats['valid']:>8,} "
              f"{stats['corrupt']:>6,} {stats['small']:>7,} {stats['grayscale']:>7,} "
              f"{stats['extreme_aspect']:>8,}")


def print_duplicate_report(duplicates: dict) -> None:
    print_section("5. DUPLICATE / DATA LEAKAGE ANALYSIS")

    exact = duplicates["exact"]
    visual = duplicates["visual"]
    near = duplicates["near_duplicates"]

    print("Exact (byte-identical, SHA-256):")
    print(f"  Unique hashes    : {exact['unique_hashes']:,}")
    print(f"  Duplicate groups : {exact['duplicate_groups']:,}")
    print(f"  Duplicate files  : {exact['duplicate_files']:,}")
    print(f"  Cross-class      : {exact['cross_class_groups']:,}")

    print()
    print("Visual (perceptual dHash, catches resized/recompressed copies):")
    print(f"  Unique hashes    : {visual['unique_hashes']:,}")
    print(f"  Duplicate groups : {visual['duplicate_groups']:,}")
    print(f"  Duplicate files  : {visual['duplicate_files']:,}")
    print(f"  Cross-class      : {visual['cross_class_groups']:,}")

    print()
    print(f"Near-duplicate pairs (Hamming <= {NEAR_DUP_HAMMING_THRESHOLD}, within-class): "
          f"{near['pair_count']:,}")
    if near["skipped_classes"]:
        skipped = ", ".join(label_for(c) for c in near["skipped_classes"])
        print(f"  Skipped (too large for pairwise scan): {skipped}")

    if exact["cross_class_groups"] or visual["cross_class_groups"]:
        print()
        print("!!! CRITICAL: cross-class duplicates found !!!")
        print("The same image (or a near-identical copy) appears under different labels.")

        for example in exact["cross_class_examples"]:
            print(f"\n  [exact] hash {example['sha256'][:16]}...")
            for item in example["files"]:
                print(f"    Class {item['class_id']} ({label_for(item['class_id'])}): {item['path']}")

        for example in visual["cross_class_examples"]:
            print(f"\n  [visual] dhash {example['dhash']}")
            for item in example["files"]:
                print(f"    Class {item['class_id']} ({label_for(item['class_id'])}): {item['path']}")


# ============================================================================
# DUPLICATE REVIEW / OPTIONAL REMOVAL
# ============================================================================

def remove_duplicates_interactively(duplicates: dict, interactive: bool) -> dict:
    """Show exact + visual duplicate groups and, if interactive, ask before
    deleting. With --no-interactive, this only reports (removes nothing) —
    useful for running the audit in a script/CI without blocking on input()."""

    result = {"exact_removed": 0, "visual_removed": 0, "removed_paths": []}

    exact_groups = duplicates["exact"]["groups"]
    visual_groups = duplicates["visual"]["groups"]

    if not exact_groups and not visual_groups:
        return result

    print_section("DUPLICATE FILE DETAILS")

    def show_and_maybe_remove(groups: dict, label: str, result_key: str) -> None:
        if not groups:
            return

        print()
        print(f"{label}: {len(groups):,} group(s)")

        for number, (digest, items) in enumerate(groups.items(), 1):
            print(f"\n[{label} group {number}] {digest if isinstance(digest, str) else f'{digest:016x}'}")
            for index, item in enumerate(items):
                status = "KEEP" if index == 0 else "DUPLICATE"
                print(f"  [{status}] Class {item['class_id']} ({label_for(item['class_id'])})")
                print(f"         {item['relative_path']}")

        if not interactive:
            print(f"  (--no-interactive: {label.lower()} were NOT removed)")
            return

        answer = input(
            f"Remove {label.lower()}, keeping the first file in each group? [y/N]: "
        ).strip().lower()

        if answer != "y":
            print(f"{label} were NOT removed.")
            return

        for digest, items in groups.items():
            for item in items[1:]:
                file_path = Path(item["path"])
                try:
                    if file_path.exists():
                        file_path.unlink()
                        result[result_key] += 1
                        result["removed_paths"].append(str(file_path))
                        print(f"Removed: {file_path}")
                except Exception as error:
                    print(f"Could not remove {file_path}: {error}")

    show_and_maybe_remove(exact_groups, "EXACT DUPLICATES", "exact_removed")
    show_and_maybe_remove(visual_groups, "VISUAL DUPLICATES", "visual_removed")

    print()
    print("Duplicate review complete.")
    print(f"Exact removed  : {result['exact_removed']:,}")
    print(f"Visual removed : {result['visual_removed']:,}")

    return result


# ============================================================================
# JSON REPORTS
# ============================================================================

def write_json_report(
    distribution: dict,
    imbalance: dict,
    completeness: dict,
    quality: dict,
    class_quality: dict,
    duplicates: dict,
    readiness: dict,
    health_score: int,
    removal_result: dict,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    duplicates_json = {
        "exact": {k: v for k, v in duplicates["exact"].items() if k != "groups"},
        "visual": {k: v for k, v in duplicates["visual"].items() if k != "groups"},
        "near_duplicates": duplicates["near_duplicates"],
    }

    report = {
        "dataset_root": str(DATASET),
        "total_images": distribution["total"],
        "distribution": distribution,
        "labels": LABELS,
        "imbalance": imbalance,
        "completeness": completeness,
        "quality": quality,
        "class_quality": class_quality,
        "duplicates": duplicates_json,
        "effnet_b3_readiness": readiness,
        "health_score": health_score,
        "duplicate_removal": removal_result,
        "configuration": {
            "min_acceptable_size": MIN_ACCEPTABLE_SIZE,
            "ideal_source_size": IDEAL_SOURCE_SIZE,
            "severely_undersized": SEVERELY_UNDERSIZED,
            "min_pixels": MIN_PIXELS,
            "min_aspect_ratio": MIN_ASPECT_RATIO,
            "max_aspect_ratio": MAX_ASPECT_RATIO,
            "large_file_mb": LARGE_FILE_MB,
            "tiny_file_kb": TINY_FILE_KB,
            "near_dup_hamming_threshold": NEAR_DUP_HAMMING_THRESHOLD,
        },
    }

    REPORT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def write_training_config(readiness: dict, distribution: dict) -> None:
    """Small, stable config a training script can load directly —
    separate from the full audit so it doesn't need to parse the whole
    report just to get input_size / normalization / class_weights."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "model": readiness["model"],
        "num_classes": len(LABELS),
        "class_names": {str(k): v for k, v in LABELS.items()},
        "input_size": readiness["input_size"],
        "resize_size": readiness["resize_size"],
        "normalization_mean": readiness["normalization_mean"],
        "normalization_std": readiness["normalization_std"],
        "class_weights": {str(k): v for k, v in readiness["class_weights"].items()},
        "class_counts": {str(k): v for k, v in distribution["counts"].items()},
        "suggested_batch_sizes": readiness["batch_size_guidance"],
        "suggested_split": readiness["suggested_split"],
    }

    TRAINING_CONFIG_JSON.write_text(json.dumps(config, indent=2), encoding="utf-8")


# ============================================================================
# HEALTH SCORE
# ============================================================================

def calculate_health_score(
    distribution: dict, quality: dict, duplicates: dict
) -> tuple[int, list[str]]:
    score = 100
    notes = []
    total = max(1, quality["total_records"])

    corrupt_rate = quality["corrupt"] / total
    if corrupt_rate > 0:
        deduction = min(15, math.ceil(corrupt_rate * 100))
        score -= deduction
        notes.append(f"-{deduction} corrupt-image penalty")

    exact = duplicates["exact"]
    visual = duplicates["visual"]

    dup_rate = exact["duplicate_files"] / total
    if dup_rate > 0:
        deduction = min(15, math.ceil(dup_rate * 50))
        score -= deduction
        notes.append(f"-{deduction} exact-duplicate penalty")

    if exact["cross_class_groups"] > 0:
        deduction = min(15, exact["cross_class_groups"] * 3)
        score -= deduction
        notes.append(f"-{deduction} exact cross-class-duplicate penalty")

    if visual["cross_class_groups"] > 0:
        deduction = min(15, visual["cross_class_groups"] * 2)
        score -= deduction
        notes.append(f"-{deduction} visual cross-class-duplicate penalty")

    small_rate = quality["small"] / total
    if small_rate > 0.10:
        score -= 8
        notes.append("-8 low-resolution penalty (>10% below 300px)")
    elif small_rate > 0.03:
        score -= 4
        notes.append("-4 low-resolution penalty (>3% below 300px)")

    severe_rate = quality["severely_undersized"] / total
    if severe_rate > 0.02:
        score -= 5
        notes.append("-5 severely-undersized penalty (>2% below 150px)")

    aspect_rate = quality["extreme_aspect"] / total
    if aspect_rate > 0.10:
        score -= 5
        notes.append("-5 aspect-ratio penalty")

    imbalance = imbalance_analysis(distribution["counts"])
    if imbalance["max_to_min"] is not None:
        if imbalance["max_to_min"] > 20:
            score -= 10
            notes.append("-10 severe class-imbalance penalty")
        elif imbalance["max_to_min"] > 10:
            score -= 6
            notes.append("-6 class-imbalance penalty")
        elif imbalance["max_to_min"] > 5:
            score -= 3
            notes.append("-3 class-imbalance penalty")

    return max(0, min(100, score)), notes


# ============================================================================
# CLI
# ============================================================================

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FieldCare dataset audit — tuned for EfficientNet-B3 training readiness."
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Parallel worker processes for image inspection (default: CPU count - 1)",
    )
    parser.add_argument(
        "--no-interactive", action="store_true",
        help="Report duplicates without prompting to delete anything (safe for CI/scripts).",
    )
    parser.add_argument(
        "--near-dup-threshold", type=int, default=NEAR_DUP_HAMMING_THRESHOLD,
        help=f"Hamming distance threshold for near-duplicate detection (0-64, default {NEAR_DUP_HAMMING_THRESHOLD}).",
    )
    parser.add_argument(
        "--logs-dir", type=Path, default=DEFAULT_LOG_DIR,
        help=f"Directory for timestamped audit logs (default: {DEFAULT_LOG_DIR}).",
    )
    return parser.parse_args(argv)


# ============================================================================
# MAIN
# ============================================================================

def run_audit(args: argparse.Namespace):

    global NEAR_DUP_HAMMING_THRESHOLD
    NEAR_DUP_HAMMING_THRESHOLD = args.near_dup_threshold

    workers = args.workers or max(1, (os.cpu_count() or 4) - 1)

    print("=" * 80)
    print("                 FIELDCARE DATASET AUDIT v3 (EfficientNet-B3)")
    print("=" * 80)

    print()
    print(f"Dataset root: {DATASET}")
    print(f"Label source: {LABEL_FILE}")
    print(f"Classes loaded from label.txt: {len(LABELS)}")

    if not DATASET.exists():
        print()
        print("ERROR: Dataset directory does not exist.")
        return 1

    print()
    print("Scanning class folders...")
    class_files = collect_files()

    distribution = audit_distribution(class_files)
    print_distribution_report(distribution)
    print_imbalance_report(distribution)

    completeness = completeness_analysis(distribution["counts"])
    print_section("DATASET COMPLETENESS")
    print(f"Expected classes     : {completeness['expected_classes']}")
    print(f"Classes with images  : {completeness['classes_with_images']}")

    if completeness["empty_classes"]:
        print()
        print("EMPTY / MISSING CLASSES:")
        for class_id in completeness["empty_classes"]:
            print(f"  {class_id}: {label_for(class_id)}")
    else:
        print("All expected classes contain images.")

    if completeness["unexpected_numeric_folders"]:
        print()
        print("UNEXPECTED NUMERIC CLASS FOLDERS:")
        for class_id in completeness["unexpected_numeric_folders"]:
            print(f"  {class_id}")

    class_quality, records = audit_image_quality(class_files, workers)
    quality = global_quality_summary(records)
    print_quality_report(quality)
    print_class_quality_report(class_quality)

    duplicates = audit_duplicates(records)
    print_duplicate_report(duplicates)

    removal_result = remove_duplicates_interactively(duplicates, interactive=not args.no_interactive)

    imbalance = imbalance_analysis(distribution["counts"])
    readiness = effnet_readiness_report(quality, distribution, imbalance)
    print_effnet_readiness(readiness)

    health_score, score_notes = calculate_health_score(distribution, quality, duplicates)

    print_section("7. DATASET HEALTH SCORE")
    print(f"FIELDCARE DATASET HEALTH SCORE: {health_score}/100")

    if health_score >= 90:
        interpretation = "EXCELLENT"
    elif health_score >= 80:
        interpretation = "GOOD"
    elif health_score >= 70:
        interpretation = "FAIR"
    elif health_score >= 50:
        interpretation = "NEEDS IMPROVEMENT"
    else:
        interpretation = "POOR"

    print(f"Overall status: {interpretation}")

    if score_notes:
        print()
        print("Score factors:")
        for note in score_notes:
            print(f"  {note}")

    write_training_config(readiness, distribution)
    write_json_report(
        distribution, imbalance, completeness, quality, class_quality,
        duplicates, readiness, health_score, removal_result,
    )

    print()
    print("=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)
    print()
    print(f"Full JSON report saved to:\n  {REPORT_JSON}")
    print(f"Training config saved to:\n  {TRAINING_CONFIG_JSON}")

    removed_total = removal_result["exact_removed"] + removal_result["visual_removed"]
    print()
    print(f"Files removed during this run: {removed_total:,}")
    if removed_total:
        print("Run the audit again for a fresh post-removal report.")

    return 0


class TeeStream:
    """Write console output to both the terminal and a persistent log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

    @property
    def encoding(self):
        return getattr(self.streams[0], "encoding", "utf-8")


def main():
    args = parse_args()
    logs_dir = args.logs_dir.resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    console_log = logs_dir / f"{timestamp}_dataset_audit.log"
    audit_json_log = logs_dir / f"{timestamp}_dataset_audit.json"
    training_config_log = logs_dir / f"{timestamp}_effnet_b3_training_config.json"

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with console_log.open("w", encoding="utf-8") as log_stream:
        sys.stdout = TeeStream(original_stdout, log_stream)
        sys.stderr = TeeStream(original_stderr, log_stream)
        try:
            result = run_audit(args)
            if result == 0:
                shutil.copy2(REPORT_JSON, audit_json_log)
                shutil.copy2(TRAINING_CONFIG_JSON, training_config_log)
                print()
                print("Timestamped audit logs saved to:")
                print(f"  Console log    : {console_log}")
                print(f"  Audit JSON     : {audit_json_log}")
                print(f"  Training config: {training_config_log}")
            else:
                print(f"Audit failed; console output was retained at: {console_log}")
            return result
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    raise SystemExit(main())
