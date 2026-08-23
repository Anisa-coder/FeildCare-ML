from __future__ import annotations


from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
import hashlib
import json
import math
import os
import sys

try:
    from PIL import Image, ImageStat
except ImportError:
    print("ERROR: Pillow is required.")
    print("Install with: pip install pillow")
    raise


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"

REPORT_DIR = ROOT / "_dataset_reports"
REPORT_JSON = REPORT_DIR / "dataset_audit.json"


# ============================================================================
# CONFIGURATION
# ============================================================================

EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

# label.txt is the single source of truth for class IDs and names.
# Expected format:
#   1 Healthy Corn
#   2 Corn Eyespot
#   3 Corn Northern Leaf Blight


LABEL_FILE = DATASET / "label.txt"


def load_labels() -> dict[int, str]:
    """Load class IDs and names directly from dataset/label.txt."""
    if not LABEL_FILE.exists():
        raise FileNotFoundError(
            f"label.txt not found: {LABEL_FILE}"
        )

    labels: dict[int, str] = {}

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
                "Expected: <number> <class name>"
            )

        class_id = int(parts[0])
        class_name = parts[1].strip()

        if class_id <= 0:
            raise ValueError(
                f"Invalid class ID {class_id} on line {line_number}."
            )

        if class_id in labels:
            raise ValueError(
                f"Duplicate class ID {class_id} in label.txt."
            )

        labels[class_id] = class_name

    if not labels:
        raise ValueError(f"No classes found in {LABEL_FILE}")

    return dict(sorted(labels.items()))


LABELS = load_labels()


# Class targets are intentionally not hard-coded.\n# Class definitions come exclusively from label.txt.\n

# Quality thresholds.
#
# These are warnings, NOT automatic deletions.
MIN_WIDTH = 128
MIN_HEIGHT = 128

# Images below this area are flagged as small.
MIN_PIXELS = 128 * 128

# Aspect ratios outside this range are flagged.
MIN_ASPECT_RATIO = 0.35
MAX_ASPECT_RATIO = 2.85

# Extremely large files may indicate uncompressed/raw data.
LARGE_FILE_MB = 10

# Very tiny files can be broken/placeholder images.
TINY_FILE_KB = 5

# Number of examples shown per issue.
MAX_EXAMPLES = 15

# Show every class even if it does not currently exist.
SHOW_EMPTY_CLASSES = True


# ============================================================================
# HELPERS
# ============================================================================

def label_for(class_id: int) -> str:
    return LABELS.get(
        class_id,
        f"Unknown Class {class_id}",
    )


def pct(value: float, total: float) -> str:
    if total <= 0:
        return "0.00%"
    return f"{(value / total) * 100:.2f}%"


def ratio_text(a: int, b: int) -> str:
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
    return (
        path.is_file()
        and path.suffix.lower() in EXTENSIONS
    )


def iter_class_images():
    if not DATASET.exists():
        return

    for folder in DATASET.iterdir():

        if not folder.is_dir():
            continue

        if not folder.name.isdigit():
            continue

        class_id = int(folder.name)

        for image in folder.rglob("*"):

            if is_image(image):
                yield class_id, image


def sha256_file(path: Path) -> str | None:
    try:

        digest = hashlib.sha256()

        with path.open("rb") as file:

            while True:

                chunk = file.read(1024 * 1024)

                if not chunk:
                    break

                digest.update(chunk)

        return digest.hexdigest()

    except Exception:
        return None


def human_size(size_bytes: int) -> str:

    units = (
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    )

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

    return (
        ordered[lower]
        + (ordered[upper] - ordered[lower]) * fraction
    )


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


def inspect_image(
    path: Path,
) -> dict:

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
        "error": None,
        "small": False,
        "extreme_aspect": False,
        "tiny_file": False,
        "large_file": False,
        "grayscale": False,
        "has_alpha": False,
    }

    try:

        result["file_size"] = path.stat().st_size

        result["tiny_file"] = (
            result["file_size"]
            < TINY_FILE_KB * 1024
        )

        result["large_file"] = (
            result["file_size"]
            > LARGE_FILE_MB * 1024 * 1024
        )

        result["sha256"] = sha256_file(path)

        with Image.open(path) as image:

            # verify() checks the encoded file without decoding all pixels.
            image.verify()

        # Reopen after verify().
        with Image.open(path) as image:

            width, height = image.size

            result["width"] = width
            result["height"] = height
            result["pixels"] = width * height
            result["aspect_ratio"] = (
                width / height
                if height
                else None
            )
            result["format"] = (
                image.format or path.suffix.lstrip(".")
            ).lower()
            result["mode"] = image.mode

            channels = {
                "1": 1,
                "L": 1,
                "LA": 2,
                "P": 1,
                "RGB": 3,
                "RGBA": 4,
                "CMYK": 4,
                "YCbCr": 3,
                "I": 1,
                "F": 1,
            }.get(image.mode)

            result["channels"] = channels

            result["grayscale"] = image.mode in {
                "1",
                "L",
                "LA",
            }

            result["has_alpha"] = (
                "A" in image.mode
                or image.mode == "P"
                and "transparency" in image.info
            )

        result["small"] = (
            result["width"] < MIN_WIDTH
            or result["height"] < MIN_HEIGHT
            or result["pixels"] < MIN_PIXELS
        )

        ratio = result["aspect_ratio"]

        result["extreme_aspect"] = (
            ratio is not None
            and (
                ratio < MIN_ASPECT_RATIO
                or ratio > MAX_ASPECT_RATIO
            )
        )

        result["valid"] = True

    except Exception as error:

        result["error"] = str(error)

    return result


# ============================================================================
# DISTRIBUTION
# ============================================================================

def audit_distribution(
    class_files: dict,
) -> dict:

    counts = {}

    for class_id in LABELS:

        counts[class_id] = len(
            class_files.get(class_id, [])
        )

    # Include unexpected numeric class folders.
    for class_id in class_files:

        if class_id not in counts:
            counts[class_id] = len(
                class_files[class_id]
            )

    present = [
        count
        for count in counts.values()
        if count > 0
    ]

    total = sum(counts.values())

    minimum = min(present) if present else 0
    maximum = max(present) if present else 0

    return {
        "counts": counts,
        "total": total,
        "minimum": minimum,
        "maximum": maximum,
        "ratio": (
            maximum / minimum
            if minimum
            else None
        ),
    }


# ============================================================================
# IMAGE QUALITY
# ============================================================================

def audit_image_quality(
    class_files: dict,
) -> tuple[dict, list[dict]]:

    stats_by_class = {}
    records = []

    all_paths = []

    for class_id, images in class_files.items():

        for path in images:
            all_paths.append(
                (class_id, path)
            )

    print_subsection(
        "IMAGE VALIDATION / QUALITY SCAN"
    )

    print(
        f"Images to inspect: {len(all_paths):,}"
    )

    for class_id, path in all_paths:

        record = inspect_image(path)
        record["class_id"] = class_id

        records.append(record)

    for class_id in class_files:

        class_records = [
            record
            for record in records
            if record["class_id"] == class_id
        ]

        valid = [
            record
            for record in class_records
            if record["valid"]
        ]

        widths = [
            record["width"]
            for record in valid
            if record["width"] is not None
        ]

        heights = [
            record["height"]
            for record in valid
            if record["height"] is not None
        ]

        pixels = [
            record["pixels"]
            for record in valid
            if record["pixels"] is not None
        ]

        file_sizes = [
            record["file_size"]
            for record in valid
        ]

        stats_by_class[class_id] = {
            "total": len(class_records),
            "valid": len(valid),
            "corrupt": sum(
                not record["valid"]
                for record in class_records
            ),
            "small": sum(
                record["small"]
                for record in valid
            ),
            "extreme_aspect": sum(
                record["extreme_aspect"]
                for record in valid
            ),
            "grayscale": sum(
                record["grayscale"]
                for record in valid
            ),
            "alpha": sum(
                record["has_alpha"]
                for record in valid
            ),
            "tiny_files": sum(
                record["tiny_file"]
                for record in valid
            ),
            "large_files": sum(
                record["large_file"]
                for record in valid
            ),
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
            "formats": dict(
                Counter(
                    record["format"]
                    for record in valid
                )
            ),
            "modes": dict(
                Counter(
                    record["mode"]
                    for record in valid
                )
            ),
        }

    return stats_by_class, records


# ============================================================================
# DUPLICATES
# ============================================================================

def audit_duplicates(
    records: list[dict],
) -> dict:

    print_subsection(
        "EXACT DUPLICATE ANALYSIS"
    )

    hash_map = defaultdict(list)

    for record in records:

        digest = record.get("sha256")

        if digest:
            hash_map[digest].append(record)

    duplicate_groups = {
        digest: items
        for digest, items in hash_map.items()
        if len(items) > 1
    }

    duplicate_files = sum(
        len(items) - 1
        for items in duplicate_groups.values()
    )

    cross_class_groups = []
    same_class_groups = []

    for digest, items in duplicate_groups.items():

        class_ids = {
            item["class_id"]
            for item in items
        }

        if len(class_ids) > 1:
            cross_class_groups.append(
                (digest, items)
            )
        else:
            same_class_groups.append(
                (digest, items)
            )

    print(
        f"Unique hashes          : {len(hash_map):,}"
    )

    print(
        f"Duplicate groups       : "
        f"{len(duplicate_groups):,}"
    )

    print(
        f"Duplicate files        : "
        f"{duplicate_files:,}"
    )

    print(
        f"Cross-class groups     : "
        f"{len(cross_class_groups):,}"
    )

    print(
        f"Within-class groups    : "
        f"{len(same_class_groups):,}"
    )

    if cross_class_groups:

        print()
        print(
            "WARNING: Cross-class exact duplicates detected."
        )

        print(
            "These are especially important because the same "
            "image exists under different labels."
        )

    return {
        "unique_hashes": len(hash_map),
        "duplicate_groups": len(duplicate_groups),
        "duplicate_files": duplicate_files,
        "cross_class_groups": len(cross_class_groups),
        "within_class_groups": len(same_class_groups),
        "groups": duplicate_groups,
        "cross_class_examples": [
            {
                "sha256": digest,
                "files": [
                    {
                        "class_id": item["class_id"],
                        "path": item["relative_path"],
                    }
                    for item in items
                ],
            }
            for digest, items in cross_class_groups[:MAX_EXAMPLES]
        ],
    }


# ============================================================================
# FORMAT / QUALITY GLOBAL ANALYSIS
# ============================================================================

def global_quality_summary(
    records: list[dict],
) -> dict:

    valid = [
        record
        for record in records
        if record["valid"]
    ]

    corrupt = [
        record
        for record in records
        if not record["valid"]
    ]

    dimensions = [
        (
            record["width"],
            record["height"],
        )
        for record in valid
        if record["width"] and record["height"]
    ]

    ratios = [
        record["aspect_ratio"]
        for record in valid
        if record["aspect_ratio"]
    ]

    sizes = [
        record["file_size"]
        for record in valid
    ]

    return {
        "total_records": len(records),
        "valid": len(valid),
        "corrupt": len(corrupt),
        "small": sum(
            record["small"]
            for record in valid
        ),
        "extreme_aspect": sum(
            record["extreme_aspect"]
            for record in valid
        ),
        "grayscale": sum(
            record["grayscale"]
            for record in valid
        ),
        "alpha": sum(
            record["has_alpha"]
            for record in valid
        ),
        "tiny_files": sum(
            record["tiny_file"]
            for record in valid
        ),
        "large_files": sum(
            record["large_file"]
            for record in valid
        ),
        "formats": dict(
            Counter(
                record["format"]
                for record in valid
            )
        ),
        "modes": dict(
            Counter(
                record["mode"]
                for record in valid
            )
        ),
        "dimensions": {
            "min_width": min(
                (x[0] for x in dimensions),
                default=0,
            ),
            "median_width": safe_median(
                [x[0] for x in dimensions]
            ),
            "min_height": min(
                (x[1] for x in dimensions),
                default=0,
            ),
            "median_height": safe_median(
                [x[1] for x in dimensions]
            ),
            "min_pixels": min(
                (
                    record["pixels"]
                    for record in valid
                    if record["pixels"]
                ),
                default=0,
            ),
            "median_pixels": safe_median(
                [
                    record["pixels"]
                    for record in valid
                    if record["pixels"]
                ]
            ),
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
        "corrupt_examples": [
            record["relative_path"]
            for record in corrupt[:MAX_EXAMPLES]
        ],
        "small_examples": [
            record["relative_path"]
            for record in valid
            if record["small"]
        ][:MAX_EXAMPLES],
        "aspect_examples": [
            record["relative_path"]
            for record in valid
            if record["extreme_aspect"]
        ][:MAX_EXAMPLES],
        "tiny_file_examples": [
            record["relative_path"]
            for record in valid
            if record["tiny_file"]
        ][:MAX_EXAMPLES],
    }


# ============================================================================
# IMBALANCE ANALYSIS
# ============================================================================

def imbalance_analysis(
    counts: dict,
) -> dict:

    nonzero = {
        class_id: count
        for class_id, count in counts.items()
        if count > 0
    }

    if not nonzero:

        return {
            "median": 0,
            "mean": 0,
            "max": 0,
            "min": 0,
            "max_to_min": None,
            "classes_below_25pct_median": [],
            "classes_below_50pct_median": [],
            "classes_above_2x_median": [],
        }

    values = list(nonzero.values())

    med = median(values)

    below_25 = [
        class_id
        for class_id, count in nonzero.items()
        if count < med * 0.25
    ]

    below_50 = [
        class_id
        for class_id, count in nonzero.items()
        if count < med * 0.50
    ]

    above_2x = [
        class_id
        for class_id, count in nonzero.items()
        if count > med * 2
    ]

    return {
        "median": med,
        "mean": mean(values),
        "max": max(values),
        "min": min(values),
        "max_to_min": (
            max(values) / min(values)
            if min(values)
            else None
        ),
        "classes_below_25pct_median": below_25,
        "classes_below_50pct_median": below_50,
        "classes_above_2x_median": above_2x,
    }


# ============================================================================
# DATASET COMPLETENESS
# ============================================================================

def completeness_analysis(
    counts: dict,
) -> dict:

    expected = set(LABELS)
    actual = set(counts)

    missing_folders = sorted(
        class_id
        for class_id in expected
        if counts.get(class_id, 0) == 0
    )

    unexpected_folders = sorted(
        class_id
        for class_id in actual
        if class_id not in expected
    )

    return {
        "expected_classes": len(expected),
        "classes_with_images": sum(
            counts.get(class_id, 0) > 0
            for class_id in expected
        ),
        "empty_classes": missing_folders,
        "unexpected_numeric_folders": unexpected_folders,
    }


# ============================================================================
# HEALTH SCORE
# ============================================================================

def calculate_health_score(
    distribution: dict,
    quality: dict,
    duplicates: dict,
) -> tuple[int, list[str]]:

    score = 100
    deductions = []

    # ------------------------------------------------------------
    # Corrupt images
    # ------------------------------------------------------------

    total = max(
        1,
        quality["total_records"],
    )

    corrupt_rate = (
        quality["corrupt"]
        / total
    )

    if corrupt_rate > 0:
        deduction = min(
            15,
            math.ceil(corrupt_rate * 100),
        )

        score -= deduction

        deductions.append(
            f"-{deduction} corrupt-image penalty"
        )

    # ------------------------------------------------------------
    # Duplicate images
    # ------------------------------------------------------------

    duplicate_rate = (
        duplicates["duplicate_files"]
        / total
    )

    if duplicate_rate > 0:
        deduction = min(
            15,
            math.ceil(duplicate_rate * 50),
        )

        score -= deduction

        deductions.append(
            f"-{deduction} duplicate penalty"
        )

    # Cross-class duplicates are more serious.
    if duplicates["cross_class_groups"] > 0:

        score -= min(
            15,
            duplicates["cross_class_groups"] * 3,
        )

        deductions.append(
            "-cross-class duplicate penalty"
        )

    # ------------------------------------------------------------
    # Small images
    # ------------------------------------------------------------

    small_rate = (
        quality["small"]
        / total
    )

    if small_rate > 0.10:

        score -= 8
        deductions.append(
            "-8 low-resolution penalty"
        )

    elif small_rate > 0.03:

        score -= 4
        deductions.append(
            "-4 low-resolution penalty"
        )

    # ------------------------------------------------------------
    # Extreme aspect ratio
    # ------------------------------------------------------------

    aspect_rate = (
        quality["extreme_aspect"]
        / total
    )

    if aspect_rate > 0.10:

        score -= 5
        deductions.append(
            "-5 aspect-ratio penalty"
        )

    # ------------------------------------------------------------
    # Class imbalance
    # ------------------------------------------------------------

    imbalance = imbalance_analysis(
        distribution["counts"]
    )

    if imbalance["max_to_min"] is not None:

        if imbalance["max_to_min"] > 20:

            score -= 10
            deductions.append(
                "-10 severe class-imbalance penalty"
            )

        elif imbalance["max_to_min"] > 10:

            score -= 6
            deductions.append(
                "-6 class-imbalance penalty"
            )

        elif imbalance["max_to_min"] > 5:

            score -= 3
            deductions.append(
                "-3 class-imbalance penalty"
            )

    return max(0, min(100, score)), deductions


# ============================================================================
# REPORT OUTPUT
# ============================================================================

def print_distribution_report(
    distribution: dict,
) -> None:

    print_section(
        "1. CLASS DISTRIBUTION"
    )

    print(
        f"{'ID':>3}  "
        f"{'Class':<60} "
        f"{'Images':>10}"
    )

    print("-" * 85)

    for class_id in sorted(LABELS):

        count = distribution["counts"].get(
            class_id,
            0,
        )

        status = "EMPTY" if count == 0 else ""

        print(
            f"{class_id:>3}  "
            f"{label_for(class_id):<60} "
            f"{count:>10,} "
            f"{status}"
        )

    unexpected = [
        class_id
        for class_id in distribution["counts"]
        if class_id not in LABELS
    ]

    if unexpected:

        print()
        print("NUMERIC FOLDERS NOT DEFINED IN label.txt:")

        for class_id in sorted(unexpected):

            print(
                f"  {class_id}: "
                f"{distribution['counts'][class_id]:,} images"
            )

    print("-" * 85)

    print(
        f"TOTAL IMAGES: "
        f"{distribution['total']:,}"
    )


def print_imbalance_report(
    distribution: dict,
) -> None:

    analysis = imbalance_analysis(
        distribution["counts"]
    )

    print_section(
        "2. CLASS IMBALANCE ANALYSIS"
    )

    print(
        f"Smallest non-empty class : "
        f"{analysis['min']:,}"
    )

    print(
        f"Largest class            : "
        f"{analysis['max']:,}"
    )

    print(
        f"Median class size        : "
        f"{analysis['median']:,.0f}"
    )

    print(
        f"Mean class size          : "
        f"{analysis['mean']:,.0f}"
    )

    if analysis["max_to_min"] is not None:

        print(
            f"Max / Min ratio          : "
            f"{analysis['max_to_min']:.2f}:1"
        )

    if analysis["classes_below_50pct_median"]:

        print()
        print(
            "Classes below 50% of median:"
        )

        for class_id in (
            analysis[
                "classes_below_50pct_median"
            ]
        ):

            print(
                f"  {class_id}: "
                f"{label_for(class_id)} "
                f"({distribution['counts'][class_id]:,})"
            )

    if analysis["classes_above_2x_median"]:

        print()
        print(
            "Classes above 2x median:"
        )

        for class_id in (
            analysis[
                "classes_above_2x_median"
            ]
        ):

            print(
                f"  {class_id}: "
                f"{label_for(class_id)} "
                f"({distribution['counts'][class_id]:,})"
            )


def print_quality_report(
    quality: dict,
) -> None:

    print_section(
        "3. IMAGE QUALITY / INTEGRITY"
    )

    total = max(
        1,
        quality["total_records"],
    )

    print(
        f"Valid images          : "
        f"{quality['valid']:,} "
        f"({pct(quality['valid'], total)})"
    )

    print(
        f"Corrupt/unreadable     : "
        f"{quality['corrupt']:,} "
        f"({pct(quality['corrupt'], total)})"
    )

    print(
        f"Small / low-res        : "
        f"{quality['small']:,} "
        f"({pct(quality['small'], total)})"
    )

    print(
        f"Extreme aspect ratio   : "
        f"{quality['extreme_aspect']:,} "
        f"({pct(quality['extreme_aspect'], total)})"
    )

    print(
        f"Grayscale              : "
        f"{quality['grayscale']:,} "
        f"({pct(quality['grayscale'], total)})"
    )

    print(
        f"Images with alpha      : "
        f"{quality['alpha']:,} "
        f"({pct(quality['alpha'], total)})"
    )

    print(
        f"Tiny files             : "
        f"{quality['tiny_files']:,}"
    )

    print(
        f"Large files > "
        f"{LARGE_FILE_MB} MB      : "
        f"{quality['large_files']:,}"
    )

    print()
    print(
        "Dimensions:"
    )

    print(
        f"  Minimum width        : "
        f"{quality['dimensions']['min_width']}"
    )

    print(
        f"  Median width         : "
        f"{quality['dimensions']['median_width']:,.0f}"
    )

    print(
        f"  Minimum height       : "
        f"{quality['dimensions']['min_height']}"
    )

    print(
        f"  Median height        : "
        f"{quality['dimensions']['median_height']:,.0f}"
    )

    print(
        f"  Median pixels        : "
        f"{quality['dimensions']['median_pixels']:,.0f}"
    )

    print()
    print(
        "Aspect ratio:"
    )

    print(
        f"  Minimum              : "
        f"{quality['aspect_ratio']['minimum']:.3f}"
    )

    print(
        f"  Median               : "
        f"{quality['aspect_ratio']['median']:.3f}"
    )

    print(
        f"  Maximum              : "
        f"{quality['aspect_ratio']['maximum']:.3f}"
    )

    print()
    print(
        "Formats:"
    )

    for fmt, count in sorted(
        quality["formats"].items(),
        key=lambda item: item[1],
        reverse=True,
    ):

        print(
            f"  {fmt:<10} "
            f"{count:>8,} "
            f"({pct(count, quality['valid'])})"
        )

    print()
    print(
        "Image modes:"
    )

    for mode, count in sorted(
        quality["modes"].items(),
        key=lambda item: item[1],
        reverse=True,
    ):

        print(
            f"  {str(mode):<10} "
            f"{count:>8,} "
            f"({pct(count, quality['valid'])})"
        )


def print_class_quality_report(
    stats_by_class: dict,
) -> None:

    print_section(
        "4. PER-CLASS QUALITY PROFILE"
    )

    print(
        f"{'ID':>3} "
        f"{'Class':<45} "
        f"{'Valid':>8} "
        f"{'Bad':>6} "
        f"{'Small':>7} "
        f"{'Gray':>7} "
        f"{'Aspect':>8}"
    )

    print("-" * 95)

    for class_id in sorted(LABELS):

        stats = stats_by_class.get(
            class_id,
            {
                "total": 0,
                "valid": 0,
                "corrupt": 0,
                "small": 0,
                "grayscale": 0,
                "extreme_aspect": 0,
            },
        )

        print(
            f"{class_id:>3} "
            f"{label_for(class_id):<45} "
            f"{stats['valid']:>8,} "
            f"{stats['corrupt']:>6,} "
            f"{stats['small']:>7,} "
            f"{stats['grayscale']:>7,} "
            f"{stats['extreme_aspect']:>8,}"
        )


def print_duplicate_report(
    duplicates: dict,
) -> None:

    print_section(
        "5. DUPLICATE / DATA LEAKAGE ANALYSIS"
    )

    print(
        f"Unique SHA-256 hashes    : "
        f"{duplicates['unique_hashes']:,}"
    )

    print(
        f"Duplicate groups         : "
        f"{duplicates['duplicate_groups']:,}"
    )

    print(
        f"Duplicate files          : "
        f"{duplicates['duplicate_files']:,}"
    )

    print(
        f"Within-class duplicates  : "
        f"{duplicates['within_class_groups']:,}"
    )

    print(
        f"Cross-class duplicates   : "
        f"{duplicates['cross_class_groups']:,}"
    )

    if duplicates["cross_class_groups"]:

        print()
        print(
            "!!! CRITICAL: CROSS-CLASS DUPLICATES FOUND !!!"
        )

        print(
            "The same exact image appears under different labels."
        )

        print(
            "These should be resolved before model training."
        )

        for example in duplicates[
            "cross_class_examples"
        ]:

            print()
            print(
                f"Hash: {example['sha256'][:16]}..."
            )

            for item in example["files"]:

                print(
                    f"  Class {item['class_id']} "
                    f"({label_for(item['class_id'])}): "
                    f"{item['path']}"
                )




# ============================================================================
# DUPLICATE REVIEW / OPTIONAL REMOVAL
# ============================================================================

def remove_duplicates_interactively(records: list[dict], duplicates: dict) -> dict:
    """Show exact duplicate groups with full paths and ask before deleting."""
    groups = duplicates.get("groups", {})
    result = {"within_class_removed": 0, "cross_class_removed": 0, "removed_paths": []}
    if not groups:
        return result

    within_groups = []
    cross_groups = []
    for digest, items in groups.items():
        class_ids = {item["class_id"] for item in items}
        (within_groups if len(class_ids) == 1 else cross_groups).append((digest, items))

    print_section("DUPLICATE FILE DETAILS")

    print()
    print(f"WITHIN-CLASS EXACT DUPLICATES: {len(within_groups):,} group(s)")
    print("Same image content appears multiple times inside the same class.")
    for number, (digest, items) in enumerate(within_groups, 1):
        print(f"\n[Within-class group {number}]\nSHA-256: {digest}")
        for index, item in enumerate(items):
            status = "KEEP" if index == 0 else "DUPLICATE"
            print(f"  [{status}] Class {item['class_id']} - {label_for(item['class_id'])}")
            print(f"         {item['path']}")

    print()
    print(f"CROSS-CLASS EXACT DUPLICATES: {len(cross_groups):,} group(s)")
    print("The exact same image appears under different class labels.")
    for number, (digest, items) in enumerate(cross_groups, 1):
        print(f"\n[Cross-class group {number}]\nSHA-256: {digest}")
        for index, item in enumerate(items):
            status = "FIRST" if index == 0 else "DUPLICATE"
            print(f"  [{status}] Class {item['class_id']} - {label_for(item['class_id'])}")
            print(f"         {item['path']}")

    if within_groups:
        print()
        answer = input("Remove WITHIN-CLASS exact duplicates, keeping the first file in each group? [y/N]: ").strip().lower()
        if answer == "y":
            for digest, items in within_groups:
                for item in items[1:]:
                    file_path = Path(item["path"])
                    try:
                        if file_path.exists():
                            file_path.unlink()
                            result["within_class_removed"] += 1
                            result["removed_paths"].append(str(file_path))
                            print(f"Removed: {file_path}")
                    except Exception as error:
                        print(f"Could not remove {file_path}: {error}")
        else:
            print("Within-class duplicates were NOT removed.")

    if cross_groups:
        print()
        print("WARNING: Cross-class duplicates may represent the same image assigned different labels.")
        answer = input("Remove CROSS-CLASS exact duplicates, keeping the first file in each group? [y/N]: ").strip().lower()
        if answer == "y":
            for digest, items in cross_groups:
                for item in items[1:]:
                    file_path = Path(item["path"])
                    try:
                        if file_path.exists():
                            file_path.unlink()
                            result["cross_class_removed"] += 1
                            result["removed_paths"].append(str(file_path))
                            print(f"Removed: {file_path}")
                    except Exception as error:
                        print(f"Could not remove {file_path}: {error}")
        else:
            print("Cross-class duplicates were NOT removed.")

    print()
    print("Duplicate review complete.")
    print(f"Within-class removed : {result['within_class_removed']:,}")
    print(f"Cross-class removed  : {result['cross_class_removed']:,}")
    return result


# ============================================================================
# JSON REPORT
# ============================================================================

def write_json_report(
    distribution: dict,
    imbalance: dict,
    completeness: dict,
    quality: dict,
    class_quality: dict,
    duplicates: dict,
    health_score: int,
    recommendations: list[str],
    removal_result: dict | None = None,
) -> None:

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Duplicate groups contain Path objects indirectly only in records,
    # but our stored cross-class examples are already serializable.
    duplicate_json = {
        key: value
        for key, value in duplicates.items()
        if key != "groups"
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
        "duplicates": duplicate_json,
        "health_score": health_score,
        "recommendations": [],
        "duplicate_removal": removal_result or {"within_class_removed": 0, "cross_class_removed": 0, "removed_paths": []},
        "configuration": {
            "min_width": MIN_WIDTH,
            "min_height": MIN_HEIGHT,
            "min_pixels": MIN_PIXELS,
            "min_aspect_ratio": MIN_ASPECT_RATIO,
            "max_aspect_ratio": MAX_ASPECT_RATIO,
            "large_file_mb": LARGE_FILE_MB,
            "tiny_file_kb": TINY_FILE_KB,
        },
    }

    REPORT_JSON.write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 80)
    print("                 FIELDCARE DATASET AUDIT v2")
    print("=" * 80)

    print()
    print(f"Dataset root: {DATASET}")
    print(f"Label source: {LABEL_FILE}")
    print(f"Classes loaded from label.txt: {len(LABELS)}")

    if not DATASET.exists():

        print()
        print(
            "ERROR: Dataset directory does not exist."
        )

        return 1

    # ------------------------------------------------------------
    # Collect
    # ------------------------------------------------------------

    print()
    print("Scanning class folders...")

    class_files = collect_files()

    # ------------------------------------------------------------
    # Distribution
    # ------------------------------------------------------------

    distribution = audit_distribution(
        class_files
    )

    print_distribution_report(
        distribution,
    )

    # ------------------------------------------------------------
    # Imbalance
    # ------------------------------------------------------------

    print_imbalance_report(
        distribution
    )

    # ------------------------------------------------------------
    # Completeness
    # ------------------------------------------------------------

    completeness = completeness_analysis(
        distribution["counts"]
    )

    print_section(
        "DATASET COMPLETENESS"
    )

    print(
        f"Expected classes       : "
        f"{completeness['expected_classes']}"
    )

    print(
        f"Classes with images     : "
        f"{completeness['classes_with_images']}"
    )

    if completeness["empty_classes"]:

        print()
        print(
            "EMPTY / MISSING CLASSES:"
        )

        for class_id in completeness[
            "empty_classes"
        ]:

            print(
                f"  {class_id}: "
                f"{label_for(class_id)}"
            )

    else:

        print(
            "All expected classes contain images."
        )

    if completeness[
        "unexpected_numeric_folders"
    ]:

        print()
        print(
            "UNEXPECTED NUMERIC CLASS FOLDERS:"
        )

        for class_id in completeness[
            "unexpected_numeric_folders"
        ]:

            print(
                f"  {class_id}"
            )

    # ------------------------------------------------------------
    # Image quality
    # ------------------------------------------------------------

    class_quality, records = audit_image_quality(
        class_files
    )

    quality = global_quality_summary(
        records
    )

    print_quality_report(
        quality
    )

    print_class_quality_report(
        class_quality
    )

    # ------------------------------------------------------------
    # Duplicate analysis
    # ------------------------------------------------------------

    duplicates = audit_duplicates(
        records
    )

    # ------------------------------------------------------------
    # Duplicate review / optional removal
    # ------------------------------------------------------------

    removal_result = remove_duplicates_interactively(
        records,
        duplicates,
    )

    # ------------------------------------------------------------
    # Health score
    # ------------------------------------------------------------

    health_score, score_notes = calculate_health_score(
        distribution,
        quality,
        duplicates,
    )

    print_section(
        "6. DATASET HEALTH SCORE"
    )

    print(
        f"FIELDCARE DATASET HEALTH SCORE: "
        f"{health_score}/100"
    )

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

    print(
        f"Overall status: {interpretation}"
    )

    if score_notes:

        print()
        print(
            "Score factors:"
        )

        for note in score_notes:
            print(
                f"  {note}"
            )

    # ------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------

    imbalance = imbalance_analysis(
        distribution["counts"]
    )

    write_json_report(
        distribution,
        imbalance,
        completeness,
        quality,
        class_quality,
        duplicates,
        health_score,
        [],
        removal_result,
    )

    print()
    print("=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)

    print()
    print(
        f"JSON report saved to:\n"
        f"  {REPORT_JSON}"
    )

    removed_total = (
        removal_result["within_class_removed"]
        + removal_result["cross_class_removed"]
    )

    print()
    print(f"Files removed during this run: {removed_total:,}")
    if removed_total:
        print("Run the audit again for a fresh post-removal report.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())