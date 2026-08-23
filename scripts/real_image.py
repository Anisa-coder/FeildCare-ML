from pathlib import Path
from urllib.parse import urlparse
import hashlib
import re

import requests
from PIL import Image
from io import BytesIO


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT_DIR / "dataset"
DATASET_LABEL_FILE = DATASET_DIR / "label.txt"

REAL_WORLD_DIR = ROOT_DIR / "real_world"
IMAGE_DIR = REAL_WORLD_DIR / "images"
REAL_LABEL_FILE = REAL_WORLD_DIR / "label.txt"


# ============================================================
# SETTINGS
# ============================================================

IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REAL_WORLD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMAGE_EXTENSIONS = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "BMP": ".bmp",
    "TIFF": ".tiff",
}


# ============================================================
# LOAD LABELS
# ============================================================

def load_labels():

    if not DATASET_LABEL_FILE.exists():

        raise FileNotFoundError(
            f"dataset/label.txt not found:\n"
            f"{DATASET_LABEL_FILE}"
        )

    labels = {}

    for line in DATASET_LABEL_FILE.read_text(
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

        class_id = int(
            parts[0]
        )

        labels[class_id] = parts[1]

    return dict(
        sorted(labels.items())
    )


LABELS = load_labels()


# ============================================================
# DOWNLOAD IMAGE
# ============================================================

def download_image(url):

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/151.0 Safari/537.36"
        )
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=30,
        allow_redirects=True,
    )

    response.raise_for_status()

    return response.content


# ============================================================
# CREATE SAFE FILENAME
# ============================================================

def create_filename(
    class_id,
    class_name,
    image,
    image_hash,
):

    safe_name = re.sub(
        r"[^a-zA-Z0-9]+",
        "_",
        class_name
    ).strip("_")

    extension = IMAGE_EXTENSIONS.get(
        image.format,
        ".jpg"
    )

    return (
        f"{class_id}_"
        f"{safe_name}_"
        f"{image_hash[:12]}"
        f"{extension}"
    )


# ============================================================
# CHECK DUPLICATE
# ============================================================

def is_duplicate(
    image_hash
):

    for path in IMAGE_DIR.iterdir():

        if not path.is_file():
            continue

        try:

            existing_hash = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()

            if existing_hash == image_hash:

                return path

        except Exception:

            continue

    return None


# ============================================================
# ADD LABEL
# ============================================================

def add_to_label_file(
    filename,
    class_id
):

    if REAL_LABEL_FILE.exists():

        existing_lines = (
            REAL_LABEL_FILE
            .read_text(
                encoding="utf-8"
            )
            .splitlines()
        )

    else:

        existing_lines = []

    # Don't add duplicate label entries.
    for line in existing_lines:

        parts = line.split()

        if (
            parts
            and
            parts[0] == filename
        ):

            return

    with open(
        REAL_LABEL_FILE,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            f"{filename} {class_id}\n"
        )


# ============================================================
# ADD IMAGE
# ============================================================

def add_image(
    url,
    class_id
):

    if class_id not in LABELS:

        print()
        print(
            f"ERROR: Class {class_id} "
            f"does not exist."
        )

        print()
        print(
            "Available classes:"
        )

        for cid, name in LABELS.items():

            print(
                f"  {cid}: {name}"
            )

        return False

    class_name = LABELS[
        class_id
    ]

    print()
    print(
        f"Actual class:"
    )

    print(
        f"  {class_id} - {class_name}"
    )

    print()
    print(
        "Downloading image..."
    )

    try:

        image_data = download_image(
            url
        )

    except Exception as error:

        print()
        print(
            "ERROR downloading image:"
        )

        print(error)

        return False

    # --------------------------------------------------------
    # Open / validate image
    # --------------------------------------------------------

    try:

        image = Image.open(
            BytesIO(image_data)
        )

        image.load()

    except Exception as error:

        print()
        print(
            "ERROR: URL did not contain "
            "a valid image."
        )

        print(error)

        return False

    # --------------------------------------------------------
    # Convert to RGB
    # --------------------------------------------------------

    if image.mode not in (
        "RGB",
        "RGBA"
    ):

        image = image.convert(
            "RGB"
        )

    # --------------------------------------------------------
    # Hash
    # --------------------------------------------------------

    image_hash = hashlib.sha256(
        image_data
    ).hexdigest()

    duplicate = is_duplicate(
        image_hash
    )

    if duplicate:

        print()
        print(
            "DUPLICATE IMAGE"
        )

        print(
            f"Already exists:"
        )

        print(
            duplicate
        )

        return False

    # --------------------------------------------------------
    # Filename
    # --------------------------------------------------------

    filename = create_filename(
        class_id,
        class_name,
        image,
        image_hash,
    )

    output = (
        IMAGE_DIR /
        filename
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    try:

        if image.mode == "RGBA":

            image = image.convert(
                "RGB"
            )

        image.save(
            output,
            "JPEG",
            quality=95,
            optimize=True,
        )

    except Exception as error:

        print()
        print(
            "ERROR saving image:"
        )

        print(error)

        return False

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    add_to_label_file(
        filename,
        class_id
    )

    print()
    print("=" * 70)
    print("IMAGE ADDED")
    print("=" * 70)

    print()
    print(
        f"File:"
    )

    print(
        f"  {output}"
    )

    print()
    print(
        f"Actual label:"
    )

    print(
        f"  {class_id} - {class_name}"
    )

    print()
    print(
        f"Updated:"
    )

    print(
        f"  {REAL_LABEL_FILE}"
    )

    print()

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FIELDCARE REAL-WORLD IMAGE COLLECTOR")
    print("=" * 70)

    print()
    print(
        "Paste the DIRECT image URL."
    )

    print(
        "Example:"
    )

    print(
        "https://example.com/tomato_leaf.jpg"
    )

    print()

    # --------------------------------------------------------
    # URL
    # --------------------------------------------------------

    url = input(
        "Image URL: "
    ).strip()

    if not url:

        print(
            "No URL provided."
        )

        return

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    print()

    print(
        "Available classes:"
    )

    print()

    for class_id, class_name in (
        LABELS.items()
    ):

        print(
            f"{class_id:>3}  "
            f"{class_name}"
        )

    print()

    class_input = input(
        "Actual class number: "
    ).strip()

    try:

        class_id = int(
            class_input
        )

    except ValueError:

        print(
            "Invalid class number."
        )

        return

    if class_id not in LABELS:

        print(
            f"Class {class_id} "
            f"does not exist."
        )

        return

    print()

    print(
        "You selected:"
    )

    print(
        f"{class_id} - "
        f"{LABELS[class_id]}"
    )

    print()

    confirm = input(
        "Is this the ACTUAL label? [y/N]: "
    ).strip().lower()

    if confirm != "y":

        print(
            "Cancelled."
        )

        return

    add_image(
        url,
        class_id
    )


if __name__ == "__main__":

    main()