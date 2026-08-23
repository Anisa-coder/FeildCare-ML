from pathlib import Path
import shutil


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
LABEL_FILE = DATASET / "label.txt"


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


# ============================================================
# READ LABELS
# ============================================================

def read_labels():

    labels = {}

    if not LABEL_FILE.exists():
        raise FileNotFoundError(
            f"label.txt not found:\n{LABEL_FILE}"
        )

    with open(
        LABEL_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

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

    return labels


# ============================================================
# COUNT IMAGES
# ============================================================

def count_images(folder):

    if not folder.exists():
        return 0

    return sum(
        1
        for path in folder.rglob("*")
        if (
            path.is_file()
            and
            path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )


# ============================================================
# SHOW CLASSES
# ============================================================

def show_classes(labels):

    print()
    print("=" * 75)
    print("CURRENT FIELDCARE CLASSES")
    print("=" * 75)

    for class_id in sorted(labels):

        folder = DATASET / str(class_id)

        count = count_images(folder)

        print(
            f"{class_id:>3} | "
            f"{labels[class_id]:<55} | "
            f"{count:>7,} images"
        )

    print("=" * 75)


# ============================================================
# MOVE FILES
# ============================================================

def move_contents(
    source,
    destination
):

    moved = 0

    if not source.exists():
        return moved

    destination.mkdir(
        parents=True,
        exist_ok=True
    )

    for item in source.iterdir():

        target = destination / item.name

        # ----------------------------------------------------
        # Handle filename collision
        # ----------------------------------------------------

        if target.exists():

            stem = item.stem
            suffix = item.suffix

            counter = 1

            while target.exists():

                target = (
                    destination /
                    f"{stem}_merged_{counter}"
                    f"{suffix}"
                )

                counter += 1

        shutil.move(
            str(item),
            str(target)
        )

        moved += 1

    return moved


# ============================================================
# MERGE
# ============================================================

def merge_classes(
    merge_from,
    merge_into
):

    labels = read_labels()

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if merge_from == merge_into:

        print(
            "\nERROR:"
            "\nYou cannot merge a class into itself."
        )

        return

    if merge_from not in labels:

        print(
            f"\nERROR:"
            f"\nClass {merge_from} does not exist."
        )

        return

    if merge_into not in labels:

        print(
            f"\nERROR:"
            f"\nClass {merge_into} does not exist."
        )

        return

    source_folder = (
        DATASET /
        str(merge_from)
    )

    destination_folder = (
        DATASET /
        str(merge_into)
    )

    if not source_folder.exists():

        print(
            f"\nERROR:"
            f"\nSource folder does not exist:"
            f"\n{source_folder}"
        )

        return

    if not destination_folder.exists():

        print(
            f"\nERROR:"
            f"\nDestination folder does not exist:"
            f"\n{destination_folder}"
        )

        return

    # --------------------------------------------------------
    # Preview
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("MERGE PREVIEW")
    print("=" * 75)

    print(
        f"\nMERGE FROM:"
        f"\n  {merge_from} - "
        f"{labels[merge_from]}"
    )

    print(
        f"\nMERGE INTO:"
        f"\n  {merge_into} - "
        f"{labels[merge_into]}"
    )

    print(
        f"\nRESULTING CLASS NAME:"
        f"\n  {labels[merge_from]}"
    )

    print()
    print(
        f"Images in class {merge_from}: "
        f"{count_images(source_folder):,}"
    )

    print(
        f"Images in class {merge_into}: "
        f"{count_images(destination_folder):,}"
    )

    print()
    print(
        "Renumbering:"
    )

    for old_id in range(
        merge_from + 1,
        max(labels) + 1
    ):

        print(
            f"  {old_id} -> {old_id - 1}"
        )

    print()

    confirmation = input(
        "Continue? [y/N]: "
    ).strip().lower()

    if confirmation != "y":

        print(
            "\nCancelled."
        )

        return

    # ========================================================
    # STEP 1
    # MERGE IMAGES
    # ========================================================

    print()
    print("=" * 75)
    print("STEP 1/3 - MERGING IMAGES")
    print("=" * 75)

    moved = move_contents(
        source_folder,
        destination_folder
    )

    print(
        f"Images moved: {moved:,}"
    )

    # Remove old source folder.
    try:

        source_folder.rmdir()

    except OSError:

        pass

    # ========================================================
    # STEP 2
    # TEMPORARILY RENAME ALL FOLDERS
    # ========================================================
    #
    # This is the important fix.
    #
    # Instead of:
    #
    # 19 -> 18
    #
    # while 18 already exists,
    #
    # we first do:
    #
    # 19 -> __temp_19
    # 18 -> __temp_18
    # ...
    #
    # Then:
    #
    # __temp_19 -> 18
    # __temp_18 -> 17
    #
    # No collisions occur.
    #

    print()
    print("=" * 75)
    print("STEP 2/3 - TEMPORARY FOLDER RENAME")
    print("=" * 75)

    highest_class = max(
        labels.keys()
    )

    temporary_folders = {}

    for old_id in range(
        merge_from + 1,
        highest_class + 1
    ):

        old_folder = (
            DATASET /
            str(old_id)
        )

        if not old_folder.exists():
            continue

        temporary_folder = (
            DATASET /
            f"__fieldcare_temp_{old_id}"
        )

        # Remove stale temporary folder if one
        # somehow exists from an interrupted run.
        if temporary_folder.exists():

            raise RuntimeError(
                f"Temporary folder already exists:"
                f"\n{temporary_folder}"
            )

        old_folder.rename(
            temporary_folder
        )

        temporary_folders[
            old_id
        ] = temporary_folder

        print(
            f"{old_id} -> "
            f"{temporary_folder.name}"
        )

    # ========================================================
    # STEP 3
    # RENAME TEMPORARY FOLDERS
    # ========================================================

    print()
    print("=" * 75)
    print("STEP 3/3 - FINAL RENumbering")
    print("=" * 75)

    for old_id, temporary_folder in (
        temporary_folders.items()
    ):

        new_id = old_id - 1

        new_folder = (
            DATASET /
            str(new_id)
        )

        if new_folder.exists():

            raise RuntimeError(
                f"Unexpected folder collision:"
                f"\n{new_folder}"
            )

        temporary_folder.rename(
            new_folder
        )

        print(
            f"{old_id} -> {new_id}"
        )

    # ========================================================
    # REBUILD LABELS
    # ========================================================

    new_labels = {}

    # --------------------------------------------------------
    # The merged destination gets the SOURCE name.
    # --------------------------------------------------------

    new_labels[
        merge_into
    ] = labels[
        merge_from
    ]

    # --------------------------------------------------------
    # Process remaining classes
    # --------------------------------------------------------

    for old_id in sorted(labels):

        if old_id == merge_from:
            continue

        if old_id == merge_into:
            continue

        if old_id > merge_from:

            new_id = old_id - 1

        else:

            new_id = old_id

        new_labels[
            new_id
        ] = labels[
            old_id
        ]

    # ========================================================
    # WRITE LABEL.TXT
    # ========================================================

    with open(
        LABEL_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for class_id in sorted(
            new_labels
        ):

            f.write(
                f"{class_id} "
                f"{new_labels[class_id]}\n"
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 75)
    print("MERGE COMPLETE")
    print("=" * 75)

    print()
    print(
        f"Class {merge_from} merged into "
        f"class {merge_into}"
    )

    print()
    print(
        f"Class {merge_into} is now:"
    )

    print(
        f"  {new_labels[merge_into]}"
    )

    print()
    print(
        f"Images moved: {moved:,}"
    )

    print()
    print(
        "Updated label.txt:"
    )

    print(
        f"  {LABEL_FILE}"
    )

    print()
    print(
        "Final classes:"
    )

    for class_id in sorted(
        new_labels
    ):

        folder = (
            DATASET /
            str(class_id)
        )

        print(
            f"{class_id:>3} | "
            f"{new_labels[class_id]:<55} | "
            f"{count_images(folder):>7,}"
        )

    print()
    print("=" * 75)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("FIELDCARE CLASS MERGER")
    print("=" * 75)

    if not DATASET.exists():

        print(
            f"\nERROR: Dataset folder not found:"
            f"\n{DATASET}"
        )

        return

    labels = read_labels()

    show_classes(
        labels
    )

    print()
    print(
        "Enter the classes you want to merge."
    )

    print(
        "Example: 10 -> 9"
    )

    print()

    try:

        merge_from = int(
            input(
                "Class to MERGE FROM: "
            ).strip()
        )

        merge_into = int(
            input(
                "Class to MERGE INTO: "
            ).strip()
        )

    except ValueError:

        print(
            "\nERROR: Please enter valid numbers."
        )

        return

    merge_classes(
        merge_from,
        merge_into
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()