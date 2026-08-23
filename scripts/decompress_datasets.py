from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


# ============================================================
# FIELDCARE DATASET DECOMPRESSOR
# ============================================================
#
# Restores:
#
# archives/dataset/*.zip
# archives/real_world/*.zip
#
# into:
#
# dataset/
# real_world/
#
# The archive contains the original paths:
#
# dataset/1/...
# dataset/2/...
# dataset/label.txt
#
# real_world/images/...
# real_world/results/...
# real_world/label.txt
#
# ============================================================


ROOT = Path(__file__).resolve().parent.parent

ARCHIVE_DIR = (
    ROOT /
    "archives"
)

DATASET_ARCHIVE_DIR = (
    ARCHIVE_DIR /
    "dataset-archive"
)

REAL_WORLD_ARCHIVE_DIR = (
    ARCHIVE_DIR /
    "real_world-archive"
)

MANIFEST_FILE = (
    ARCHIVE_DIR /
    "manifest.json"
)


# ============================================================
# SETTINGS
# ============================================================

VERIFY_HASHES = True

# Prevent accidental overwriting.
#
# If False, files inside the archive can replace existing
# files with the same name.

SKIP_EXISTING = False


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path) -> str:

    sha = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            sha.update(
                chunk
            )

    return sha.hexdigest()


def format_size(size: int) -> str:

    if size < 1024:

        return f"{size} B"

    if size < 1024 * 1024:

        return f"{size / 1024:.2f} KB"

    if size < 1024 * 1024 * 1024:

        return f"{size / (1024 * 1024):.2f} MB"

    return f"{size / (1024 * 1024 * 1024):.2f} GB"


# ============================================================
# ZIP SAFETY
# ============================================================

def safe_extract(
    archive: zipfile.ZipFile,
    destination: Path,
):

    destination = (
        destination.resolve()
    )

    for member in archive.infolist():

        member_path = (
            destination /
            member.filename
        ).resolve()

        # Protect against malicious ZIP paths such as:
        #
        # ../../something
        #

        if not str(
            member_path
        ).startswith(
            str(destination)
        ):

            raise RuntimeError(
                "Unsafe archive path detected:\n"
                f"{member.filename}"
            )

    archive.extractall(
        destination
    )


# ============================================================
# VERIFY ZIP
# ============================================================

def verify_archive(
    archive_path: Path
):

    print()
    print(
        f"Testing archive:"
    )

    print(
        f"  {archive_path.name}"
    )

    try:

        with zipfile.ZipFile(
            archive_path,
            "r"
        ) as archive:

            bad_file = (
                archive.testzip()
            )

            if bad_file:

                print(
                    "FAILED:"
                )

                print(
                    f"Corrupt file: "
                    f"{bad_file}"
                )

                return False

        print(
            "OK"
        )

        return True

    except Exception as error:

        print(
            "FAILED:"
        )

        print(
            error
        )

        return False


# ============================================================
# MANIFEST VERIFICATION
# ============================================================

def load_manifest():

    if not MANIFEST_FILE.exists():

        print()
        print(
            "WARNING:"
        )

        print(
            "manifest.json not found."
        )

        print(
            "Archive SHA256 verification "
            "will be skipped."
        )

        return {}

    try:

        with open(
            MANIFEST_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(
                file
            )

        return {
            item["file"]: item
            for item in data.get(
                "archives",
                []
            )
        }

    except Exception as error:

        print(
            "WARNING: Could not read manifest:"
        )

        print(
            error
        )

        return {}


# ============================================================
# VERIFY HASH
# ============================================================

def verify_manifest_hash(
    archive_path: Path,
    manifest
):

    if not VERIFY_HASHES:

        return True

    relative = str(
        archive_path.relative_to(
            ROOT
        )
    )

    entry = manifest.get(
        relative
    )

    if not entry:

        print(
            "No manifest entry."
        )

        return True

    expected = entry[
        "sha256"
    ]

    print(
        "Checking SHA256..."
    )

    actual = sha256_file(
        archive_path
    )

    if actual != expected:

        print()
        print(
            "ERROR: SHA256 mismatch!"
        )

        print(
            f"Expected: {expected}"
        )

        print(
            f"Actual:   {actual}"
        )

        return False

    print(
        "SHA256 OK."
    )

    return True


# ============================================================
# EXTRACT ONE ARCHIVE
# ============================================================

def extract_archive(
    archive_path: Path
):

    print()
    print("=" * 80)

    print(
        f"EXTRACTING"
    )

    print(
        archive_path.name
    )

    print("=" * 80)

    # --------------------------------------------------------
    # Verify ZIP structure.
    # --------------------------------------------------------

    if not verify_archive(
        archive_path
    ):

        raise RuntimeError(
            f"Archive failed verification:\n"
            f"{archive_path}"
        )

    # --------------------------------------------------------
    # Open ZIP.
    # --------------------------------------------------------

    with zipfile.ZipFile(
        archive_path,
        "r"
    ) as archive:

        members = (
            archive.infolist()
        )

        print()
        print(
            f"Files in archive: "
            f"{len(members):,}"
        )

        # ----------------------------------------------------
        # Determine extraction root.
        #
        # Dataset archives contain:
        #
        # dataset/...
        #
        # Real-world archives contain:
        #
        # real_world/...
        # ----------------------------------------------------

        names = [
            member.filename
            for member in members
        ]

        if any(
            name.startswith(
                "dataset/"
            )
            for name in names
        ):

            destination = ROOT

        elif any(
            name.startswith(
                "real_world/"
            )
            for name in names
        ):

            destination = ROOT

        else:

            raise RuntimeError(
                "Unknown archive structure:\n"
                f"{archive_path}"
            )

        # ----------------------------------------------------
        # Existing-file handling.
        # ----------------------------------------------------

        if SKIP_EXISTING:

            print(
                "Existing files will be skipped."
            )

            for member in members:

                if member.is_dir():

                    continue

                target = (
                    destination /
                    member.filename
                )

                if target.exists():

                    print(
                        f"SKIP: {target}"
                    )

                    continue

                target.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                with (
                    archive.open(member)
                    as source,
                    open(
                        target,
                        "wb"
                    ) as output
                ):

                    output.write(
                        source.read()
                    )

        else:

            safe_extract(
                archive,
                destination
            )

    print()
    print(
        "Extraction complete."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "FIELDCARE DATASET DECOMPRESSOR"
    )

    print("=" * 80)

    print()

    if not ARCHIVE_DIR.exists():

        print(
            "ERROR:"
        )

        print(
            f"Archive directory does not exist:"
        )

        print(
            ARCHIVE_DIR
        )

        return

    archives = sorted(
    list(
        (
            ARCHIVE_DIR /
            "dataset"
        ).glob(
            "*.zip"
        )
    )
    +
    list(
        (
            ARCHIVE_DIR /
            "real_world"
        ).glob(
            "*.zip"
        )
    )
)

    if not archives:

        print(
            "No ZIP archives found."
        )

        return

    print(
        f"Archives found: "
        f"{len(archives)}"
    )

    for archive in archives:

        print(
            f"  {archive.name:<45}"
            f"{format_size(archive.stat().st_size):>12}"
        )

    print()

    confirm = input(
        "Extract all archives? [y/N]: "
    ).strip().lower()

    if confirm != "y":

        print(
            "Cancelled."
        )

        return

    # ========================================================
    # MANIFEST
    # ========================================================

    manifest = (
        load_manifest()
    )

    # ========================================================
    # VERIFY + EXTRACT
    # ========================================================

    success = 0

    failed = 0

    for archive in archives:

        print()
        print(
            "=" * 80
        )

        print(
            f"PROCESSING: "
            f"{archive.name}"
        )

        print(
            "=" * 80
        )

        if not verify_manifest_hash(
            archive,
            manifest
        ):

            print(
                "Skipping corrupted/"
                "modified archive."
            )

            failed += 1

            continue

        try:

            extract_archive(
                archive
            )

            success += 1

        except Exception as error:

            print()
            print(
                "ERROR:"
            )

            print(
                error
            )

            failed += 1

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 80)

    print(
        "DECOMPRESSION COMPLETE"
    )

    print("=" * 80)

    print()

    print(
        f"Successful archives: "
        f"{success}"
    )

    print(
        f"Failed archives: "
        f"{failed}"
    )

    print()

    print(
        f"Dataset:"
    )

    print(
        f"  {DATASET_DIR}"
    )

    print()

    print(
        f"Real-world:"
    )

    print(
        f"  {REAL_WORLD_DIR}"
    )

    print()

    if failed == 0:

        print(
            "All archives were successfully "
            "verified and extracted."
        )

    else:

        print(
            "Some archives failed."
        )

        print(
            "Check the errors above before "
            "using the restored dataset."
        )


if __name__ == "__main__":

    main()