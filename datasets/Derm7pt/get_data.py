#!/usr/bin/env python3
"""Download and validate the complete Derm7pt release from Kaggle."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


DATASET_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = DATASET_DIR / "raw"
KAGGLE_DATASET_PAGE = "https://www.kaggle.com/datasets/menakamohanakumar/derm7pt"
DOWNLOAD_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "menakamohanakumar/derm7pt"
)
UPSTREAM_DATASET_PAGE = "https://derm.cs.sfu.ca/Download.html"
EXPECTED_CASES = 1011
EXPECTED_SPLIT_SIZES = {
    "train_indexes.csv": 413,
    "valid_indexes.csv": 203,
    "test_indexes.csv": 395,
}
REQUIRED_PATHS = (
    "images",
    "meta/meta.csv",
    "meta/train_indexes.csv",
    "meta/valid_indexes.csv",
    "meta/test_indexes.csv",
)


class DatasetError(RuntimeError):
    """Raised when the downloaded dataset does not satisfy the contract."""


def _dataset_root_candidates(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for meta_csv in root.rglob("meta/meta.csv"):
        candidate = meta_csv.parent.parent
        if all((candidate / relative).exists() for relative in REQUIRED_PATHS):
            candidates.append(candidate)
    return candidates


def _read_indexes(path: Path, n_cases: int) -> list[int]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "indexes" not in reader.fieldnames:
            raise DatasetError(f"Missing 'indexes' column: {path}")
        try:
            values = [int(row["indexes"]) for row in reader]
        except (TypeError, ValueError) as exc:
            raise DatasetError(f"Non-integer split index in {path}") from exc
    if len(values) != len(set(values)):
        raise DatasetError(f"Duplicate split index in {path}")
    if any(value < 0 or value >= n_cases for value in values):
        raise DatasetError(f"Out-of-range split index in {path}")
    return values


def validate_dataset(root: Path) -> dict[str, int]:
    """Validate release_v0 metadata, official splits, and referenced images."""

    missing = [relative for relative in REQUIRED_PATHS if not (root / relative).exists()]
    if missing:
        raise DatasetError(f"Missing required paths in {root}: {', '.join(missing)}")

    meta_path = root / "meta" / "meta.csv"
    with meta_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"clinic", "derm", "diagnosis"}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise DatasetError(
                f"{meta_path} must contain columns: {sorted(required_columns)}"
            )
        rows = list(reader)

    if len(rows) != EXPECTED_CASES:
        raise DatasetError(
            f"Expected {EXPECTED_CASES} cases, found {len(rows)} in {meta_path}"
        )

    missing_images: list[str] = []
    referenced_images: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        for column in ("clinic", "derm"):
            relative = (row.get(column) or "").strip()
            if not relative:
                raise DatasetError(f"Empty {column!r} image path at metadata row {row_number}")
            image_path = root / "images" / relative
            referenced_images.add(relative)
            if not image_path.is_file():
                missing_images.append(relative)
    if missing_images:
        preview = ", ".join(missing_images[:5])
        suffix = " ..." if len(missing_images) > 5 else ""
        raise DatasetError(
            f"Missing {len(missing_images)} metadata-referenced images: {preview}{suffix}"
        )

    split_sets: dict[str, set[int]] = {}
    for filename, expected_size in EXPECTED_SPLIT_SIZES.items():
        values = _read_indexes(root / "meta" / filename, len(rows))
        if len(values) != expected_size:
            raise DatasetError(
                f"Expected {expected_size} rows in {filename}, found {len(values)}"
            )
        split_sets[filename] = set(values)

    split_names = list(split_sets)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = split_sets[left] & split_sets[right]
            if overlap:
                raise DatasetError(f"Official splits overlap: {left} and {right}")
    combined = set().union(*split_sets.values())
    if combined != set(range(len(rows))):
        raise DatasetError("Official splits do not cover every metadata row exactly once")

    return {
        "cases": len(rows),
        "referenced_images": len(referenced_images),
        "train": len(split_sets["train_indexes.csv"]),
        "valid": len(split_sets["valid_indexes.csv"]),
        "test": len(split_sets["test_indexes.csv"]),
    }


def _safe_extract(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if not members:
            raise DatasetError(f"Archive is empty: {archive}")
        for member in members:
            path = PurePosixPath(member.filename)
            unix_mode = member.external_attr >> 16
            if path.is_absolute() or ".." in path.parts:
                raise DatasetError(f"Unsafe archive path: {member.filename}")
            if (unix_mode & 0o170000) == 0o120000:
                raise DatasetError(f"Archive contains a symbolic link: {member.filename}")
        bundle.extractall(destination)


def download_archive(archive: Path) -> None:
    """Download once to a temporary file and install it after transfer completes."""
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(f"{archive.name}.part")
    request = urllib.request.Request(
        DOWNLOAD_URL,
        headers={
            "User-Agent": "Fusion-Anything-Derm7pt-Downloader/1.0",
            "Accept": "application/zip",
        },
    )
    print(f"Downloading the official Derm7pt archive to {archive}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            total = int(response.headers.get("Content-Length", "0"))
            received = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                received += len(chunk)
                if total:
                    print(
                        f"\rDownloaded {received / 1024**2:.1f}/{total / 1024**2:.1f} MiB",
                        end="",
                        flush=True,
                    )
            if total:
                print()
    except urllib.error.HTTPError as exc:
        temporary.unlink(missing_ok=True)
        if exc.code in (401, 403):
            raise DatasetError(
                "Kaggle rejected the anonymous download request. Open the dataset page "
                f"and confirm that it is still publicly available: {KAGGLE_DATASET_PAGE}"
            ) from exc
        raise DatasetError(f"Download failed with HTTP {exc.code}: {exc.reason}") from exc
    except (OSError, urllib.error.URLError) as exc:
        temporary.unlink(missing_ok=True)
        raise DatasetError(f"Download failed: {exc}") from exc
    temporary.replace(archive)


def install_archive(archive: Path, output_dir: Path, force: bool) -> dict[str, int]:
    """Validate and install the archive into the requested raw-data directory."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="derm7pt-extract-", dir=output_dir.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        _safe_extract(archive, temporary)
        candidates = _dataset_root_candidates(temporary)
        if len(candidates) != 1:
            raise DatasetError(
                f"Expected one Derm7pt dataset root in {archive}, found {len(candidates)}"
            )
        source = candidates[0]
        report = validate_dataset(source)
        if output_dir.exists():
            if not force:
                raise DatasetError(
                    f"Output already exists but is not a valid complete release: {output_dir}. "
                    "Inspect it or rerun with --force."
                )
            shutil.rmtree(output_dir)
        shutil.move(str(source), str(output_dir))
    return report


def build_parser() -> argparse.ArgumentParser:
    """Define the dataset download CLI and explicit data-license acknowledgement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="extracted dataset directory (default: datasets/Derm7pt/raw)",
    )
    parser.add_argument(
        "--archive", type=Path,
        help="use an existing Derm7pt ZIP instead of downloading it from Kaggle",
    )
    parser.add_argument(
        "--accept-license", action="store_true",
        help="confirm acceptance of the upstream Derm7pt usage restrictions",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="replace an existing incomplete output directory",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="validate an already extracted dataset without downloading",
    )
    return parser


def main() -> int:
    """Download or reuse an archive and install validated Derm7pt inputs."""
    args = build_parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    try:
        if args.check_only:
            report = validate_dataset(output_dir)
            print(f"Derm7pt is complete: {output_dir}")
            print(report)
            return 0

        if output_dir.exists():
            if not output_dir.is_dir():
                raise DatasetError(f"Output path exists and is not a directory: {output_dir}")
            try:
                report = validate_dataset(output_dir)
            except DatasetError:
                if not args.force:
                    raise
            else:
                print(f"Derm7pt is already complete: {output_dir}")
                print(report)
                return 0

        if not args.accept_license:
            raise DatasetError(
                "Derm7pt images are for academic research and may not be redistributed. "
                "Review the upstream terms, then rerun with --accept-license: "
                f"{UPSTREAM_DATASET_PAGE}"
            )

        if args.archive:
            archive = args.archive.expanduser().resolve()
            if not archive.is_file():
                raise DatasetError(f"Archive not found: {archive}")
        else:
            archive = output_dir.parent / ".downloads" / "kaggle-derm7pt.zip"
            if not archive.is_file():
                download_archive(archive)
            else:
                print(f"Using cached archive: {archive}")

        report = install_archive(archive, output_dir, args.force)
        print(f"Derm7pt is ready: {output_dir}")
        print(report)
        return 0
    except (DatasetError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
