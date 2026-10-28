"""Create a deterministic inventory of PDFs in the local Dataset directory."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import sys
import tempfile
import unicodedata
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


MANIFEST_FIELDS = (
    "document_id",
    "filename",
    "relative_path",
    "file_size_bytes",
    "page_count",
    "sha256",
    "status",
    "product_title",
    "error_message",
)


class DatasetDirectoryError(Exception):
    """Raised when the requested dataset directory does not exist."""


class PyMuPDFUnavailableError(Exception):
    """Raised when PDF inspection is requested without PyMuPDF installed."""


def discover_pdfs(dataset_dir: Path) -> list[Path]:
    """Return PDF files recursively in deterministic relative-path order."""
    if not dataset_dir.is_dir():
        raise DatasetDirectoryError(
            f"Dataset directory not found: {dataset_dir}. Create it or pass --dataset PATH."
        )

    try:
        pdfs = [
            path
            for path in dataset_dir.rglob("*")
            if path.is_file() and path.suffix.lower() == ".pdf"
        ]
    except OSError as exc:
        detail = exc.strerror or type(exc).__name__
        raise DatasetDirectoryError(f"Could not scan the dataset directory: {detail}.") from exc

    return sorted(pdfs, key=lambda path: path.relative_to(dataset_dir).as_posix())


def document_id_for(relative_path: str) -> str:
    """Create a stable ID from a normalized, dataset-relative path."""
    normalized_path = unicodedata.normalize("NFC", relative_path.replace("\\", "/"))
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"multimodal-rag:{normalized_path}"))


def product_title_for(filename: str) -> str:
    """Turn a filename stem into a readable title without changing its case."""
    stem = Path(filename).stem
    return re.sub(r"\s+", " ", re.sub(r"[_-]", " ", stem)).strip()


def sha256_file(path: Path) -> str:
    """Hash a file incrementally so large PDFs are not loaded into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pymupdf() -> Any:
    """Import PyMuPDF only when there is at least one PDF to inspect."""
    try:
        return importlib.import_module("pymupdf")
    except ImportError as exc:
        raise PyMuPDFUnavailableError(
            "PyMuPDF is not installed for this Python interpreter. Install it with: "
            "python -m pip install pymupdf"
        ) from exc


def _filesystem_error(action: str, exc: OSError) -> str:
    detail = exc.strerror or type(exc).__name__
    return f"{action}: {' '.join(detail.split())[:120]}"


def inspect_pdf(path: Path, dataset_dir: Path, pymupdf: Any) -> dict[str, Any]:
    """Collect file metadata and inspect one PDF without stopping the scan."""
    relative_path = path.relative_to(dataset_dir).as_posix()
    file_size_bytes: int | None = None
    sha256: str | None = None
    page_count: int | None = None
    status = "error"
    errors: list[str] = []

    try:
        file_size_bytes = path.stat().st_size
    except OSError as exc:
        errors.append(_filesystem_error("Could not read file size", exc))

    try:
        sha256 = sha256_file(path)
    except OSError as exc:
        errors.append(_filesystem_error("Could not calculate SHA-256", exc))

    document = None
    try:
        document = pymupdf.open(path)
        if document is None or not getattr(document, "is_pdf", True):
            status = "unreadable"
        elif document.needs_pass:
            status = "encrypted"
        else:
            page_count = int(document.page_count)
            status = "valid"
    except OSError as exc:
        errors.append(_filesystem_error("Could not inspect PDF", exc))
    except RuntimeError as exc:
        # PyMuPDF reports malformed documents as RuntimeError-derived exceptions.
        # Its missing-file exception is also RuntimeError-derived, but is a filesystem error.
        if type(exc).__name__ == "FileNotFoundError":
            errors.append("Could not inspect PDF: file not found")
        else:
            status = "unreadable"
    except Exception as exc:  # Keep unexpected per-file processing failures isolated.
        errors.append(f"Unexpected PDF processing error ({type(exc).__name__})")
    finally:
        if document is not None:
            try:
                document.close()
            except Exception as exc:  # Closing failures also make the inspection incomplete.
                errors.append(f"Could not close PDF document ({type(exc).__name__})")

    if errors:
        status = "error"
        error_message = "; ".join(errors)
    elif status == "encrypted":
        error_message = "PDF requires a password; page count is unavailable."
    elif status == "unreadable":
        error_message = "PDF is malformed, unsupported, or unreadable."
    else:
        error_message = None

    return {
        "document_id": document_id_for(relative_path),
        "filename": path.name,
        "relative_path": relative_path,
        "file_size_bytes": file_size_bytes,
        "page_count": page_count,
        "sha256": sha256,
        "status": status,
        "product_title": product_title_for(path.name),
        "error_message": error_message,
    }


def find_duplicate_groups(records: Sequence[dict[str, Any]]) -> list[list[str]]:
    """Return groups of relative paths with the same successfully computed hash."""
    paths_by_hash: dict[str, list[str]] = defaultdict(list)
    for record in records:
        digest = record["sha256"]
        if digest:
            paths_by_hash[digest].append(record["relative_path"])

    groups = [sorted(paths) for paths in paths_by_hash.values() if len(paths) > 1]
    return sorted(groups, key=lambda paths: paths[0])


def write_manifest(records: Sequence[dict[str, Any]], output_path: Path) -> None:
    """Atomically write JSON Lines, replacing the destination after a full write."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            for record in records:
                ordered_record = {field: record[field] for field in MANIFEST_FIELDS}
                temporary_file.write(json.dumps(ordered_record, ensure_ascii=False))
                temporary_file.write("\n")

        temporary_path.replace(output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass


def print_summary(records: Sequence[dict[str, Any]], output_path: Path) -> None:
    """Print inventory counts and duplicate groups for the completed manifest."""
    counts = {status: sum(record["status"] == status for record in records)
              for status in ("valid", "encrypted", "unreadable", "error")}
    total_pages = sum(record["page_count"] or 0 for record in records)
    duplicate_groups = find_duplicate_groups(records)

    print("PDF inventory summary")
    print(f"  PDFs discovered: {len(records)}")
    print(f"  Valid: {counts['valid']}")
    print(f"  Encrypted: {counts['encrypted']}")
    print(f"  Unreadable: {counts['unreadable']}")
    print(f"  Errors: {counts['error']}")
    print(f"  Total readable pages: {total_pages}")
    print(f"  Duplicate groups: {len(duplicate_groups)}")
    for group in duplicate_groups:
        print("    - " + " | ".join(group))
    if not records:
        print("  Warning: no PDF files were found in the dataset directory.")
    print(f"  Manifest: {output_path}")


def generate_manifest(
    dataset_dir: Path,
    output_path: Path,
    *,
    pymupdf: Any | None = None,
) -> list[dict[str, Any]]:
    """Scan a dataset, write its manifest, and return the records."""
    dataset_dir = dataset_dir.resolve()
    output_path = output_path.resolve()
    pdf_paths = discover_pdfs(dataset_dir)

    if pdf_paths and pymupdf is None:
        pymupdf = load_pymupdf()

    records = [inspect_pdf(path, dataset_dir, pymupdf) for path in pdf_paths]
    write_manifest(records, output_path)
    print_summary(records, output_path)
    return records


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recursively inventory PDFs in Dataset/ into a JSONL manifest."
    )
    parser.add_argument("--dataset", type=Path, help="Dataset directory (default: <repo>/Dataset)")
    parser.add_argument(
        "--output",
        type=Path,
        help="Manifest path (default: <repo>/data/processed/manifest.jsonl)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    repository_root = Path(__file__).resolve().parents[1]
    dataset_dir = (args.dataset or repository_root / "Dataset").expanduser().resolve()
    output_path = (
        args.output or repository_root / "data" / "processed" / "manifest.jsonl"
    ).expanduser().resolve()

    try:
        generate_manifest(dataset_dir, output_path)
    except DatasetDirectoryError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except PyMuPDFUnavailableError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except OSError as exc:
        detail = exc.strerror or type(exc).__name__
        print(f"Could not write manifest: {detail}.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
