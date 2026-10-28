"""Shared manifest, path-safety, identifier, and output helpers."""

from __future__ import annotations

import json
import hashlib
import re
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


VALID_STATUSES = {"valid", "encrypted", "unreadable", "error"}
ASSET_FIELDS = (
    "asset_id",
    "document_id",
    "filename",
    "relative_path",
    "page_number",
    "asset_type",
    "asset_path",
    "image_format",
    "width_px",
    "height_px",
    "bbox",
    "image_xref",
    "image_index",
    "sha256",
    "extraction_method",
    "status",
    "error_message",
)
_SAFE_DOCUMENT_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


class ManifestError(Exception):
    """Raised when the inventory manifest cannot be safely processed."""


@dataclass(frozen=True)
class ManifestDocument:
    """A validated inventory row with a source path confined to Dataset/."""

    document_id: str
    filename: str
    relative_path: str
    status: str
    page_count: int | None
    source_path: Path


@dataclass
class AssetProcessingResult:
    records: list[dict[str, Any]]
    pdfs_processed: int
    file_errors: list[str]
    reused_files: int = 0


def _safe_source_path(relative_path: str, dataset_dir: Path) -> Path:
    if not relative_path or "\\" in relative_path or ":" in relative_path or "\x00" in relative_path:
        raise ManifestError("relative_path must be a normalized Dataset-relative path.")

    posix_path = PurePosixPath(relative_path)
    if (
        posix_path.is_absolute()
        or not posix_path.parts
        or any(part in {".", ".."} for part in posix_path.parts)
        or posix_path.as_posix() != relative_path
    ):
        raise ManifestError("relative_path must not escape the Dataset directory.")

    root = dataset_dir.resolve()
    source_path = root.joinpath(*posix_path.parts).resolve()
    if not source_path.is_relative_to(root):
        raise ManifestError("relative_path resolves outside the Dataset directory.")
    return source_path


def read_manifest(manifest_path: Path, dataset_dir: Path) -> list[ManifestDocument]:
    """Read and validate the inventory JSONL without scanning Dataset independently."""
    if not manifest_path.is_file():
        raise ManifestError(
            f"Inventory manifest not found: {manifest_path}. Rerun scripts/inspect_dataset.py."
        )

    documents: list[ManifestDocument] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    try:
        with manifest_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ManifestError(
                        f"Malformed inventory JSON on line {line_number}. "
                        "Rerun scripts/inspect_dataset.py."
                    ) from exc

                if not isinstance(row, dict):
                    raise ManifestError(
                        f"Inventory line {line_number} must contain a JSON object. "
                        "Rerun scripts/inspect_dataset.py."
                    )

                required = ("document_id", "filename", "relative_path", "status")
                if any(not isinstance(row.get(field), str) or not row[field] for field in required):
                    raise ManifestError(
                        f"Inventory line {line_number} has missing or invalid required fields. "
                        "Rerun scripts/inspect_dataset.py."
                    )

                document_id = row["document_id"]
                filename = row["filename"]
                relative_path = row["relative_path"]
                status = row["status"]
                if not _SAFE_DOCUMENT_ID.fullmatch(document_id):
                    raise ManifestError(
                        f"Inventory line {line_number} has an unsafe document_id. "
                        "Rerun scripts/inspect_dataset.py."
                    )
                if status not in VALID_STATUSES:
                    raise ManifestError(
                        f"Inventory line {line_number} has an unknown status. "
                        "Rerun scripts/inspect_dataset.py."
                    )
                if document_id in seen_ids or relative_path in seen_paths:
                    raise ManifestError(
                        f"Inventory line {line_number} duplicates a document ID or path. "
                        "Rerun scripts/inspect_dataset.py."
                    )

                source_path = _safe_source_path(relative_path, dataset_dir)
                if source_path.name != filename:
                    raise ManifestError(
                        f"Inventory line {line_number} filename does not match relative_path. "
                        "Rerun scripts/inspect_dataset.py."
                    )

                page_count = row.get("page_count")
                if status == "valid":
                    if isinstance(page_count, bool) or not isinstance(page_count, int) or page_count < 0:
                        raise ManifestError(
                            f"Inventory line {line_number} has an invalid page_count. "
                            "Rerun scripts/inspect_dataset.py."
                        )
                    if not source_path.is_file():
                        raise ManifestError(
                            f"Source PDF is missing for inventory line {line_number}: {relative_path}. "
                            "Rerun scripts/inspect_dataset.py."
                        )
                elif page_count is not None and (
                    isinstance(page_count, bool) or not isinstance(page_count, int) or page_count < 0
                ):
                    raise ManifestError(
                        f"Inventory line {line_number} has an invalid page_count. "
                        "Rerun scripts/inspect_dataset.py."
                    )

                seen_ids.add(document_id)
                seen_paths.add(relative_path)
                documents.append(
                    ManifestDocument(
                        document_id=document_id,
                        filename=filename,
                        relative_path=relative_path,
                        status=status,
                        page_count=page_count,
                        source_path=source_path,
                    )
                )
    except UnicodeDecodeError as exc:
        raise ManifestError(
            "Inventory manifest is not valid UTF-8. Rerun scripts/inspect_dataset.py."
        ) from exc
    except OSError as exc:
        detail = exc.strerror or type(exc).__name__
        raise ManifestError(f"Could not read inventory manifest: {detail}.") from exc

    return documents


def read_documents_from_assets_manifest(
    assets_manifest_path: Path, dataset_dir: Path
) -> list[ManifestDocument]:
    """Recover document IDs and page counts for assets-only repair if inventory is absent.

    This reads only the existing assets manifest and never scans Dataset independently.
    """
    if not assets_manifest_path.is_file():
        raise ManifestError(
            f"Neither inventory nor an assets manifest is available. Rerun "
            f"scripts/inspect_dataset.py; missing: {assets_manifest_path}."
        )

    grouped: dict[str, dict[str, Any]] = {}
    path_to_id: dict[str, str] = {}
    try:
        with assets_manifest_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ManifestError(
                        f"Malformed assets JSON on line {line_number}; cannot safely repair assets."
                    ) from exc
                if not isinstance(row, dict) or any(field not in row for field in ASSET_FIELDS):
                    raise ManifestError(
                        f"Assets line {line_number} has an invalid schema; cannot safely repair assets."
                    )

                document_id = row.get("document_id")
                filename = row.get("filename")
                relative_path = row.get("relative_path")
                page_number = row.get("page_number")
                asset_type = row.get("asset_type")
                if (
                    not isinstance(document_id, str)
                    or not _SAFE_DOCUMENT_ID.fullmatch(document_id)
                    or not isinstance(filename, str)
                    or not filename
                    or not isinstance(relative_path, str)
                    or not isinstance(page_number, int)
                    or isinstance(page_number, bool)
                    or page_number < 1
                    or asset_type not in ("embedded_image", "image_crop_fallback", "page_preview")
                    or row.get("status") not in ("ok", "error")
                ):
                    raise ManifestError(
                        f"Assets line {line_number} has invalid document/page fields."
                    )

                source_path = _safe_source_path(relative_path, dataset_dir)
                if source_path.name != filename or not source_path.is_file():
                    raise ManifestError(
                        f"Assets line {line_number} points to a missing or mismatched source PDF."
                    )

                existing_id = path_to_id.setdefault(relative_path, document_id)
                if existing_id != document_id:
                    raise ManifestError("Assets manifest assigns multiple document IDs to one source PDF.")

                item = grouped.setdefault(
                    document_id,
                    {
                        "filename": filename,
                        "relative_path": relative_path,
                        "source_path": source_path,
                        "preview_pages": set(),
                    },
                )
                if item["filename"] != filename or item["relative_path"] != relative_path:
                    raise ManifestError("Assets manifest has inconsistent source metadata for a document ID.")
                if asset_type == "page_preview":
                    item["preview_pages"].add(page_number)
    except UnicodeDecodeError as exc:
        raise ManifestError("Assets manifest is not valid UTF-8.") from exc
    except OSError as exc:
        detail = exc.strerror or type(exc).__name__
        raise ManifestError(f"Could not read assets manifest: {detail}.") from exc

    documents: list[ManifestDocument] = []
    for document_id, item in grouped.items():
        preview_pages = item["preview_pages"]
        if not preview_pages:
            raise ManifestError(
                f"Assets manifest has no page previews for {item['relative_path']}; "
                "rerun the inventory step to restore a trusted document list."
            )
        page_count = max(preview_pages)
        if preview_pages != set(range(1, page_count + 1)):
            raise ManifestError(
                f"Assets manifest has incomplete preview page numbering for {item['relative_path']}."
            )
        documents.append(
            ManifestDocument(
                document_id=document_id,
                filename=item["filename"],
                relative_path=item["relative_path"],
                status="valid",
                page_count=page_count,
                source_path=item["source_path"],
            )
        )
    return documents


def stable_asset_id(
    document_id: str, page_number: int, asset_kind: str, occurrence_index: int
) -> str:
    """Create a stable UUID for an image occurrence or rendered page."""
    value = f"digital-product-rag:asset:{document_id}:{page_number}:{asset_kind}:{occurrence_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


def make_asset_record(
    document: ManifestDocument,
    page_number: int,
    asset_kind: str,
    occurrence_index: int,
    *,
    asset_type: str,
    asset_path: str | None = None,
    image_format: str | None = None,
    width_px: int | None = None,
    height_px: int | None = None,
    bbox: list[float] | None = None,
    image_xref: int | None = None,
    image_index: int | None = None,
    sha256: str | None = None,
    extraction_method: str,
    status: str,
    error_message: str | None = None,
) -> dict[str, Any]:
    return {
        "asset_id": stable_asset_id(
            document.document_id, page_number, asset_kind, occurrence_index
        ),
        "document_id": document.document_id,
        "filename": document.filename,
        "relative_path": document.relative_path,
        "page_number": page_number,
        "asset_type": asset_type,
        "asset_path": asset_path,
        "image_format": image_format,
        "width_px": width_px,
        "height_px": height_px,
        "bbox": bbox,
        "image_xref": image_xref,
        "image_index": image_index,
        "sha256": sha256,
        "extraction_method": extraction_method,
        "status": status,
        "error_message": error_message,
    }


def atomic_write_bytes(output_path: Path, contents: bytes) -> None:
    """Write bytes to a temporary sibling and atomically replace the target."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(contents)
        temporary_path.replace(output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass


def inspect_raster_file(path: Path, pymupdf: Any) -> tuple[int, int, str]:
    """Verify a saved image can be decoded and return its dimensions and file hash."""
    if not path.is_file():
        raise FileNotFoundError("Generated image file is missing")
    if path.stat().st_size == 0:
        raise ValueError("Generated image file is empty")

    pixmap = pymupdf.Pixmap(str(path))
    width = int(pixmap.width)
    height = int(pixmap.height)
    if width <= 0 or height <= 0:
        raise ValueError("Generated image has invalid dimensions")

    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return width, height, digest.hexdigest()


def write_jsonl(records: Iterable[dict[str, Any]], output_path: Path) -> None:
    """Atomically serialize records as UTF-8 JSON Lines."""
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
                if "asset_id" in record:
                    serializable_record = {field: record[field] for field in ASSET_FIELDS}
                else:
                    serializable_record = record
                temporary_file.write(json.dumps(serializable_record, ensure_ascii=False))
                temporary_file.write("\n")
        temporary_path.replace(output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass
