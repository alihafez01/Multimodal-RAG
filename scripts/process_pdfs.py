"""Run page-wise text extraction, image extraction, and page rendering."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path, PurePosixPath
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from digital_product_rag.ingestion.common import (  # noqa: E402
    ManifestDocument,
    ManifestError,
    inspect_raster_file,
    read_documents_from_assets_manifest,
    read_manifest,
    write_jsonl,
)
from digital_product_rag.ingestion.image_extractor import extract_images  # noqa: E402
from digital_product_rag.ingestion.page_renderer import (  # noqa: E402
    DEFAULT_DPI,
    validate_dpi,
    render_pages,
)
from digital_product_rag.ingestion.text_extractor import extract_pages  # noqa: E402


def _dpi_argument(value: str) -> int:
    try:
        dpi = int(value)
        return validate_dpi(dpi)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _asset_storage_size(records: list[dict[str, Any]], repository_root: Path) -> int:
    unique_paths: set[Path] = set()
    total_bytes = 0
    root = repository_root.resolve()
    for record in records:
        if record.get("status") != "ok":
            continue
        relative_path = record.get("asset_path")
        if not relative_path:
            continue
        path = (root / Path(*PurePosixPath(relative_path).parts)).resolve()
        if not path.is_relative_to(root) or path in unique_paths or not path.is_file():
            continue
        unique_paths.add(path)
        try:
            total_bytes += path.stat().st_size
        except OSError:
            continue
    return total_bytes


def _ordered_asset_records(
    image_records: list[dict[str, Any]],
    preview_records: list[dict[str, Any]],
    documents: list[ManifestDocument],
) -> list[dict[str, Any]]:
    manifest_order = {document.document_id: index for index, document in enumerate(documents)}
    records = image_records + preview_records
    return sorted(
        records,
        key=lambda record: (
            manifest_order.get(record["document_id"], len(manifest_order)),
            record["page_number"],
            1 if record["asset_type"] == "page_preview" else 0,
            record["image_index"] or 0,
            record["asset_id"],
        ),
    )


def _asset_file_path(record: dict[str, Any], repository_root: Path) -> Path | None:
    relative_path = record.get("asset_path")
    if not isinstance(relative_path, str) or not relative_path or "\\" in relative_path:
        return None
    asset_path = PurePosixPath(relative_path)
    if (
        asset_path.is_absolute()
        or any(part in {".", ".."} for part in asset_path.parts)
        or asset_path.parts[:2] != ("data", "assets")
    ):
        return None
    root = repository_root.resolve()
    assets_root = (root / "data" / "assets").resolve()
    candidate = (root / Path(*asset_path.parts)).resolve()
    return candidate if candidate.is_relative_to(assets_root) else None


def _asset_record_is_valid(
    record: dict[str, Any], repository_root: Path, pymupdf: Any
) -> bool:
    path = _asset_file_path(record, repository_root)
    if path is None:
        return False
    try:
        width, height, digest = inspect_raster_file(path, pymupdf)
        if (
            digest != record.get("sha256")
            or width != record.get("width_px")
            or height != record.get("height_px")
            or path.suffix.lower() != f".{str(record.get('image_format', '')).lower()}"
        ):
            return False
        if record.get("asset_type") == "page_preview":
            with path.open("rb") as preview_file:
                if preview_file.read(8) != b"\x89PNG\r\n\x1a\n":
                    return False
        return True
    except (OSError, RuntimeError, ValueError, TypeError):
        return False


def _audit_asset_records(
    records: list[dict[str, Any]], repository_root: Path, pymupdf: Any
) -> list[dict[str, Any]]:
    """Find records marked successful whose files fail a path, hash, or image check."""
    return [
        record
        for record in records
        if record.get("status") == "ok"
        and not _asset_record_is_valid(record, repository_root, pymupdf)
    ]


def _replace_document_records(
    records: list[dict[str, Any]], document_ids: set[str], replacements: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [record for record in records if record["document_id"] not in document_ids] + replacements


def _repair_and_audit_assets(
    documents: list[ManifestDocument],
    image_records: list[dict[str, Any]],
    preview_records: list[dict[str, Any]],
    repository_root: Path,
    assets_root: Path,
    *,
    dpi: int,
    pymupdf: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int, int]:
    """Regenerate stale outputs by document, then downgrade any remaining invalid files."""
    combined = _ordered_asset_records(image_records, preview_records, documents)
    initially_invalid = _audit_asset_records(combined, repository_root, pymupdf)
    image_doc_ids = {
        record["document_id"]
        for record in initially_invalid
        if record.get("asset_type") != "page_preview"
    }
    preview_doc_ids = {
        record["document_id"]
        for record in initially_invalid
        if record.get("asset_type") == "page_preview"
    }
    repair_count = len(initially_invalid)
    doc_by_id = {document.document_id: document for document in documents}

    if image_doc_ids:
        selected = [doc_by_id[document_id] for document_id in image_doc_ids]
        repaired = extract_images(
            selected,
            repository_root,
            assets_root,
            dpi=dpi,
            pymupdf=pymupdf,
        )
        image_records = _replace_document_records(image_records, image_doc_ids, repaired.records)

    if preview_doc_ids:
        selected = [doc_by_id[document_id] for document_id in preview_doc_ids]
        repaired = render_pages(
            selected,
            repository_root,
            assets_root,
            dpi=dpi,
            pymupdf=pymupdf,
        )
        preview_records = _replace_document_records(
            preview_records, preview_doc_ids, repaired.records
        )

    combined = _ordered_asset_records(image_records, preview_records, documents)
    still_invalid = _audit_asset_records(combined, repository_root, pymupdf)
    for record in still_invalid:
        record["status"] = "error"
        record["asset_path"] = None
        record["sha256"] = None
        record["error_message"] = "Generated asset is missing or failed integrity validation after repair."

    return image_records, preview_records, repair_count, len(still_invalid)


def _process_asset_stages(
    documents: list[ManifestDocument],
    repository_root: Path,
    *,
    dpi: int,
    pymupdf: Any,
) -> dict[str, Any]:
    assets_root = repository_root / "data" / "assets"
    processed_dir = repository_root / "data" / "processed"
    image_result = extract_images(
        documents,
        repository_root,
        assets_root,
        dpi=dpi,
        pymupdf=pymupdf,
    )
    preview_result = render_pages(
        documents,
        repository_root,
        assets_root,
        dpi=dpi,
        pymupdf=pymupdf,
    )
    (
        image_result.records,
        preview_result.records,
        repair_attempts,
        remaining_invalid,
    ) = _repair_and_audit_assets(
        documents,
        image_result.records,
        preview_result.records,
        repository_root,
        assets_root,
        dpi=dpi,
        pymupdf=pymupdf,
    )
    asset_records = _ordered_asset_records(
        image_result.records,
        preview_result.records,
        documents,
    )
    assets_manifest_path = processed_dir / "assets.jsonl"
    write_jsonl(asset_records, assets_manifest_path)
    return {
        "image_result": image_result,
        "preview_result": preview_result,
        "asset_records": asset_records,
        "assets_manifest_path": assets_manifest_path,
        "repair_attempts": repair_attempts,
        "remaining_invalid": remaining_invalid,
        "storage_bytes": _asset_storage_size(asset_records, repository_root),
    }


def run_pipeline(
    repository_root: Path = REPOSITORY_ROOT,
    *,
    dpi: int = DEFAULT_DPI,
    pymupdf: Any | None = None,
) -> dict[str, Any]:
    """Run the full PDF ingestion stages from the existing inventory manifest."""
    validate_dpi(dpi)
    repository_root = repository_root.resolve()
    processed_dir = repository_root / "data" / "processed"
    assets_root = repository_root / "data" / "assets"
    documents = read_manifest(processed_dir / "manifest.jsonl", repository_root / "Dataset")

    if pymupdf is None:
        try:
            pymupdf = importlib.import_module("pymupdf")
        except ImportError as exc:
            raise RuntimeError(
                "PyMuPDF is unavailable for this Python interpreter. Install it globally with: "
                "python -m pip install pymupdf"
            ) from exc

    text_result = extract_pages(
        documents,
        processed_dir / "pages.jsonl",
        pymupdf,
    )
    asset_result = _process_asset_stages(
        documents,
        repository_root,
        dpi=dpi,
        pymupdf=pymupdf,
    )
    image_result = asset_result["image_result"]
    preview_result = asset_result["preview_result"]
    asset_records = asset_result["asset_records"]
    assets_manifest_path = asset_result["assets_manifest_path"]

    page_successes = sum(record["status"] == "ok" for record in text_result.records)
    page_failures = len(text_result.records) - page_successes
    ocr_pages = sum(bool(record["ocr_required"]) for record in text_result.records)
    detected_occurrences = sum(
        record["image_index"] is not None for record in image_result.records
    )
    extracted_images = sum(
        record["status"] == "ok" and record["asset_type"] == "embedded_image"
        for record in image_result.records
    )
    fallback_crops = sum(
        record["status"] == "ok" and record["asset_type"] == "image_crop_fallback"
        for record in image_result.records
    )
    image_failures = sum(record["status"] == "error" for record in image_result.records)
    rendered_pages = sum(
        record["status"] == "ok" and record["asset_type"] == "page_preview"
        for record in preview_result.records
    )
    rendering_failures = sum(
        record["status"] == "error" and record["asset_type"] == "page_preview"
        for record in preview_result.records
    )
    storage_bytes = asset_result["storage_bytes"]

    skipped = text_result.skipped_by_status
    print("PDF processing summary")
    print("  Text extraction")
    print(f"    PDFs processed: {text_result.pdfs_processed}")
    print(f"    PDFs skipped by inventory status: {sum(skipped.values())}")
    if skipped:
        print("    Skipped status counts: " + ", ".join(f"{key}={value}" for key, value in sorted(skipped.items())))
    print(f"    Manifest expected pages: {text_result.expected_page_count}")
    print(f"    Page records produced: {len(text_result.records)}")
    print(f"    Pages extracted successfully: {page_successes}")
    print(f"    Page extraction errors: {page_failures}")
    print(f"    Pages flagged for possible OCR: {ocr_pages}")
    print(f"    Output: {processed_dir / 'pages.jsonl'}")
    print("  Image extraction and page rendering")
    print(f"    PDFs processed for images: {image_result.pdfs_processed}")
    print(f"    PDFs processed for previews: {preview_result.pdfs_processed}")
    print(f"    Embedded image occurrences detected: {detected_occurrences}")
    print(f"    Embedded images extracted: {extracted_images}")
    print(f"    Fallback image crops created: {fallback_crops}")
    print(f"    Image extraction failures: {image_failures}")
    print(f"    Existing image files reused: {image_result.reused_files}")
    print(f"    Page previews rendered: {rendered_pages}")
    print(f"    Page rendering failures: {rendering_failures}")
    print(f"    Existing page previews reused: {preview_result.reused_files}")
    print(f"    Final integrity repair attempts: {asset_result['repair_attempts']}")
    print(f"    Assets still invalid after repair: {asset_result['remaining_invalid']}")
    print(f"    Asset records written: {len(asset_records)}")
    print(f"    Generated image and preview storage: {storage_bytes:,} bytes")
    print(f"    Output: {assets_manifest_path}")

    for label, errors in (
        ("Text file errors", text_result.file_errors),
        ("Image extraction errors", image_result.file_errors),
        ("Rendering errors", preview_result.file_errors),
    ):
        if errors:
            print(f"  {label}: {len(errors)}")
            for message in errors[:10]:
                print(f"    - {message}")
            if len(errors) > 10:
                print(f"    - ... and {len(errors) - 10} more")

    return {
        "documents": documents,
        "text_records": text_result.records,
        "image_records": image_result.records,
        "preview_records": preview_result.records,
        "asset_records": asset_records,
        "text_file_errors": text_result.file_errors,
        "image_file_errors": image_result.file_errors,
        "render_file_errors": preview_result.file_errors,
        "storage_bytes": storage_bytes,
    }


def run_asset_pipeline(
    repository_root: Path = REPOSITORY_ROOT,
    *,
    dpi: int = DEFAULT_DPI,
    pymupdf: Any | None = None,
) -> dict[str, Any]:
    """Repair image and preview outputs without rewriting pages.jsonl."""
    validate_dpi(dpi)
    repository_root = repository_root.resolve()
    processed_dir = repository_root / "data" / "processed"
    inventory_path = processed_dir / "manifest.jsonl"
    recovery_source = False
    if inventory_path.is_file():
        documents = read_manifest(inventory_path, repository_root / "Dataset")
    else:
        documents = read_documents_from_assets_manifest(
            processed_dir / "assets.jsonl", repository_root / "Dataset"
        )
        recovery_source = True
    if pymupdf is None:
        try:
            pymupdf = importlib.import_module("pymupdf")
        except ImportError as exc:
            raise RuntimeError(
                "PyMuPDF is unavailable for this Python interpreter. Install it globally with: "
                "python -m pip install pymupdf"
            ) from exc

    result = _process_asset_stages(
        documents,
        repository_root,
        dpi=dpi,
        pymupdf=pymupdf,
    )
    image_result = result["image_result"]
    preview_result = result["preview_result"]
    records = result["asset_records"]
    print("PDF asset repair summary")
    if recovery_source:
        print(
            "  Inventory manifest is absent; used existing assets.jsonl source references "
            "for this assets-only repair. pages.jsonl was not touched."
        )
    print(f"  PDFs processed for images: {image_result.pdfs_processed}")
    print(f"  Embedded image occurrences: {sum(r['image_index'] is not None for r in image_result.records)}")
    print(f"  Embedded images extracted: {sum(r['status'] == 'ok' and r['asset_type'] == 'embedded_image' for r in image_result.records)}")
    print(f"  Fallback crops created: {sum(r['status'] == 'ok' and r['asset_type'] == 'image_crop_fallback' for r in image_result.records)}")
    print(f"  Image failures: {sum(r['status'] == 'error' for r in image_result.records)}")
    print(f"  Existing image files reused: {image_result.reused_files}")
    print(f"  Page previews valid after processing: {sum(r['status'] == 'ok' and r['asset_type'] == 'page_preview' for r in preview_result.records)}")
    print(f"  Existing page previews reused: {preview_result.reused_files}")
    print(f"  Integrity repair attempts: {result['repair_attempts']}")
    print(f"  Assets still invalid after repair: {result['remaining_invalid']}")
    print(f"  Asset records written: {len(records)}")
    print(f"  Generated storage: {result['storage_bytes']:,} bytes")
    print(f"  Output: {result['assets_manifest_path']}")
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract PDF page text, embedded images, and page previews."
    )
    parser.add_argument(
        "--dpi",
        type=_dpi_argument,
        default=DEFAULT_DPI,
        help=f"Preview and fallback-crop resolution ({72}-{300}; default: {DEFAULT_DPI}).",
    )
    parser.add_argument(
        "--assets-only",
        action="store_true",
        help="Repair images and page previews without rewriting pages.jsonl.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        if args.assets_only:
            run_asset_pipeline(REPOSITORY_ROOT, dpi=args.dpi)
        else:
            run_pipeline(REPOSITORY_ROOT, dpi=args.dpi)
    except (ManifestError, OSError, RuntimeError, ValueError) as exc:
        print(f"PDF processing could not continue: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
