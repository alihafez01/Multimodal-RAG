"""Extract displayed raster image occurrences from valid PDFs."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any

from .common import (
    AssetProcessingResult,
    ManifestDocument,
    atomic_write_bytes,
    inspect_raster_file,
    make_asset_record,
)
from .page_renderer import DEFAULT_DPI, validate_dpi


def _normalise_bbox(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        bbox = [float(coordinate) for coordinate in value]
    except (TypeError, ValueError):
        return None
    if len(bbox) != 4 or not all(math.isfinite(coordinate) for coordinate in bbox):
        return None
    return bbox


def _positive_int(value: Any) -> int | None:
    try:
        integer = int(value)
    except (TypeError, ValueError):
        return None
    return integer if integer > 0 else None


def _safe_extension(value: Any) -> str:
    extension = re.sub(r"[^a-z0-9]", "", str(value).lower())[:8]
    return extension or "bin"


def _source_image_bytes(pdf: Any, xref: int, pymupdf: Any) -> tuple[bytes, str, int | None, int | None]:
    extracted = pdf.extract_image(xref)
    image_bytes = extracted.get("image")
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise ValueError("PyMuPDF returned no image data")

    extension = _safe_extension(extracted.get("ext", "bin"))
    width = _positive_int(extracted.get("width"))
    height = _positive_int(extracted.get("height"))
    soft_mask = _positive_int(extracted.get("smask"))
    if soft_mask is not None:
        base_pixmap = pymupdf.Pixmap(pdf, xref)
        mask_pixmap = pymupdf.Pixmap(pdf, soft_mask)
        combined_pixmap = pymupdf.Pixmap(base_pixmap, mask_pixmap)
        image_bytes = combined_pixmap.tobytes("png")
        extension = "png"
        width = int(combined_pixmap.width)
        height = int(combined_pixmap.height)

    return image_bytes, extension, width, height


def _persist_verified_image(
    output_path: Path,
    image_bytes: bytes,
    pymupdf: Any,
) -> tuple[int, int, str, bool]:
    """Reuse an intact matching image or atomically write and verify a replacement."""
    expected_hash = hashlib.sha256(image_bytes).hexdigest()
    try:
        width, height, actual_hash = inspect_raster_file(output_path, pymupdf)
        if actual_hash == expected_hash:
            return width, height, actual_hash, True
    except Exception:
        pass

    atomic_write_bytes(output_path, image_bytes)
    width, height, actual_hash = inspect_raster_file(output_path, pymupdf)
    if actual_hash != expected_hash:
        raise OSError("Saved image content does not match the extracted image")
    return width, height, actual_hash, False


def _crop_image_bytes(
    page: Any,
    bbox: list[float] | None,
    dpi: int,
    pymupdf: Any,
) -> tuple[bytes, int, int]:
    if bbox is None:
        raise ValueError("Image occurrence has no usable bounding box")
    clip = pymupdf.Rect(bbox) & page.rect
    if not clip.is_valid or clip.is_empty or clip.width <= 0 or clip.height <= 0:
        raise ValueError("Image bounding box does not intersect the page")
    scale = dpi / 72.0
    pixmap = page.get_pixmap(
        matrix=pymupdf.Matrix(scale, scale),
        colorspace=pymupdf.csRGB,
        alpha=False,
        clip=clip,
    )
    return pixmap.tobytes("png"), int(pixmap.width), int(pixmap.height)


def extract_images(
    documents: list[ManifestDocument],
    repository_root: Path,
    assets_root: Path,
    *,
    dpi: int = DEFAULT_DPI,
    pymupdf: Any,
) -> AssetProcessingResult:
    """Extract every displayed image occurrence, using page crops as fallback."""
    validate_dpi(dpi)
    repository_root = repository_root.resolve()
    assets_root = assets_root.resolve()
    records: list[dict[str, Any]] = []
    file_errors: list[str] = []
    pdfs_processed = 0
    reused_paths: set[Path] = set()

    for document in documents:
        if document.status != "valid":
            continue

        pdf = None
        try:
            pdf = pymupdf.open(document.source_path)
            if pdf is None or not getattr(pdf, "is_pdf", True) or pdf.needs_pass:
                raise RuntimeError("PDF is not readable without a password")
            page_count = int(pdf.page_count)
            pdfs_processed += 1
        except Exception as exc:
            file_errors.append(
                f"{document.relative_path}: could not open valid inventory PDF "
                f"({type(exc).__name__})."
            )
            if pdf is not None:
                try:
                    pdf.close()
                except Exception:
                    pass
            continue

        saved_by_hash: dict[str, tuple[Path, str]] = {}
        try:
            for zero_based_page in range(page_count):
                page_number = zero_based_page + 1
                try:
                    page = pdf.load_page(zero_based_page)
                    image_infos = page.get_image_info(xrefs=True)
                    if not isinstance(image_infos, list):
                        raise TypeError("image occurrence metadata is not a list")
                except Exception as exc:
                    message = f"Could not inspect page images ({type(exc).__name__})."
                    file_errors.append(f"{document.relative_path}, page {page_number}: {message}")
                    records.append(
                        make_asset_record(
                            document,
                            page_number,
                            "image_occurrence",
                            0,
                            asset_type="embedded_image",
                            extraction_method="source_image",
                            status="error",
                            error_message=message,
                        )
                    )
                    continue

                for occurrence_index, info in enumerate(image_infos, start=1):
                    xref = _positive_int(info.get("xref")) if isinstance(info, dict) else None
                    bbox = _normalise_bbox(info.get("bbox")) if isinstance(info, dict) else None
                    image_type = "embedded_image" if xref is not None else "image_crop_fallback"
                    method = "source_image" if xref is not None else "page_crop_fallback"
                    image_bytes: bytes | None = None
                    extension: str | None = None
                    width = _positive_int(info.get("width")) if isinstance(info, dict) else None
                    height = _positive_int(info.get("height")) if isinstance(info, dict) else None
                    error_message: str | None = None

                    if xref is not None:
                        try:
                            image_bytes, extension, source_width, source_height = _source_image_bytes(
                                pdf, xref, pymupdf
                            )
                            width = source_width or width
                            height = source_height or height
                        except Exception:
                            image_bytes = None

                    if image_bytes is None:
                        image_type = "image_crop_fallback"
                        method = "page_crop_fallback"
                        try:
                            image_bytes, width, height = _crop_image_bytes(
                                page, bbox, dpi, pymupdf
                            )
                            extension = "png"
                        except Exception as exc:
                            error_message = (
                                "Image extraction and page-crop fallback failed "
                                f"({type(exc).__name__})."
                            )

                    if image_bytes is None or extension is None:
                        file_errors.append(
                            f"{document.relative_path}, page {page_number}, "
                            f"image {occurrence_index}: {error_message}"
                        )
                        records.append(
                            make_asset_record(
                                document,
                                page_number,
                                "image_occurrence",
                                occurrence_index,
                                asset_type=image_type,
                                bbox=bbox,
                                image_xref=xref,
                                image_index=occurrence_index,
                                extraction_method=method,
                                status="error",
                                error_message=error_message,
                            )
                        )
                        continue

                    digest = hashlib.sha256(image_bytes).hexdigest()
                    cached = saved_by_hash.get(digest)
                    if cached is None:
                        extension = _safe_extension(extension)
                        output_path = (
                            assets_root
                            / "images"
                            / document.document_id
                            / f"image_{digest}.{extension}"
                        )
                    else:
                        output_path, extension = cached

                    try:
                        width, height, saved_hash, reused = _persist_verified_image(
                            output_path, image_bytes, pymupdf
                        )
                    except Exception as exc:
                        error_message = (
                            "Could not save or verify extracted image "
                            f"({type(exc).__name__})."
                        )
                        file_errors.append(
                            f"{document.relative_path}, page {page_number}, "
                            f"image {occurrence_index}: {error_message}"
                        )
                        records.append(
                            make_asset_record(
                                document,
                                page_number,
                                "image_occurrence",
                                occurrence_index,
                                asset_type=image_type,
                                image_format=extension,
                                width_px=width,
                                height_px=height,
                                bbox=bbox,
                                image_xref=xref,
                                image_index=occurrence_index,
                                extraction_method=method,
                                status="error",
                                error_message=error_message,
                            )
                        )
                        continue

                    saved_by_hash[digest] = (output_path, extension)
                    if reused:
                        reused_paths.add(output_path)

                    records.append(
                        make_asset_record(
                            document,
                            page_number,
                            "image_occurrence",
                            occurrence_index,
                            asset_type=image_type,
                            asset_path=output_path.relative_to(repository_root).as_posix(),
                            image_format=extension,
                            width_px=width,
                            height_px=height,
                            bbox=bbox,
                            image_xref=xref,
                            image_index=occurrence_index,
                            sha256=saved_hash,
                            extraction_method=method,
                            status="ok",
                        )
                    )
        except Exception as exc:
            file_errors.append(
                f"{document.relative_path}: image occurrence iteration failed "
                f"({type(exc).__name__})."
            )
        finally:
            try:
                pdf.close()
            except Exception as exc:
                file_errors.append(
                    f"{document.relative_path}: PDF close failed ({type(exc).__name__})."
                )

    return AssetProcessingResult(
        records, pdfs_processed, file_errors, reused_files=len(reused_paths)
    )
