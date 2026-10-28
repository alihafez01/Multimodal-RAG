"""Render every page of valid manifest PDFs to RGB PNG previews."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import (
    AssetProcessingResult,
    ManifestDocument,
    atomic_write_bytes,
    inspect_raster_file,
    make_asset_record,
)


MIN_DPI = 72
MAX_DPI = 300
DEFAULT_DPI = 120


def validate_dpi(dpi: int) -> int:
    """Validate a practical rendering resolution."""
    if isinstance(dpi, bool) or not isinstance(dpi, int) or not MIN_DPI <= dpi <= MAX_DPI:
        raise ValueError(f"DPI must be between {MIN_DPI} and {MAX_DPI}.")
    return dpi


def render_pages(
    documents: list[ManifestDocument],
    repository_root: Path,
    assets_root: Path,
    *,
    dpi: int = DEFAULT_DPI,
    pymupdf: Any,
) -> AssetProcessingResult:
    """Render valid PDFs page by page and return page-preview asset records."""
    validate_dpi(dpi)
    repository_root = repository_root.resolve()
    assets_root = assets_root.resolve()
    records: list[dict[str, Any]] = []
    file_errors: list[str] = []
    pdfs_processed = 0
    reused_files = 0

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
            message = f"Could not open valid inventory PDF ({type(exc).__name__})."
            file_errors.append(f"{document.relative_path}: {message}")
            for page_number in range(1, (document.page_count or 0) + 1):
                records.append(
                    make_asset_record(
                        document,
                        page_number,
                        "page_preview",
                        1,
                        asset_type="page_preview",
                        extraction_method="page_render",
                        status="error",
                        error_message=message,
                    )
                )
            if pdf is not None:
                try:
                    pdf.close()
                except Exception:
                    pass
            continue

        output_dir = assets_root / "page_previews" / document.document_id
        scale = dpi / 72.0
        try:
            for zero_based_page in range(page_count):
                page_number = zero_based_page + 1
                output_path = output_dir / f"page_{page_number:04d}.png"
                try:
                    page = pdf.load_page(zero_based_page)
                    matrix = pymupdf.Matrix(scale, scale)
                    transformed_rect = page.rect * matrix
                    expected_rect = transformed_rect.irect
                    expected_size = (expected_rect.width, expected_rect.height)

                    reused = False
                    try:
                        width, height, digest = inspect_raster_file(output_path, pymupdf)
                        with output_path.open("rb") as existing_file:
                            valid_png = existing_file.read(8) == b"\x89PNG\r\n\x1a\n"
                        reused = valid_png and (width, height) == expected_size
                    except Exception:
                        reused = False

                    if not reused:
                        pixmap = page.get_pixmap(
                            matrix=matrix,
                            colorspace=pymupdf.csRGB,
                            alpha=False,
                        )
                        image_bytes = pixmap.tobytes("png")
                        atomic_write_bytes(output_path, image_bytes)
                        width, height, digest = inspect_raster_file(output_path, pymupdf)
                        if (width, height) != expected_size or not image_bytes.startswith(
                            b"\x89PNG\r\n\x1a\n"
                        ):
                            raise ValueError(
                                "Saved page preview failed format or dimension validation"
                            )
                    else:
                        reused_files += 1

                    records.append(
                        make_asset_record(
                            document,
                            page_number,
                            "page_preview",
                            1,
                            asset_type="page_preview",
                            asset_path=output_path.relative_to(repository_root).as_posix(),
                            image_format="png",
                            width_px=width,
                            height_px=height,
                            sha256=digest,
                            extraction_method="page_render",
                            status="ok",
                        )
                    )
                except Exception as exc:
                    file_errors.append(
                        f"{document.relative_path}, page {page_number}: "
                        f"page rendering failed ({type(exc).__name__})."
                    )
                    records.append(
                        make_asset_record(
                            document,
                            page_number,
                            "page_preview",
                            1,
                            asset_type="page_preview",
                            extraction_method="page_render",
                            status="error",
                            error_message=(
                                "Page rendering or validation failed "
                                f"({type(exc).__name__})."
                            ),
                        )
                    )
        except Exception as exc:
            file_errors.append(
                f"{document.relative_path}: page rendering iteration failed ({type(exc).__name__})."
            )
        finally:
            try:
                pdf.close()
            except Exception as exc:
                file_errors.append(
                    f"{document.relative_path}: PDF close failed ({type(exc).__name__})."
                )

    return AssetProcessingResult(records, pdfs_processed, file_errors, reused_files)
