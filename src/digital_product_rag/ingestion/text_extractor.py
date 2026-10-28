"""Extract native text into one JSONL record per PDF page."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import ManifestDocument, write_jsonl


OCR_TEXT_THRESHOLD = 30
"""Pages with fewer non-whitespace characters may benefit from later OCR/visual review."""


@dataclass
class TextExtractionResult:
    records: list[dict[str, Any]]
    pdfs_processed: int
    expected_page_count: int
    skipped_by_status: dict[str, int]
    file_errors: list[str]


def _page_record(
    document: ManifestDocument,
    page_number: int,
    text: str,
    *,
    extraction_method: str,
    status: str,
    error_message: str | None = None,
) -> dict[str, Any]:
    return {
        "document_id": document.document_id,
        "filename": document.filename,
        "relative_path": document.relative_path,
        "page_number": page_number,
        "text": text,
        "text_char_count": len(text),
        "extraction_method": extraction_method,
        "ocr_required": (
            sum(not character.isspace() for character in text) < OCR_TEXT_THRESHOLD
            if status == "ok"
            else True
        ),
        "status": status,
        "error_message": error_message,
    }


def extract_pages(
    documents: list[ManifestDocument],
    output_path: Path,
    pymupdf: Any,
) -> TextExtractionResult:
    """Extract each valid manifest PDF page and atomically write pages.jsonl."""
    valid_documents = [document for document in documents if document.status == "valid"]
    skipped_by_status: dict[str, int] = {}
    for document in documents:
        if document.status != "valid":
            skipped_by_status[document.status] = skipped_by_status.get(document.status, 0) + 1

    records: list[dict[str, Any]] = []
    file_errors: list[str] = []
    pdfs_processed = 0
    expected_page_count = sum(document.page_count or 0 for document in valid_documents)

    for document in valid_documents:
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

        try:
            for zero_based_page in range(page_count):
                page_number = zero_based_page + 1
                try:
                    page = pdf.load_page(zero_based_page)
                    text = page.get_text("text", sort=True)
                    if not isinstance(text, str):
                        raise TypeError("page text is not a string")
                    text = text.strip()
                    records.append(
                        _page_record(
                            document,
                            page_number,
                            text,
                            extraction_method="native_text",
                            status="ok",
                        )
                    )
                except Exception as exc:
                    records.append(
                        _page_record(
                            document,
                            page_number,
                            "",
                            extraction_method="failed",
                            status="error",
                            error_message=f"Page text extraction failed ({type(exc).__name__}).",
                        )
                    )
        except Exception as exc:
            file_errors.append(
                f"{document.relative_path}: page iteration failed ({type(exc).__name__})."
            )
        finally:
            try:
                pdf.close()
            except Exception as exc:
                file_errors.append(
                    f"{document.relative_path}: PDF close failed ({type(exc).__name__})."
                )

    write_jsonl(records, output_path)
    return TextExtractionResult(
        records=records,
        pdfs_processed=pdfs_processed,
        expected_page_count=expected_page_count,
        skipped_by_status=skipped_by_status,
        file_errors=file_errors,
    )
