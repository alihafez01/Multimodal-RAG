"""Selective OCR for pages whose native PDF text is likely incomplete."""

from __future__ import annotations

import argparse
import importlib
import json
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import ManifestDocument, ManifestError, read_manifest, write_jsonl
from .progress import ProgressReporter, format_duration


LOW_TEXT_CHARACTERS = 40
LIMITED_TEXT_CHARACTERS = 250
LARGE_IMAGE_COVERAGE = 0.50
DEFAULT_DPI = 250
MIN_DPI = 100
MAX_DPI = 400
_EDGE_PUNCTUATION = " \t\r\n.,;:!?()[]{}<>\"'`“”‘’—–-"


@dataclass(frozen=True)
class TesseractStatus:
    executable: str | None
    version: str | None
    languages: tuple[str, ...]
    error: str | None = None


def non_whitespace_count(text: str) -> int:
    """Count visible characters without counting whitespace."""
    return sum(not character.isspace() for character in text)


def _rect_values(rect: Any) -> tuple[float, float, float, float] | None:
    try:
        return float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)
    except (AttributeError, TypeError, ValueError):
        try:
            return tuple(float(value) for value in rect)  # type: ignore[return-value]
        except (TypeError, ValueError):
            return None


def image_union_coverage(page: Any) -> float:
    """Return the fraction of page area covered by the union of image boxes."""
    try:
        page_rect = _rect_values(page.rect)
        infos = page.get_image_info(xrefs=True)
    except Exception:
        return 0.0
    if page_rect is None or not isinstance(infos, (list, tuple)):
        return 0.0
    px0, py0, px1, py1 = page_rect
    page_width, page_height = px1 - px0, py1 - py0
    if page_width <= 0 or page_height <= 0:
        return 0.0
    boxes: list[tuple[float, float, float, float]] = []
    for info in infos:
        if not isinstance(info, dict):
            continue
        box = _rect_values(info.get("bbox"))
        if box is None:
            continue
        x0, y0, x1, y1 = box
        box = (max(px0, x0), max(py0, y0), min(px1, x1), min(py1, y1))
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append(box)
    if not boxes:
        return 0.0

    # Sweep across unique x edges so overlapping images contribute only once.
    x_edges = sorted({coordinate for box in boxes for coordinate in (box[0], box[2])})
    union_area = 0.0
    for left, right in zip(x_edges, x_edges[1:]):
        if right <= left:
            continue
        intervals = sorted(
            (box[1], box[3]) for box in boxes if box[0] < right and box[2] > left
        )
        covered_y = 0.0
        if intervals:
            start, end = intervals[0]
            for next_start, next_end in intervals[1:]:
                if next_start <= end:
                    end = max(end, next_end)
                else:
                    covered_y += end - start
                    start, end = next_start, next_end
            covered_y += end - start
        union_area += (right - left) * covered_y
    return min(1.0, max(0.0, union_area / (page_width * page_height)))


def evaluate_page(
    native_text: str,
    page: Any | None = None,
    *,
    force: bool = False,
    low_text_characters: int = LOW_TEXT_CHARACTERS,
    limited_text_characters: int = LIMITED_TEXT_CHARACTERS,
    large_image_coverage: float = LARGE_IMAGE_COVERAGE,
) -> tuple[list[str], float]:
    """Select likely OCR pages from text amount and non-overlapping image coverage."""
    character_count = non_whitespace_count(native_text)
    coverage = image_union_coverage(page) if page is not None else 0.0
    reasons: list[str] = []
    if force:
        reasons.append("manual_override")
    else:
        if character_count < low_text_characters:
            reasons.append("very_low_native_text")
        if character_count < limited_text_characters and coverage >= large_image_coverage:
            reasons.append("large_image_with_limited_text")
    return reasons, coverage


def check_tesseract(timeout_seconds: float = 5.0) -> TesseractStatus:
    """Inspect Tesseract without installing or changing system configuration."""
    executable = shutil.which("tesseract")
    if executable is None:
        return TesseractStatus(None, None, (), "tesseract executable was not found on PATH")
    try:
        version_result = subprocess.run(
            [executable, "--version"], capture_output=True, text=True,
            timeout=timeout_seconds, check=False,
        )
        version_text = (version_result.stdout or version_result.stderr).splitlines()
        version = version_text[0].strip() if version_text else None
        language_result = subprocess.run(
            [executable, "--list-langs"], capture_output=True, text=True,
            timeout=timeout_seconds, check=False,
        )
        if language_result.returncode != 0:
            error = (language_result.stderr or language_result.stdout).strip()
            return TesseractStatus(executable, version, (), error or "could not list OCR languages")
        languages = tuple(
            line.strip() for line in language_result.stdout.splitlines()
            if line.strip() and not line.lower().startswith("list of available languages")
        )
        return TesseractStatus(executable, version, languages)
    except (OSError, subprocess.SubprocessError) as exc:
        return TesseractStatus(executable, None, (), f"Tesseract check failed ({type(exc).__name__})")


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ManifestError(f"Required {label} not found: {path}.")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ManifestError(f"Malformed {label} JSON on line {line_number}.") from exc
                if not isinstance(row, dict):
                    raise ManifestError(f"{label} line {line_number} must be a JSON object.")
                rows.append(row)
    except UnicodeDecodeError as exc:
        raise ManifestError(f"{label} must be UTF-8 JSON Lines.") from exc
    return rows


def load_page_records(
    manifest_path: Path, pages_path: Path, dataset_dir: Path
) -> tuple[list[ManifestDocument], list[dict[str, Any]], dict[str, int]]:
    """Validate pages.jsonl against the inventory order and expected page counts."""
    documents = read_manifest(manifest_path, dataset_dir)
    rows = _read_jsonl(pages_path, "page manifest")
    valid_documents = [document for document in documents if document.status == "valid"]
    expected: dict[str, ManifestDocument] = {doc.document_id: doc for doc in valid_documents}
    pages_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for line_number, row in enumerate(rows, start=1):
        document_id = row.get("document_id")
        page_number = row.get("page_number")
        text = row.get("text")
        if (
            document_id not in expected
            or isinstance(page_number, bool)
            or not isinstance(page_number, int)
            or page_number < 1
            or not isinstance(text, str)
        ):
            raise ManifestError(f"Page record {line_number} has invalid document/page/text fields.")
        document = expected[document_id]
        if page_number > (document.page_count or 0):
            raise ManifestError(f"Page record {line_number} exceeds its inventory page count.")
        key = (document_id, page_number)
        if key in pages_by_key:
            raise ManifestError(f"Duplicate page record for {document_id}, page {page_number}.")
        if row.get("filename") != document.filename or row.get("relative_path") != document.relative_path:
            raise ManifestError(f"Page record {line_number} source references disagree with inventory.")
        if row.get("text_char_count") != len(text):
            raise ManifestError(f"Page record {line_number} text_char_count is inconsistent.")
        pages_by_key[key] = row

    expected_count = sum(document.page_count or 0 for document in valid_documents)
    if len(pages_by_key) != expected_count:
        raise ManifestError(
            f"pages.jsonl has {len(pages_by_key)} records but inventory expects {expected_count}; "
            "rerun the existing PDF text extraction stage first."
        )
    ordered_rows = [
        pages_by_key[(document.document_id, page_number)]
        for document in valid_documents
        for page_number in range(1, (document.page_count or 0) + 1)
    ]
    skipped = Counter(document.status for document in documents if document.status != "valid")
    return documents, ordered_rows, dict(skipped)


def _normalise_line(line: str) -> str:
    normalised = unicodedata.normalize("NFKC", line).casefold()
    normalised = re.sub(r"\s+", " ", normalised).strip(_EDGE_PUNCTUATION)
    return normalised


def merge_native_and_ocr(native_text: str, ocr_text: str) -> str:
    """Keep native text verbatim and append only OCR lines not already represented."""
    if not ocr_text.strip():
        return native_text
    known = {_normalise_line(line) for line in native_text.splitlines() if _normalise_line(line)}
    additions: list[str] = []
    for line in ocr_text.splitlines():
        signature = _normalise_line(line)
        if signature and signature not in known:
            additions.append(line)
            known.add(signature)
    if not additions:
        return native_text
    if not native_text:
        return "\n".join(additions)
    separator = "" if native_text.endswith("\n") else "\n"
    return native_text + separator + "\n".join(additions)


def _extract_ocr_only_text(page: Any, text_page: Any, native_text: str) -> str:
    """Extract OCR-font spans from a TextPage while excluding retained native text."""
    try:
        layout = page.get_text("dict", textpage=text_page, sort=True)
        ocr_lines: list[str] = []
        has_ocr_font = False
        if isinstance(layout, dict):
            for block in layout.get("blocks", []):
                if not isinstance(block, dict):
                    continue
                for line in block.get("lines", []):
                    if not isinstance(line, dict):
                        continue
                    spans = line.get("spans", [])
                    selected = []
                    for span in spans:
                        if not isinstance(span, dict):
                            continue
                        if "glyphlessfont" in str(span.get("font", "")).casefold():
                            has_ocr_font = True
                            selected.append(span.get("text", ""))
                    text = "".join(selected)
                    if text.strip():
                        ocr_lines.append(text)
        if ocr_lines:
            return "\n".join(ocr_lines).strip()
        if has_ocr_font:
            return ""
    except Exception:
        pass
    # Compatibility fallback for bindings that cannot expose span font metadata.
    extracted = page.get_text("text", textpage=text_page, sort=True)
    if not isinstance(extracted, str):
        raise TypeError("PyMuPDF OCR returned non-text output")
    known = {_normalise_line(line) for line in native_text.splitlines() if _normalise_line(line)}
    return "\n".join(
        line for line in extracted.strip().splitlines()
        if _normalise_line(line) and _normalise_line(line) not in known
    )


def _page_candidates(
    documents: list[ManifestDocument],
    pages: list[dict[str, Any]],
    pymupdf: Any,
    *,
    force: bool,
    progress: ProgressReporter | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    page_by_key = {(row["document_id"], row["page_number"]): row for row in pages}
    candidates: list[dict[str, Any]] = []
    errors: list[str] = []
    visited: set[tuple[str, int]] = set()

    def advance(row: dict[str, Any], detail: str) -> None:
        key = (row["document_id"], row["page_number"])
        if key not in visited:
            visited.add(key)
            if progress is not None:
                progress.advance(detail)

    for document in documents:
        if document.status != "valid":
            continue
        pdf = None
        try:
            pdf = pymupdf.open(document.source_path)
            if getattr(pdf, "needs_pass", False):
                raise RuntimeError("PDF requires a password")
            for page_number in range(1, (document.page_count or 0) + 1):
                row = page_by_key[(document.document_id, page_number)]
                try:
                    page = pdf.load_page(page_number - 1)
                    reasons, coverage = evaluate_page(row["text"], page, force=force)
                except Exception as exc:
                    reasons, coverage = evaluate_page(row["text"], None, force=force)
                    errors.append(
                        f"{document.relative_path}, page {page_number}: page inspection failed ({type(exc).__name__})."
                    )
                if reasons:
                    candidates.append({"document": document, "page": row, "reasons": reasons, "coverage": coverage})
                advance(row, f"{document.filename} page {page_number}")
        except Exception as exc:
            errors.append(f"{document.relative_path}: PDF inspection failed ({type(exc).__name__}).")
            # A textless page remains an OCR candidate even if its PDF cannot be inspected.
            for page_number in range(1, (document.page_count or 0) + 1):
                row = page_by_key[(document.document_id, page_number)]
                if (document.document_id, page_number) in visited:
                    continue
                reasons, coverage = evaluate_page(row["text"], None, force=force)
                if reasons:
                    candidates.append({"document": document, "page": row, "reasons": reasons, "coverage": coverage})
                advance(row, f"{document.filename} page {page_number}")
        finally:
            if pdf is not None:
                try:
                    pdf.close()
                except Exception as exc:
                    errors.append(f"{document.relative_path}: PDF close failed ({type(exc).__name__}).")
    unique: dict[tuple[str, int], dict[str, Any]] = {}
    for candidate in candidates:
        key = (candidate["page"]["document_id"], candidate["page"]["page_number"])
        unique[key] = candidate
    return list(unique.values()), errors


def _empty_enriched_record(row: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    native = row["text"]
    copied = dict(row)
    copied.update({
        "native_text": native,
        "native_text_char_count": len(native),
        "ocr_text": "",
        "ocr_text_char_count": 0,
        "text": native,
        "text_char_count": len(native),
        "extraction_method": "native_text" if native else "none",
        "ocr_required": bool(reasons),
        "ocr_status": "not_needed" if not reasons else "pending",
        "ocr_trigger_reasons": list(reasons),
        "ocr_language": None,
        "ocr_dpi": None,
        "ocr_error": None,
    })
    return copied


def process_ocr(
    repository_root: Path,
    *,
    language: str = "eng",
    dpi: int = DEFAULT_DPI,
    force: bool = False,
    max_pages: int | None = None,
    dry_run: bool = False,
    pymupdf: Any | None = None,
    tesseract_status: TesseractStatus | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Select OCR candidates and atomically write enriched page records."""
    if not MIN_DPI <= dpi <= MAX_DPI:
        raise ValueError(f"DPI must be between {MIN_DPI} and {MAX_DPI}.")
    if max_pages is not None and max_pages < 0:
        raise ValueError("--max-pages must be zero or a positive integer.")
    run_started = time.perf_counter()
    root = repository_root.resolve()
    processed = root / "data" / "processed"
    documents, pages, skipped_documents = load_page_records(
        processed / "manifest.jsonl", processed / "pages.jsonl", root / "Dataset"
    )
    if pymupdf is None:
        pymupdf = importlib.import_module("pymupdf")
    valid_page_total = sum(document.page_count or 0 for document in documents if document.status == "valid")
    scan_progress = ProgressReporter("OCR candidate scan", valid_page_total, enabled=show_progress)
    scan_progress.start(f"Inspecting {len(documents)} inventory PDFs and {valid_page_total} valid pages.")
    candidates, inspection_errors = _page_candidates(
        documents, pages, pymupdf, force=force, progress=scan_progress
    )
    scan_progress.finish("OCR candidate scan")
    reason_counts = Counter(reason for item in candidates for reason in item["reasons"])
    output_path = processed / "pages_enriched.jsonl"
    if dry_run:
        print("OCR selection dry run")
        print(f"  PDFs inspected: {len(documents)}")
        print(f"  Page records examined: {len(pages)}")
        print(f"  Pages selected for OCR: {len(candidates)}")
        print(f"  Selection reasons: {dict(sorted(reason_counts.items())) or 'none'}")
        for candidate in candidates[:30]:
            row = candidate["page"]
            print(f"  - {row['relative_path']} page {row['page_number']}: {', '.join(candidate['reasons'])}")
        if len(candidates) > 30:
            print(f"  - ... and {len(candidates) - 30} more")
        print(f"  Enriched output not written: {output_path}")
        print(f"  Elapsed: {format_duration(time.perf_counter() - run_started)}")
        return {"dry_run": True, "documents": len(documents), "pages": len(pages), "candidates": candidates, "reason_counts": dict(reason_counts), "inspection_errors": inspection_errors}

    tesseract_status = tesseract_status or check_tesseract()
    candidate_keys = {
        (item["page"]["document_id"], item["page"]["page_number"]): item
        for item in candidates
    }
    allowed_keys = set(candidate_keys)
    if max_pages is not None:
        allowed_keys = set(list(candidate_keys)[:max_pages])
    records: list[dict[str, Any]] = []
    for row in pages:
        key = (row["document_id"], row["page_number"])
        candidate = candidate_keys.get(key)
        reasons = candidate["reasons"] if candidate else []
        records.append(_empty_enriched_record(row, reasons))
    by_key = {(row["document_id"], row["page_number"]): record for row, record in zip(pages, records)}

    requested_languages = tuple(part for part in language.split("+") if part)
    missing_languages = [code for code in requested_languages if code not in tesseract_status.languages]
    attempted = {
        key for key in allowed_keys
        if tesseract_status.executable and not missing_languages and requested_languages
    }
    if tesseract_status.executable is None:
        for key in allowed_keys:
            record = by_key[key]
            record["ocr_status"] = "skipped_tesseract_unavailable"
            record["ocr_error"] = tesseract_status.error or "Tesseract was not found on PATH."
    elif missing_languages or not requested_languages:
        for key in allowed_keys:
            record = by_key[key]
            record["ocr_status"] = "skipped_language_unavailable"
            if tesseract_status.error:
                record["ocr_error"] = f"Tesseract language data could not be checked: {tesseract_status.error}"
            elif missing_languages:
                record["ocr_error"] = f"Tesseract language data unavailable: {', '.join(missing_languages)}."
            else:
                record["ocr_error"] = "No usable Tesseract language code was supplied."
    for key in set(candidate_keys) - allowed_keys:
        by_key[key]["ocr_status"] = "skipped_max_pages"
        by_key[key]["ocr_error"] = f"Not processed because --max-pages limited OCR to {max_pages} page(s)."

    attempts_by_document: dict[str, list[tuple[str, int]]] = {}
    for key in candidate_keys:
        if key in attempted:
            attempts_by_document.setdefault(key[0], []).append(key)
    doc_by_id = {document.document_id: document for document in documents}
    success_count = 0
    processed_keys: set[tuple[str, int]] = set()
    ocr_progress = ProgressReporter("OCR selected pages", len(candidate_keys), enabled=show_progress)
    ocr_progress.start(
        f"Processing {len(candidate_keys)} selected pages; {len(attempted)} have Tesseract and language data available."
    )
    for document_id, keys in attempts_by_document.items():
        document = doc_by_id[document_id]
        pdf = None
        try:
            pdf = pymupdf.open(document.source_path)
            if getattr(pdf, "needs_pass", False):
                raise RuntimeError("PDF requires a password")
            for key in keys:
                record = by_key[key]
                page_number = key[1]
                try:
                    ocr_progress.set_detail(f"{document.filename} page {page_number}: OCR", refresh=ocr_progress.is_terminal)
                    page = pdf.load_page(page_number - 1)
                    full_page = non_whitespace_count(record["native_text"]) < LOW_TEXT_CHARACTERS
                    text_page = page.get_textpage_ocr(language=language, dpi=dpi, full=full_page)
                    raw_ocr = _extract_ocr_only_text(page, text_page, record["native_text"])
                    record["ocr_text"] = raw_ocr
                    record["ocr_text_char_count"] = len(raw_ocr)
                    record["text"] = merge_native_and_ocr(record["native_text"], raw_ocr)
                    record["text_char_count"] = len(record["text"])
                    if raw_ocr and record["native_text"]:
                        record["extraction_method"] = "native_text+ocr"
                    elif raw_ocr:
                        record["extraction_method"] = "ocr"
                    else:
                        record["extraction_method"] = "native_text" if record["native_text"] else "none"
                    record["ocr_status"] = "completed"
                    record["ocr_language"] = language
                    record["ocr_dpi"] = dpi
                    success_count += 1
                except Exception as exc:
                    record["ocr_status"] = "failed"
                    record["ocr_error"] = f"OCR failed ({type(exc).__name__})."
                    ocr_progress.log(
                        f"OCR failed for {document.filename} page {page_number} ({type(exc).__name__}).",
                        level="ERROR",
                    )
                processed_keys.add(key)
                ocr_progress.advance(f"{document.filename} page {page_number}: {record['ocr_status']}")
        except Exception as exc:
            for key in keys:
                record = by_key[key]
                record["ocr_status"] = "failed"
                record["ocr_error"] = f"PDF could not be opened for OCR ({type(exc).__name__})."
                processed_keys.add(key)
                ocr_progress.log(
                    f"PDF open failed for OCR: {document.filename} ({type(exc).__name__}).",
                    level="ERROR",
                )
                ocr_progress.advance(f"{document.filename} page {key[1]}: failed")
        finally:
            if pdf is not None:
                try:
                    pdf.close()
                except Exception:
                    pass
    for key, candidate in candidate_keys.items():
        if key not in processed_keys:
            record = by_key[key]
            ocr_progress.advance(
                f"{candidate['document'].filename} page {key[1]}: {record['ocr_status']}"
            )
    ocr_progress.finish("OCR candidate processing")
    write_jsonl(records, output_path)
    status_counts = Counter(record["ocr_status"] for record in records)
    print("Selective OCR summary")
    print(f"  PDFs inspected: {len(documents)}")
    print(f"  PDFs skipped by inventory status: {sum(skipped_documents.values())}")
    print(f"  Page records examined: {len(pages)}")
    print(f"  Pages selected for OCR: {len(candidates)}")
    print(f"  OCR successes: {success_count}")
    print(f"  Pages not requiring OCR: {status_counts.get('not_needed', 0)}")
    print(f"  Pages skipped because Tesseract is unavailable: {status_counts.get('skipped_tesseract_unavailable', 0)}")
    print(f"  Pages skipped because language data is unavailable: {status_counts.get('skipped_language_unavailable', 0)}")
    print(f"  Pages skipped by --max-pages: {status_counts.get('skipped_max_pages', 0)}")
    print(f"  Pages with OCR errors: {status_counts.get('failed', 0)}")
    print(f"  Total elapsed: {format_duration(time.perf_counter() - run_started)}")
    print(f"  Selection reasons: {dict(sorted(reason_counts.items())) or 'none'}")
    print(f"  Tesseract version: {tesseract_status.version or 'unavailable'}")
    print(f"  Requested language: {language}; available: {', '.join(tesseract_status.languages) or 'none'}")
    if tesseract_status.error:
        print(f"  Tesseract detail: {tesseract_status.error}")
    if candidates and not tesseract_status.executable:
        print("  Prerequisite: install Tesseract and requested language data, then rerun python scripts/run_ocr.py.")
    if not candidates:
        print("  The heuristic found no OCR candidates.")
    print(f"  Output: {output_path}")
    return {
        "dry_run": False, "documents": len(documents), "pages": len(pages),
        "candidates": candidates, "reason_counts": dict(reason_counts),
        "status_counts": dict(status_counts), "successes": success_count,
        "inspection_errors": inspection_errors, "output_path": output_path,
        "tesseract": tesseract_status,
    }


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _dpi(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("DPI must be an integer") from exc
    if not MIN_DPI <= parsed <= MAX_DPI:
        raise argparse.ArgumentTypeError(f"DPI must be between {MIN_DPI} and {MAX_DPI}")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run selective OCR on PDF pages.")
    parser.add_argument("--dry-run", action="store_true", help="List candidates without writing output.")
    parser.add_argument("--language", default="eng", help="Tesseract language code (default: eng).")
    parser.add_argument("--dpi", type=_dpi, default=DEFAULT_DPI, help="OCR resolution, 100-400 DPI.")
    parser.add_argument("--force", action="store_true", help="Select every valid page for OCR.")
    parser.add_argument("--max-pages", type=_positive_int, help="Limit OCR attempts for controlled runs.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars; timestamped phase logs remain.")
    return parser


def main(argv: list[str] | None = None, *, repository_root: Path | None = None) -> int:
    root = repository_root or Path(__file__).resolve().parents[3]
    args = build_argument_parser().parse_args(argv)
    try:
        process_ocr(root, language=args.language, dpi=args.dpi, force=args.force,
                    max_pages=args.max_pages, dry_run=args.dry_run,
                    show_progress=not args.no_progress)
    except (ManifestError, OSError, RuntimeError, ValueError, ImportError) as exc:
        print(f"OCR processing could not continue: {exc}", file=sys.stderr)
        return 2
    return 0
