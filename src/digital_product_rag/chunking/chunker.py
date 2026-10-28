"""Build deterministic, source-linked chunks from processed PDF data."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from ..catalog import CatalogResult, prepare_product_catalog
from ..ingestion.common import ManifestDocument, ManifestError, atomic_write_bytes, read_manifest
from ..schemas import Chunk, Product


DEFAULT_MAX_CHARS = 1800
DEFAULT_OVERLAP_CHARS = 180
DEFAULT_MIN_CHARS = 24
CHUNK_NAMESPACE = uuid.UUID("ec4c27ad-65a6-47a0-96bd-05f88624fc95")

_HEADING_LABELS = {
    "overview", "introduction", "materials", "supplies", "tools", "abbreviations",
    "notes", "gauge", "sizes", "sizing", "finished size", "instructions",
    "assembly", "finishing", "troubleshooting", "contents", "usage", "how to use",
    "page instructions", "worksheet", "planning", "setup", "preparation", "care",
    "terms", "glossary", "safety", "step-by-step", "getting started", "conclusion",
}
_ROUND_ROW_STEP = re.compile(r"^(?:round|rnd|row|step|part|section)\s*#?\s*\d+[a-z]?(?:\s*[:—-].*)?$", re.I)
_NUMBERED_HEADING = re.compile(r"^\d+(?:\.\d+)*[.)]?\s+(.+)$")
_IMPERATIVES = {
    "add", "attach", "begin", "chain", "choose", "click", "close", "continue", "create",
    "cut", "fold", "fasten", "fill", "insert", "join", "leave", "make", "open", "place",
    "press", "repeat", "select", "sew", "stitch", "turn", "use", "work", "write",
}


class ChunkingError(ValueError):
    """Raised when source records cannot be safely converted into chunks."""


@dataclass
class _Atom:
    text: str
    page_number: int


@dataclass
class _Section:
    title: str | None
    heading: _Atom | None = None
    atoms: list[_Atom] = field(default_factory=list)


@dataclass
class PageInput:
    pages: list[dict[str, Any]]
    source_name: str
    fallback_reason: str | None
    blank_pages: int
    page_errors: int
    skipped_documents: int


@dataclass
class ChunkBuildResult:
    chunks: list[dict[str, Any]]
    catalog: CatalogResult
    source_name: str
    fallback_reason: str | None
    pages_examined: int
    blank_pages: int
    page_errors: int
    skipped_documents: int
    invalid_assets: int
    ignored_visual_records: int
    skipped_short_text: int


def validate_chunk_settings(max_chars: int, overlap_chars: int, min_chars: int) -> None:
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 64:
        raise ValueError("max_chars must be an integer of at least 64.")
    if isinstance(overlap_chars, bool) or not isinstance(overlap_chars, int) or overlap_chars < 0:
        raise ValueError("overlap_chars must be a non-negative integer.")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars.")
    if isinstance(min_chars, bool) or not isinstance(min_chars, int) or min_chars < 1:
        raise ValueError("min_chars must be a positive integer.")


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ChunkingError(f"Malformed {label} JSON on line {line_number}.") from exc
                if not isinstance(value, dict):
                    raise ChunkingError(f"{label} line {line_number} must be an object.")
                rows.append(value)
    except UnicodeDecodeError as exc:
        raise ChunkingError(f"{label} must be UTF-8 JSON Lines.") from exc
    except OSError as exc:
        raise ChunkingError(f"Could not read {label} at {path}: {exc}") from exc
    return rows


def _validate_pages(
    rows: list[dict[str, Any]], documents: list[ManifestDocument], label: str
) -> list[dict[str, Any]]:
    document_by_id = {doc.document_id: doc for doc in documents if doc.status == "valid"}
    output: dict[tuple[str, int], dict[str, Any]] = {}
    for line_number, row in enumerate(rows, start=1):
        document_id, page_number = row.get("document_id"), row.get("page_number")
        document = document_by_id.get(document_id)
        text, count = row.get("text"), row.get("text_char_count")
        if (
            document is None
            or isinstance(page_number, bool)
            or not isinstance(page_number, int)
            or page_number < 1
            or page_number > (document.page_count or 0)
            or row.get("filename") != document.filename
            or row.get("relative_path") != document.relative_path
            or not isinstance(text, str)
            or isinstance(count, bool)
            or count != len(text)
        ):
            raise ChunkingError(f"{label} page record {line_number} has invalid source or text fields.")
        if label == "enriched pages" and row.get("ocr_status") not in (None, "completed"):
            native = row.get("native_text")
            if isinstance(native, str):
                row = dict(row)
                row["text"] = native
                row["text_char_count"] = len(native)
        key = (document_id, page_number)
        if key in output:
            raise ChunkingError(f"{label} has duplicate page {page_number} for {document_id}.")
        output[key] = row

    expected = sum(doc.page_count or 0 for doc in document_by_id.values())
    if len(output) != expected:
        raise ChunkingError(f"{label} has {len(output)} pages; inventory expects {expected}.")
    order = {doc.document_id: index for index, doc in enumerate(documents)}
    return sorted(output.values(), key=lambda row: (order[row["document_id"]], row["page_number"]))


def load_page_input(repository_root: Path, documents: list[ManifestDocument]) -> PageInput:
    processed = repository_root / "data" / "processed"
    enriched_path = processed / "pages_enriched.jsonl"
    base_path = processed / "pages.jsonl"
    reason: str | None = None
    if enriched_path.is_file():
        try:
            pages = _validate_pages(_read_jsonl(enriched_path, "enriched pages"), documents, "enriched pages")
            source_name = "pages_enriched.jsonl"
        except ChunkingError as exc:
            reason = str(exc)
        else:
            return _page_stats(pages, source_name, None, documents)
    if not base_path.is_file():
        if reason:
            raise ChunkingError(f"Enriched pages are invalid ({reason}) and pages.jsonl is missing.")
        raise ChunkingError(f"Neither a valid pages_enriched.jsonl nor pages.jsonl exists under {processed}.")
    try:
        pages = _validate_pages(_read_jsonl(base_path, "pages"), documents, "pages")
    except ChunkingError as exc:
        if reason:
            raise ChunkingError(f"Enriched pages are invalid ({reason}); fallback pages are also invalid ({exc}).") from exc
        raise
    return _page_stats(pages, "pages.jsonl", reason, documents)


def _page_stats(
    pages: list[dict[str, Any]], source_name: str, reason: str | None, documents: list[ManifestDocument]
) -> PageInput:
    nonvalid = sum(doc.status != "valid" for doc in documents)
    return PageInput(
        pages=pages,
        source_name=source_name,
        fallback_reason=reason,
        blank_pages=sum(not row["text"].strip() for row in pages),
        page_errors=sum(row.get("status") != "ok" for row in pages),
        skipped_documents=nonvalid,
    )


def _next_nonempty(lines: list[str], start: int) -> str | None:
    for line in lines[start:]:
        if line.strip():
            return line.strip()
    return None


def _looks_like_heading(line: str, following: str | None) -> bool:
    candidate = line.strip().strip("•*- ")
    if not candidate or len(candidate) > 100:
        return False
    normalized = candidate.casefold().rstrip(":")
    if normalized in _HEADING_LABELS or _ROUND_ROW_STEP.match(candidate):
        return True
    numbered = _NUMBERED_HEADING.match(candidate)
    if numbered:
        title = numbered.group(1).strip()
        first = re.match(r"[A-Za-z]+", title)
        if first and first.group(0).casefold() in _IMPERATIVES:
            return False
        if title.casefold().rstrip(":") in _HEADING_LABELS or _ROUND_ROW_STEP.match(title):
            return True
        if len(title.split()) <= 6 and not re.search(r"[.!?]$", title):
            return True
    if candidate.isupper() and any(char.isalpha() for char in candidate):
        return True
    if following and not re.search(r"[.!?]$", candidate):
        words = candidate.split()
        title_case = 1 <= len(words) <= 7 and all(
            word[:1].isupper() or word.casefold() in {"and", "or", "of", "the", "to", "for", "in"}
            for word in words
        )
        if title_case and len(following) > len(candidate):
            return True
    return False


def _split_page_into_sections(page: dict[str, Any], sections: list[_Section]) -> None:
    lines = page["text"].splitlines()
    body_lines: list[str] = []

    def flush_body() -> None:
        nonlocal body_lines
        text = "\n".join(line.rstrip() for line in body_lines).strip()
        if text:
            if not sections:
                sections.append(_Section(title=None))
            sections[-1].atoms.append(_Atom(text, page["page_number"]))
        body_lines = []

    for index, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            flush_body()
            continue
        following = _next_nonempty(lines, index + 1)
        if _looks_like_heading(line, following):
            flush_body()
            section = _Section(title=line, heading=_Atom(line, page["page_number"]))
            sections.append(section)
            continue
        body_lines.append(raw)
    flush_body()


def _split_long_text(text: str, limit: int) -> list[str]:
    """Split a long paragraph at sentence/word boundaries, preserving its wording."""
    remaining = text.strip()
    pieces: list[str] = []
    while len(remaining) > limit:
        window = remaining[:limit + 1]
        sentence_ends = [match.end() for match in re.finditer(r"[.!?][\"')\]]*(?:\s+|$)", window)]
        cut = max((position for position in sentence_ends if position >= limit * 0.45), default=0)
        if not cut:
            whitespace = [match.start() for match in re.finditer(r"\s+", window) if match.start() >= limit * 0.5]
            cut = max(whitespace, default=limit)
        piece = remaining[:cut].strip()
        if not piece:
            cut = limit
            piece = remaining[:cut]
        pieces.append(piece)
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _overlap_tail(atoms: list[_Atom], overlap_chars: int) -> list[_Atom]:
    if overlap_chars <= 0:
        return []
    selected: list[_Atom] = []
    used = 0
    for atom in reversed(atoms):
        room = overlap_chars - used
        if room <= 0:
            break
        text = atom.text[-room:] if len(atom.text) > room else atom.text
        if text:
            selected.append(_Atom(text, atom.page_number))
            used += len(text) + (1 if used else 0)
    return list(reversed(selected))


def _render_section_text(section: _Section, atoms: list[_Atom]) -> str:
    parts: list[str] = []
    if section.heading:
        parts.append(section.heading.text)
    parts.extend(atom.text for atom in atoms if atom.text)
    return "\n".join(parts).strip()


def _section_chunks(
    section: _Section,
    *,
    max_chars: int,
    overlap_chars: int,
    min_chars: int,
) -> tuple[list[tuple[str, tuple[int, ...]]], int]:
    if not section.atoms:
        return [], 0
    prefix_length = len(section.heading.text) if section.heading else 0
    # Reserve room for the overlap so that subsequent chunks fit the configured limit.
    payload_limit = max(1, max_chars - prefix_length - overlap_chars - (1 if prefix_length else 0))
    atoms: list[_Atom] = []
    for atom in section.atoms:
        pieces = _split_long_text(atom.text, payload_limit)
        atoms.extend(_Atom(piece, atom.page_number) for piece in pieces)

    output: list[tuple[str, tuple[int, ...]]] = []
    current: list[_Atom] = []
    skipped_short = 0

    def emit(items: list[_Atom]) -> None:
        nonlocal skipped_short
        if not items:
            return
        text = _render_section_text(section, items)
        if len(text) < min_chars and section.title is None:
            skipped_short += 1
            return
        pages = {atom.page_number for atom in items}
        if section.heading:
            pages.add(section.heading.page_number)
        output.append((text, tuple(sorted(pages))))

    for atom in atoms:
        proposed = current + [atom]
        if len(_render_section_text(section, proposed)) <= max_chars:
            current.append(atom)
            continue
        if current:
            emit(current)
            current = _overlap_tail(current, overlap_chars)
        # An unusually long heading can consume the configured limit; retain the original
        # heading and permit that indivisible heading as a documented boundary exception.
        if len(_render_section_text(section, current + [atom])) > max_chars and current:
            current = []
        current.append(atom)
    emit(current)
    return output, skipped_short


def _chunk_id(
    document_id: str,
    product_id: str,
    chunk_type: str,
    source_pages: Iterable[int],
    section_title: str | None,
    text: str,
    asset_id: str | None = None,
    boundary_key: str = "",
) -> str:
    content_signature = hashlib.sha256(text.encode("utf-8")).hexdigest()
    parts = [document_id, product_id, chunk_type, ",".join(map(str, source_pages)), section_title or "", content_signature, asset_id or "", boundary_key]
    return str(uuid.uuid5(CHUNK_NAMESPACE, "\x1f".join(parts)))


def _text_chunks_for_document(
    document: ManifestDocument,
    product: Product,
    pages: list[dict[str, Any]],
    *,
    max_chars: int,
    overlap_chars: int,
    min_chars: int,
) -> tuple[list[dict[str, Any]], int]:
    sections: list[_Section] = []
    for page in pages:
        _split_page_into_sections(page, sections)
    output: list[dict[str, Any]] = []
    skipped = 0
    for section_index, section in enumerate(sections):
        section_parts, skipped_count = _section_chunks(
            section, max_chars=max_chars, overlap_chars=overlap_chars, min_chars=min_chars
        )
        skipped += skipped_count
        for part_index, (text, source_pages) in enumerate(section_parts):
            if not text.strip() or not source_pages:
                continue
            chunk_id = _chunk_id(
                document.document_id, product.product_id, "text", source_pages,
                section.title, text, boundary_key=f"section:{section_index}:part:{part_index}"
            )
            output.append(Chunk(
                chunk_id=chunk_id,
                document_id=document.document_id,
                product_id=product.product_id,
                product_title=product.title,
                product_category=product.category,
                product_version=product.version,
                filename=document.filename,
                relative_path=document.relative_path,
                page_start=min(source_pages),
                page_end=max(source_pages),
                source_pages=source_pages,
                section_title=section.title,
                chunk_type="text",
                text=text,
                metadata_status=product.metadata_status,
            ).to_dict())
    return output, skipped


def _safe_asset_path(value: Any, repository_root: Path) -> Path | None:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        return None
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {".", ".."} for part in relative.parts):
        return None
    if relative.parts[:2] != ("data", "assets"):
        return None
    root = repository_root.resolve()
    asset_root = (root / "data" / "assets").resolve()
    path = (root / Path(*relative.parts)).resolve()
    return path if path.is_relative_to(asset_root) else None


def _visual_chunks(
    repository_root: Path,
    documents: list[ManifestDocument],
    product_by_document: dict[str, Product],
) -> tuple[list[dict[str, Any]], int, int]:
    processed = repository_root / "data" / "processed"
    asset_path = processed / "assets.jsonl"
    visual_path = processed / "visual_metadata.jsonl"
    if not visual_path.is_file():
        return [], 0, 0
    if not asset_path.is_file():
        return [], 0, 0
    document_by_id = {doc.document_id: doc for doc in documents}
    assets = _read_jsonl(asset_path, "assets manifest")
    assets_by_id: dict[str, dict[str, Any]] = {}
    invalid_assets = 0
    for row in assets:
        asset_id = row.get("asset_id")
        document = document_by_id.get(row.get("document_id"))
        page_number = row.get("page_number")
        file_path = _safe_asset_path(row.get("asset_path"), repository_root)
        if (
            not isinstance(asset_id, str) or not asset_id or asset_id in assets_by_id
            or document is None or row.get("relative_path") != document.relative_path
            or row.get("filename") != document.filename
            or isinstance(page_number, bool) or not isinstance(page_number, int)
            or page_number < 1 or page_number > (document.page_count or 0)
        ):
            invalid_assets += 1
            continue
        assets_by_id[asset_id] = row
        if row.get("status") != "ok" or file_path is None or not file_path.is_file():
            invalid_assets += 1

    visual_rows = _read_jsonl(visual_path, "visual metadata")
    output: list[dict[str, Any]] = []
    ignored = 0
    for row in visual_rows:
        description = row.get("description")
        status = row.get("description_status")
        source = row.get("description_source")
        generated = status == "generated" and source == "vision_model" and isinstance(row.get("description_model"), str) and bool(row["description_model"].strip())
        verified = status in {"verified", "manually_verified"} and source in {"manual", "user"}
        if not isinstance(description, str) or not description.strip() or not (generated or verified):
            ignored += 1
            continue
        asset_id = row.get("asset_id")
        asset = assets_by_id.get(asset_id)
        if asset is None or asset.get("status") != "ok":
            invalid_assets += 1
            continue
        document = document_by_id.get(asset.get("document_id"))
        asset_file = _safe_asset_path(asset.get("asset_path"), repository_root)
        if (
            document is None or asset_file is None or not asset_file.is_file()
            or row.get("document_id") != asset.get("document_id")
            or row.get("page_number") != asset.get("page_number")
            or row.get("asset_path") != asset.get("asset_path")
        ):
            invalid_assets += 1
            continue
        product = product_by_document[document.document_id]
        page_number = asset["page_number"]
        text = description.strip()
        source_pages = (page_number,)
        output.append(Chunk(
            chunk_id=_chunk_id(document.document_id, product.product_id, "image_description", source_pages, None, text, asset_id),
            document_id=document.document_id,
            product_id=product.product_id,
            product_title=product.title,
            product_category=product.category,
            product_version=product.version,
            filename=document.filename,
            relative_path=document.relative_path,
            page_start=page_number,
            page_end=page_number,
            source_pages=source_pages,
            section_title=None,
            chunk_type="image_description",
            text=text,
            asset_id=asset_id,
            asset_path=asset["asset_path"],
            metadata_status=product.metadata_status,
        ).to_dict())
    return output, invalid_assets, ignored


def build_chunks(
    repository_root: Path,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
    output_path: Path | None = None,
) -> ChunkBuildResult:
    """Validate inputs and atomically write structure-aware text and visual chunks."""
    validate_chunk_settings(max_chars, overlap_chars, min_chars)
    repository_root = repository_root.resolve()
    manifest_path = repository_root / "data" / "processed" / "manifest.jsonl"
    dataset_dir = repository_root / "Dataset"
    try:
        documents = read_manifest(manifest_path, dataset_dir)
    except ManifestError as exc:
        raise ChunkingError(str(exc)) from exc
    title_by_document: dict[str, str] = {}
    try:
        for row in _read_jsonl(manifest_path, "inventory manifest"):
            title = row.get("product_title")
            if isinstance(title, str) and title.strip():
                title_by_document[row["document_id"]] = title.strip()
    except (ChunkingError, KeyError) as exc:
        raise ChunkingError(f"Could not read manifest product titles: {exc}") from exc
    catalog = prepare_product_catalog(
        repository_root, documents, title_by_document=title_by_document
    )
    page_input = load_page_input(repository_root, documents)

    pages_by_document: dict[str, list[dict[str, Any]]] = {}
    for page in page_input.pages:
        pages_by_document.setdefault(page["document_id"], []).append(page)
    chunks: list[dict[str, Any]] = []
    skipped_short = 0
    for document in documents:
        if document.status != "valid":
            continue
        doc_chunks, skipped = _text_chunks_for_document(
            document,
            catalog.product_by_document[document.document_id],
            pages_by_document.get(document.document_id, []),
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            min_chars=min_chars,
        )
        chunks.extend(doc_chunks)
        skipped_short += skipped

    visual, invalid_assets, ignored_visual = _visual_chunks(
        repository_root, documents, catalog.product_by_document
    )
    chunks.extend(visual)
    document_order = {doc.document_id: index for index, doc in enumerate(documents)}
    chunks.sort(key=lambda row: (
        document_order[row["document_id"]], row["page_start"],
        0 if row["chunk_type"] == "text" else 1,
        row["asset_id"] or "", row["chunk_id"],
    ))
    ids = [row["chunk_id"] for row in chunks]
    if len(ids) != len(set(ids)):
        raise ChunkingError("Duplicate deterministic chunk IDs were generated.")
    destination = output_path or (repository_root / "data" / "processed" / "chunks.jsonl")
    contents = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in chunks).encode("utf-8")
    atomic_write_bytes(destination, contents)
    return ChunkBuildResult(
        chunks=chunks,
        catalog=catalog,
        source_name=page_input.source_name,
        fallback_reason=page_input.fallback_reason,
        pages_examined=len(page_input.pages),
        blank_pages=page_input.blank_pages,
        page_errors=page_input.page_errors,
        skipped_documents=page_input.skipped_documents,
        invalid_assets=invalid_assets,
        ignored_visual_records=ignored_visual,
        skipped_short_text=skipped_short,
    )
