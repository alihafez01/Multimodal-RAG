"""Streamlit interface for the local, evidence-grounded digital-product assistant."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
# The app may use the local cached embedding model, but must never download weights itself.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import streamlit as st  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402

from digital_product_rag.application import (  # noqa: E402
    load_catalog,
    load_document_inventory,
    query_knowledge_base,
)
from digital_product_rag.ingestion.common import ManifestDocument  # noqa: E402
from digital_product_rag.retrieval.embeddings import load_sentence_transformer  # noqa: E402
from digital_product_rag.retrieval.keyword_search import KeywordSearchIndex  # noqa: E402
from digital_product_rag.retrieval.vector_store import (  # noqa: E402
    VectorStoreError,
    index_paths,
    load_index_bundle,
)


PAGE_PREVIEW_LIMIT = 2
PAGE_PREVIEW_WIDTH = 360
CONVERSATION_SCHEMA_VERSION = 2
EVALUATION_RESULTS_PATH = Path("data/processed/evaluation_results.json")


def _signature(paths: list[Path]) -> tuple[tuple[str, int, int], ...]:
    """Cheap cache key that invalidates resources when source files are replaced or edited."""
    parts: list[tuple[str, int, int]] = []
    for path in paths:
        try:
            stat = path.stat()
            parts.append((str(path.resolve()), stat.st_mtime_ns, stat.st_size))
        except OSError:
            parts.append((str(path.resolve()), -1, -1))
    return tuple(parts)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required metadata file is missing: {path}.")
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSON on line {line_number} of {path}.") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Record on line {line_number} of {path} must be an object.")
                records.append(record)
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc
    return records


@st.cache_data(show_spinner=False)
def _cached_catalog(root: str, signature: tuple[tuple[str, int, int], ...]):
    del signature  # included in Streamlit's cache key
    return load_catalog(Path(root))


@st.cache_data(show_spinner=False)
def _cached_inventory(root: str, signature: tuple[tuple[str, int, int], ...]):
    del signature
    return load_document_inventory(Path(root))


@st.cache_data(show_spinner=False)
def _cached_assets(root: str, signature: tuple[tuple[str, int, int], ...]):
    del signature
    return _read_jsonl(Path(root) / "data" / "processed" / "assets.jsonl")


@st.cache_resource(show_spinner=False)
def _cached_keyword_index(root: str, signature: tuple[tuple[str, int, int], ...]):
    del signature
    return KeywordSearchIndex(Path(root) / "data" / "processed" / "chunks.jsonl")


@st.cache_resource(show_spinner=False)
def _cached_index_bundle(root: str, signature: tuple[tuple[str, int, int], ...]):
    del signature
    return load_index_bundle(Path(root))


@st.cache_resource(show_spinner="Loading the cached local embedding model…")
def _cached_embedding_model(model_name: str):
    return load_sentence_transformer(model_name, device="cpu")


def _safe_image_path(repository_root: Path, asset_record: dict[str, Any]) -> tuple[Path | None, str | None]:
    """Resolve and verify an image path that came from the processed asset manifest."""
    value = asset_record.get("asset_path")
    if not isinstance(value, str) or not value.strip():
        return None, "The asset manifest has no image path."
    candidate_path = Path(value)
    if candidate_path.is_absolute():
        return None, "The asset manifest contains an unexpected absolute path."
    root = repository_root.resolve()
    try:
        path = (root / candidate_path).resolve(strict=True)
    except OSError:
        return None, "The referenced image file is missing."
    if not path.is_relative_to(root):
        return None, "The referenced image path is outside the repository."
    if not path.is_file():
        return None, "The referenced image file is unavailable."
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError, ValueError):
        return None, "The referenced asset is not a readable image."
    return path, None


def _serialize_evidence(result: Any, position: int) -> dict[str, Any]:
    if hasattr(result, "to_dict"):
        record = result.to_dict()
    else:
        record = asdict(result)
    record["reference"] = f"E{position}"
    record["source_pages"] = list(record.get("source_pages") or [])
    return record


def _citation_record(citation: Any) -> dict[str, Any]:
    record = asdict(citation)
    record["source_pages"] = list(record.get("source_pages") or [])
    return record


def _scope_label(
    products: dict[str, dict[str, Any]],
    product_id: str | None,
    document_id: str | None,
    product_by_document: dict[str, str],
) -> str:
    if document_id:
        return products[product_by_document[document_id]]["title"]
    if product_id:
        return products[product_id]["title"]
    return "All products"


def _render_document_library(
    documents: list[ManifestDocument],
    products: dict[str, dict[str, Any]],
    product_by_document: dict[str, str],
) -> None:
    rows: list[dict[str, Any]] = []
    for document in documents:
        product_id = product_by_document.get(document.document_id)
        product = products.get(product_id or "")
        row: dict[str, Any] = {
            "Filename": document.filename,
            "Product": product["title"] if product else "Unmapped",
            "Page count": document.page_count,
            "Status": document.status,
            "Document ID": document.document_id,
        }
        if product and product.get("category") not in (None, "", "unknown"):
            row["Category"] = product["category"]
        if product and product.get("version"):
            row["Version"] = product["version"]
        rows.append(row)
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.info("No inventory documents are available.")


def _render_evaluation_summary(repository_root: Path) -> None:
    path = repository_root / EVALUATION_RESULTS_PATH
    if not path.is_file():
        st.caption("Run `python scripts/evaluate_rag.py --top-k 5` to create a local summary.")
        return
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        st.warning("The saved evaluation summary is unreadable. Rerun the evaluation script.")
        return
    st.caption(f"Most recent saved evaluation: {result.get('generated_at', 'timestamp unavailable')}")
    for mode, metrics in result.get("retrieval", {}).items():
        st.markdown(f"**{mode.title()}**")
        st.json(metrics, expanded=False)


def _asset_maps(records: list[dict[str, Any]]):
    by_id: dict[str, dict[str, Any]] = {}
    by_page: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in records:
        asset_id = record.get("asset_id")
        if isinstance(asset_id, str):
            by_id[asset_id] = record
        document_id, page_number = record.get("document_id"), record.get("page_number")
        if isinstance(document_id, str) and isinstance(page_number, int):
            by_page.setdefault((document_id, page_number), []).append(record)
    return by_id, by_page


def _show_asset(
    repository_root: Path,
    asset: dict[str, Any] | None,
    *,
    caption: str,
    widget_key: str | None = None,
) -> None:
    if asset is None:
        st.info("No page preview is available for this cited page.")
        return
    if asset.get("status") != "ok":
        st.warning("The referenced image is marked unavailable in the asset manifest.")
        return
    path, error = _safe_image_path(repository_root, asset)
    if error or path is None:
        st.warning(error or "The referenced image cannot be displayed.")
        return
    st.image(str(path), caption=caption, width=PAGE_PREVIEW_WIDTH)
    if widget_key:
        st.download_button(
            "Download original image", data=path.read_bytes(), file_name=path.name,
            key=f"download_{widget_key}", on_click="ignore",
        )


def _reset_conversation() -> None:
    """Remove answers and image opt-ins together when a new conversation starts."""
    st.session_state.chat_history = []
    st.session_state.conversation_serial = st.session_state.get("conversation_serial", 0) + 1
    for key in list(st.session_state):
        if key.startswith("preview_") or key in {"last_answer", "last_evidence", "selected_preview"}:
            del st.session_state[key]


def _initialize_conversation() -> None:
    # A rerun after an app upgrade must not revive legacy arbitrary evidence state.
    if st.session_state.get("conversation_schema_version") != CONVERSATION_SCHEMA_VERSION:
        _reset_conversation()
        st.session_state.conversation_schema_version = CONVERSATION_SCHEMA_VERSION
    elif "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def _render_citations(citations: list[dict[str, Any]]) -> None:
    if not citations:
        return
    st.markdown("**Sources cited in the answer**")
    for item in citations:
        pages = ", ".join(str(page) for page in item.get("source_pages", []))
        st.caption(
            f"[{item['reference']}] {item.get('product_title', 'Unknown product')} · "
            f"{item.get('filename', 'Unknown PDF')} · page(s) {pages}"
        )


def _render_evidence(
    repository_root: Path,
    evidence: list[dict[str, Any]],
    assets_by_id: dict[str, dict[str, Any]],
    assets_by_page: dict[tuple[str, int], list[dict[str, Any]]],
    *,
    label: str = "Supporting evidence",
    allow_previews: bool = False,
    preview_limit: int = PAGE_PREVIEW_LIMIT,
    widget_key: str = "evidence",
) -> None:
    if not evidence:
        st.info("No useful passages were retrieved for the selected product/document scope.")
        return
    with st.expander(f"{label} ({len(evidence)} passage(s))", expanded=False):
        show_previews = False
        if allow_previews and preview_limit:
            show_previews = st.checkbox(
                "Show page/image previews", value=False, key=f"preview_{widget_key}",
                help=f"Show up to {preview_limit} thumbnails for this answer. Use the image fullscreen control for a closer view.",
            )
        shown_paths: set[str] = set()
        remaining_previews = preview_limit
        for item in evidence:
            ref = item.get("reference", "Source")
            pages = item.get("source_pages") or list(range(
                int(item.get("page_start", 0)), int(item.get("page_end", 0)) + 1
            ))
            page_label = ", ".join(str(page) for page in pages) or "unknown"
            title = item.get("product_title") or "Unknown product"
            filename = item.get("filename") or "Unknown PDF"
            with st.expander(f"[{ref}] {title} · {filename} · page(s) {page_label}", expanded=False):
                st.caption(
                    f"Document {item.get('document_id')} · section {item.get('section_title') or '—'} · "
                    f"chunk {item.get('chunk_id')}"
                )
                st.text(item.get("text", ""))
                if not show_previews or remaining_previews <= 0:
                    continue
                if item.get("chunk_type") == "image_description":
                    asset_id = item.get("asset_id")
                    asset = assets_by_id.get(asset_id) if asset_id else None
                    if (
                        asset is None
                        or asset.get("document_id") != item.get("document_id")
                        or asset.get("page_number") not in pages
                        or asset.get("asset_path") != item.get("asset_path")
                    ):
                        st.warning("The image reference could not be matched safely to the asset manifest.")
                    else:
                        asset_path = asset["asset_path"]
                        if asset_path not in shown_paths:
                            _show_asset(
                                repository_root, asset,
                                caption=f"{title} · {filename} · page {asset.get('page_number')}",
                                widget_key=f"{widget_key}_{ref}_{asset_id}",
                            )
                            shown_paths.add(asset_path)
                            remaining_previews -= 1
                else:
                    available = 0
                    for page in pages:
                        if remaining_previews <= 0:
                            break
                        options = assets_by_page.get((item.get("document_id"), page), [])
                        preview = next((asset for asset in options if asset.get("asset_type") == "page_preview"), None)
                        if preview is None:
                            continue
                        available += 1
                        asset_path = preview.get("asset_path", "")
                        if asset_path in shown_paths:
                            continue
                        _show_asset(
                            repository_root, preview,
                            caption=f"{title} · {filename} · page {page} preview",
                            widget_key=f"{widget_key}_{ref}_{page}",
                        )
                        shown_paths.add(asset_path)
                        remaining_previews -= 1
                    if not available:
                        st.warning("No page preview is available for this passage; its text source is shown above.")


def _render_entry(
    repository_root: Path,
    entry: dict[str, Any],
    assets_by_id: dict[str, dict[str, Any]],
    assets_by_page: dict[tuple[str, int], list[dict[str, Any]]],
    *,
    allow_previews: bool,
    preview_limit: int,
) -> None:
    with st.chat_message("user"):
        st.write(entry["query"])
        st.caption(f"{entry['answer_mode']} answer · {entry['retrieval_mode']} retrieval · {entry['scope_label']}")
    with st.chat_message("assistant"):
        if entry.get("abstained"):
            st.warning(entry.get("answer", "The supplied documents do not provide enough verified evidence to answer reliably."))
        elif entry.get("answer_mode") == "evidence":
            st.info(entry.get("answer", "Evidence-only mode — no LLM-generated synthesis was performed."))
        elif entry.get("status") in {"ollama_unavailable", "llm_failed"}:
            st.warning(entry.get("answer", "Local Ollama could not produce an answer."))
        else:
            st.markdown(entry.get("answer", ""))
        if entry.get("status") == "answered":
            _render_citations(entry.get("citations", []))
        if entry.get("catalog_scope"):
            st.caption("Catalog-title scope: " + entry["catalog_scope"])
        _render_evidence(
            repository_root, entry.get("display_evidence", entry.get("retrieved_evidence", [])), assets_by_id, assets_by_page,
            allow_previews=allow_previews, preview_limit=preview_limit,
            widget_key=entry["request_key"],
        )
        if entry.get("diagnostic"):
            with st.expander("Technical diagnostic", expanded=False):
                st.code(entry["diagnostic"])


def _health_rows(repository_root: Path) -> list[tuple[str, bool, str]]:
    required = {
        "Product catalog": repository_root / "config" / "product_catalog.json",
        "Document inventory": repository_root / "data" / "processed" / "manifest.jsonl",
        "Chunk dataset": repository_root / "data" / "processed" / "chunks.jsonl",
        "Asset manifest": repository_root / "data" / "processed" / "assets.jsonl",
        "FAISS index": index_paths(repository_root)["index"],
        "Vector mapping": index_paths(repository_root)["mapping"],
        "Index metadata": index_paths(repository_root)["info"],
    }
    return [
        (name, path.is_file(), str(path.relative_to(repository_root)))
        for name, path in required.items()
    ]


def _load_runtime_assets(root: Path) -> tuple[dict[str, Any], dict[str, str], list[ManifestDocument], list[dict[str, Any]]]:
    catalog_path = root / "config" / "product_catalog.json"
    manifest_path = root / "data" / "processed" / "manifest.jsonl"
    asset_path = root / "data" / "processed" / "assets.jsonl"
    products, product_by_document = _cached_catalog(
        str(root), _signature([catalog_path, manifest_path])
    )
    documents = _cached_inventory(str(root), _signature([manifest_path]))
    assets = _cached_assets(str(root), _signature([asset_path]))
    return products, product_by_document, documents, assets


def main() -> None:
    st.set_page_config(page_title="Local Product RAG", page_icon="📚", layout="wide")
    _initialize_conversation()
    st.title("Digital Product Assistant")
    st.caption("Ask questions about your local PDF products. Answers stay grounded in retrieved passages and citations.")

    root = REPOSITORY_ROOT.resolve()
    sidebar = st.sidebar
    sidebar.header("Knowledge base")
    health = _health_rows(root)
    with sidebar.expander("Application health", expanded=False):
        for name, exists, relative in health:
            if exists:
                st.sidebar.caption(f"✓ {name}: `{relative}`")
            else:
                st.sidebar.warning(f"Missing {name}: `{relative}`")

    try:
        products, product_by_document, documents, assets = _load_runtime_assets(root)
    except (OSError, ValueError, KeyError) as exc:
        sidebar.error("The product catalog or document inventory is invalid.")
        st.error("The knowledge base could not be loaded.")
        st.info("Check `config/product_catalog.json` and `data/processed/manifest.jsonl`, then rerun the relevant processing step.")
        st.caption(f"Diagnostic: {exc}")
        return
    if not products:
        st.error("No valid products are configured in the catalog.")
        return

    paths = index_paths(root)
    index_signature = _signature([paths["index"], paths["mapping"], paths["info"], root / "data/processed/chunks.jsonl"])
    index_bundle = None
    index_error: str | None = None
    try:
        index_bundle = _cached_index_bundle(str(root), index_signature)
        sidebar.success(f"Index validated · {index_bundle[2].get('vector_count', 0)} vectors")
    except (OSError, ValueError, RuntimeError) as exc:
        index_error = str(exc)
        sidebar.warning("Semantic and hybrid search need a valid, current FAISS index.")

    product_options: list[str | None] = [None, *sorted(products, key=lambda pid: products[pid]["title"].casefold())]
    if st.session_state.get("product_filter") not in product_options:
        st.session_state.product_filter = None
    product_id = sidebar.selectbox(
        "Product",
        options=product_options,
        format_func=lambda value: "All products" if value is None else f"{products[value]['title']} ({value})",
        key="product_filter",
    )
    allowed_document_ids = [
        document_id for document_id in product_by_document
        if product_id is None or product_by_document[document_id] == product_id
    ]
    documents_by_id = {document.document_id: document for document in documents}
    document_options: list[str | None] = [None, *sorted(
        allowed_document_ids,
        key=lambda document_id: documents_by_id.get(document_id).filename.casefold()
        if document_id in documents_by_id else document_id,
    )]
    if st.session_state.get("document_filter") not in document_options:
        st.session_state.document_filter = None
    document_id = sidebar.selectbox(
        "Document",
        options=document_options,
        format_func=lambda value: (
            "All documents in scope" if value is None
            else f"{documents_by_id[value].filename} ({value})" if value in documents_by_id
            else value
        ),
        key="document_filter",
        disabled=not allowed_document_ids,
    )
    selected_product_title = _scope_label(products, product_id, document_id, product_by_document)
    with sidebar.expander("Document library", expanded=False):
        _render_document_library(documents, products, product_by_document)

    sidebar.divider()
    sidebar.subheader("Retrieval settings")
    retrieval_mode = sidebar.selectbox(
        "Retrieval method", options=["hybrid", "semantic", "keyword"], key="retrieval_mode"
    )
    answer_mode = sidebar.selectbox("Answer mode", options=["evidence", "ollama"], key="answer_mode")
    top_k = sidebar.slider("Passages to retrieve", min_value=3, max_value=10, value=5, key="top_k")
    chunk_choice = sidebar.selectbox(
        "Evidence type", options=["auto", "all", "text", "image_description"], key="chunk_type",
        format_func=lambda value: {
            "auto": "Auto (recommended)", "all": "All evidence", "text": "Text only",
            "image_description": "Visual descriptions only",
        }[value],
    )
    chunk_type = chunk_choice if chunk_choice in {"text", "image_description"} else None
    preview_limit = sidebar.slider(
        "Maximum previews per answer", min_value=0, max_value=4, value=PAGE_PREVIEW_LIMIT,
        key="evidence_preview_limit", help="Previews appear only when you enable them in Supporting evidence.",
    )
    if answer_mode == "ollama":
        sidebar.caption(f"Model: {os.environ.get('RAG_GENERATION_MODEL') or 'not configured'}")
        sidebar.caption(f"Local host: {os.environ.get('OLLAMA_HOST') or 'http://localhost:11434'}")
    if retrieval_mode in {"semantic", "hybrid"} and index_error:
        st.warning(index_error)

    assets_by_id, assets_by_page = _asset_maps(assets)
    st.markdown(f"**Current scope:** {selected_product_title}")
    st.caption("Conversation history is kept in this browser session and is not saved to disk.")

    left, right = st.columns([1, 5])
    with left:
        st.button("New conversation", key="clear_history", on_click=_reset_conversation)
    with right:
        with st.expander("Latest retrieval evaluation", expanded=False):
            _render_evaluation_summary(root)

    question = st.chat_input("Ask about a product, materials, instructions, or a diagram…")
    history = [entry for entry in st.session_state.chat_history if entry.get("submitted") and entry.get("request_key")]
    if not history and question is None:
        st.info("Choose a product or search all products, then ask a question to begin.")
        st.caption('Example: “Which product contains a penguin sweater, and what are its construction step names? Names only.”')
    for index, entry in enumerate(history):
        _render_entry(
            root, entry, assets_by_id, assets_by_page,
            allow_previews=question is None and index == len(history) - 1,
            preview_limit=preview_limit,
        )
    if question is None:
        return
    query = question.strip()
    if not query:
        st.warning("Enter a question before sending.")
        return

    filter_product_id = product_id
    filter_document_id = document_id
    filter_chunk_type = chunk_type
    with st.chat_message("user"):
        st.write(query)
        st.caption(f"{answer_mode} answer · {retrieval_mode} retrieval · {selected_product_title}")
    with st.chat_message("assistant"):
        try:
            keyword_index = None
            encoder = None
            if retrieval_mode in {"keyword", "hybrid"}:
                keyword_index = _cached_keyword_index(
                    str(root), _signature([root / "data" / "processed" / "chunks.jsonl"])
                )
            if retrieval_mode in {"semantic", "hybrid"}:
                if index_bundle is None:
                    st.error("Semantic retrieval is unavailable because the FAISS index failed validation.")
                    return
                encoder = _cached_embedding_model(index_bundle[2]["model_name"])
            with st.spinner("Searching the selected documents…"):
                results, response, anchors, focus_ids = query_knowledge_base(
                    root,
                    query,
                    products=products,
                    product_by_document=product_by_document,
                    retrieval_mode=retrieval_mode,
                    answer_mode=answer_mode,
                    top_k=top_k,
                    product_id=filter_product_id,
                    document_id=filter_document_id,
                    chunk_type=filter_chunk_type,
                    evidence_selection=chunk_choice,
                    keyword_index=keyword_index,
                    index_bundle=index_bundle,
                    encoder=encoder,
                )
            evidence_all = [_serialize_evidence(result, index) for index, result in enumerate(results, start=1)]
            citations = [_citation_record(citation) for citation in response.citations]
            display_evidence = evidence_all
            if response.status == "answered":
                cited_ids = {item["chunk_id"] for item in citations}
                display_evidence = citations + [item for item in evidence_all if item["chunk_id"] not in cited_ids]
            if answer_mode == "evidence":
                answer_text = "Evidence-only mode — no LLM-generated synthesis was performed."
                if response.abstained:
                    answer_text += f"\n\n{response.abstention_reason}"
            elif response.abstained:
                answer_text = response.abstention_reason or response.answer or "The supplied documents do not provide enough verified evidence to answer reliably."
            elif response.status == "answered":
                answer_text = response.answer or ""
            elif response.status == "ollama_unavailable":
                answer_text = "Local Ollama is unavailable or the configured model is missing. Start Ollama or choose Evidence-only mode."
            else:
                answer_text = "Local answer generation failed. No generated answer was accepted; the retrieved passages are shown below."

            entry = {
                "submitted": True,
                "request_key": f"{st.session_state.conversation_serial}_{len(st.session_state.chat_history)}",
                "query": query,
                "answer_mode": answer_mode,
                "retrieval_mode": retrieval_mode,
                "top_k": top_k,
                "product_id": filter_product_id,
                "document_id": filter_document_id,
                "chunk_type": filter_chunk_type,
                "evidence_selection": chunk_choice,
                "scope_label": selected_product_title,
                "answer": answer_text,
                "status": response.status,
                "abstained": response.abstained,
                "citations": citations,
                "retrieved_evidence": evidence_all,
                "display_evidence": display_evidence,
                "diagnostic": response.error,
                "catalog_scope": ", ".join(products[pid]["title"] for pid in sorted(focus_ids)) if anchors and focus_ids else None,
            }
            st.session_state.chat_history.append(entry)
            st.rerun()
        except (ValueError, OSError, RuntimeError, KeyError) as exc:
            st.error("The request could not be completed. Check the selected filters and local search artifacts.")
            st.caption(f"Diagnostic: {exc}")


if __name__ == "__main__":
    main()
