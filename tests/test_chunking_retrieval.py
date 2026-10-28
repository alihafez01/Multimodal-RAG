"""Unit tests for editable product metadata, source-aware chunks, and FAISS search."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from digital_product_rag.catalog import CatalogError, prepare_product_catalog  # noqa: E402
from digital_product_rag.chunking.chunker import (  # noqa: E402
    ChunkingError,
    build_chunks,
    load_page_input,
    validate_chunk_settings,
)
from digital_product_rag.ingestion.common import read_manifest  # noqa: E402
from digital_product_rag.retrieval.embeddings import EmbeddingError, encode_texts  # noqa: E402
from digital_product_rag.retrieval.vector_store import (  # noqa: E402
    VectorStoreError,
    build_vector_index,
    index_paths,
    search,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def make_repo(root: Path, texts_by_doc: list[list[str]]) -> tuple[list[str], list[dict], list[dict]]:
    dataset = root / "Dataset"
    processed = root / "data" / "processed"
    dataset.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    manifest, pages, docs = [], [], []
    for doc_index, page_texts in enumerate(texts_by_doc, start=1):
        relative = f"sample-{doc_index}.pdf"
        pdf = dataset / relative
        pdf.write_bytes(b"%PDF-1.4 synthetic fixture\n")
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, relative))
        docs.append(doc_id)
        manifest.append({
            "document_id": doc_id,
            "filename": relative,
            "relative_path": relative,
            "file_size_bytes": pdf.stat().st_size,
            "page_count": len(page_texts),
            "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
            "status": "valid",
            "product_title": f"Sample {doc_index}",
            "error_message": None,
        })
        for page_number, text in enumerate(page_texts, start=1):
            pages.append({
                "document_id": doc_id,
                "filename": relative,
                "relative_path": relative,
                "page_number": page_number,
                "text": text,
                "text_char_count": len(text),
                "extraction_method": "native_text",
                "ocr_required": False,
                "status": "ok",
                "error_message": None,
            })
    write_jsonl(processed / "manifest.jsonl", manifest)
    write_jsonl(processed / "pages.jsonl", pages)
    enriched = []
    for row in pages:
        enriched.append({
            **row,
            "native_text": row["text"],
            "native_text_char_count": len(row["text"]),
            "ocr_text": "",
            "ocr_text_char_count": 0,
            "ocr_status": "not_needed",
            "ocr_language": None,
            "ocr_dpi": None,
            "ocr_error": None,
        })
    write_jsonl(processed / "pages_enriched.jsonl", enriched)
    return docs, manifest, pages


class FakeEncoder:
    def __init__(self, fail: bool = False):
        self.fail = fail

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts, **kwargs):
        if self.fail:
            raise RuntimeError("synthetic encoder failure")
        output = []
        for text in texts:
            lowered = text.casefold()
            vector = np.zeros(3, dtype=np.float32)
            if any(word in lowered for word in ("material", "yarn", "cotton", "hook")):
                vector[0] += 1
            if any(word in lowered for word in ("finish", "sew", "fasten")):
                vector[1] += 1
            if any(word in lowered for word in ("diagram", "snowman", "image")):
                vector[2] += 1
            if not vector.any():
                vector[0] = 1
            output.append(vector)
        return np.asarray(output, dtype=np.float32)


def vector_chunk(chunk_id: str, text: str, *, doc="doc-1", product="prod-1", kind="text") -> dict:
    page = 1 if doc == "doc-1" else 2
    image = kind == "image_description"
    return {
        "chunk_id": chunk_id,
        "document_id": doc,
        "product_id": product,
        "product_title": product,
        "product_category": "unknown",
        "product_version": None,
        "filename": f"{doc}.pdf",
        "relative_path": f"{doc}.pdf",
        "page_start": page,
        "page_end": page,
        "source_pages": [page],
        "section_title": None,
        "chunk_type": kind,
        "text": text,
        "asset_id": f"asset-{chunk_id}" if image else None,
        "asset_path": "data/assets/images/sample.png" if image else None,
        "metadata_status": "needs_review",
    }


class ProductCatalogAndChunkTests(unittest.TestCase):
    def test_catalog_is_deterministic_and_preserves_existing_metadata(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            docs, _, _ = make_repo(root, [["Materials\nUse cotton yarn."]])
            manifest_docs = read_manifest(root / "data/processed/manifest.jsonl", root / "Dataset")
            first = prepare_product_catalog(root, manifest_docs)
            first_id = first.products[0].product_id
            self.assertEqual(first.products[0].category, "unknown")
            self.assertIsNone(first.products[0].version)
            raw = json.loads(first.path.read_text(encoding="utf-8"))
            raw["products"][0].update({"title": "My edited name", "category": "craft", "version": "v2"})
            first.path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
            second = prepare_product_catalog(root, manifest_docs)
            self.assertEqual(second.products[0].product_id, first_id)
            self.assertEqual(second.products[0].title, "My edited name")
            self.assertEqual(second.products[0].category, "craft")
            self.assertEqual(second.products[0].version, "v2")
            self.assertEqual(second.product_by_document[docs[0]].metadata_status, "needs_review")

    def test_catalog_allows_multiple_documents_per_product_and_rejects_ambiguity(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            docs, _, _ = make_repo(root, [["A source text."], ["Another source text."]])
            manifest_docs = read_manifest(root / "data/processed/manifest.jsonl", root / "Dataset")
            catalog = prepare_product_catalog(root, manifest_docs)
            shared = {
                "product_id": "shared-product", "title": "Shared", "category": "unknown",
                "version": None, "document_ids": docs, "metadata_status": "needs_review",
            }
            catalog.path.write_text(json.dumps({"schema_version": 1, "products": [shared]}), encoding="utf-8")
            result = prepare_product_catalog(root, manifest_docs)
            self.assertEqual(result.product_by_document[docs[0]].product_id, "shared-product")
            self.assertEqual(result.product_by_document[docs[1]].product_id, "shared-product")
            duplicate = {**shared, "product_id": "another"}
            catalog.path.write_text(json.dumps({"schema_version": 1, "products": [shared, duplicate]}), encoding="utf-8")
            with self.assertRaisesRegex(CatalogError, "multiple products"):
                prepare_product_catalog(root, manifest_docs)

    def test_catalog_reports_duplicate_ids_and_unknown_document_references(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            _, _, _ = make_repo(root, [["A source text."]])
            docs = read_manifest(root / "data/processed/manifest.jsonl", root / "Dataset")
            path = root / "config/product_catalog.json"
            row = {
                "product_id": "same", "title": "Test", "category": "unknown", "version": None,
                "document_ids": ["missing-doc"],
            }
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"schema_version": 1, "products": [row, row]}), encoding="utf-8")
            with self.assertRaisesRegex(CatalogError, "Duplicate product_id"):
                prepare_product_catalog(root, docs)

    def test_heading_page_references_visual_validation_and_deterministic_jsonl(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            docs, manifest, page_rows = make_repo(root, [
                ["Materials\nUse cotton yarn and a hook.\n\n1. Attach the loop.", "Continue the materials instructions."],
                [""],
            ])
            enriched = []
            for row in page_rows:
                enriched.append({**row, "native_text": row["text"], "ocr_status": "not_needed"})
            native = "Materials\nUse cotton yarn and a hook.\n\n1. Attach the loop."
            enriched[0].update({
                "native_text": native,
                "text": native + " BAD OCR SHOULD NOT BE USED",
                "text_char_count": len(native + " BAD OCR SHOULD NOT BE USED"),
                "ocr_status": "failed",
            })
            write_jsonl(root / "data/processed/pages_enriched.jsonl", enriched)
            asset_dir = root / "data/assets/images"
            asset_dir.mkdir(parents=True)
            (asset_dir / "sample.png").write_bytes(b"fixture image bytes")
            asset = {
                "asset_id": "asset-one", "document_id": docs[0], "filename": manifest[0]["filename"],
                "relative_path": manifest[0]["relative_path"], "page_number": 1,
                "asset_type": "embedded_image", "asset_path": "data/assets/images/sample.png",
                "status": "ok",
            }
            write_jsonl(root / "data/processed/assets.jsonl", [asset])
            visual = [
                {**asset, "description": "A clear diagram of a snowman.", "description_status": "generated",
                 "description_source": "vision_model", "description_model": "test-vl"},
                {**asset, "asset_id": "not-selected", "description": None,
                 "description_status": "not_selected", "description_source": "none"},
                {**asset, "asset_id": "missing-asset", "description": "caption", "description_status": "generated",
                 "description_source": "vision_model", "description_model": "test-vl"},
            ]
            write_jsonl(root / "data/processed/visual_metadata.jsonl", visual)
            first = build_chunks(root, max_chars=180, overlap_chars=20, min_chars=1)
            first_bytes = (root / "data/processed/chunks.jsonl").read_bytes()
            second = build_chunks(root, max_chars=180, overlap_chars=20, min_chars=1)
            self.assertEqual(first_bytes, (root / "data/processed/chunks.jsonl").read_bytes())
            self.assertEqual(first.chunks, second.chunks)
            self.assertGreater(first.blank_pages, 0)
            self.assertEqual(first.invalid_assets, 1)
            text_chunks = [row for row in first.chunks if row["chunk_type"] == "text"]
            self.assertTrue(all(row["product_id"] and row["document_id"] in docs for row in text_chunks))
            first_doc_text = "\n".join(row["text"] for row in text_chunks if row["document_id"] == docs[0])
            self.assertIn("Use cotton yarn", first_doc_text)
            self.assertIn("1. Attach the loop.", first_doc_text)
            self.assertNotIn("BAD OCR SHOULD NOT BE USED", first_doc_text)
            materials = next(row for row in text_chunks if row["section_title"] == "Materials")
            self.assertIn("Use cotton yarn", materials["text"])
            self.assertEqual(materials["page_start"], min(materials["source_pages"]))
            self.assertEqual(materials["page_end"], max(materials["source_pages"]))
            image_chunks = [row for row in first.chunks if row["chunk_type"] == "image_description"]
            self.assertEqual(len(image_chunks), 1)
            self.assertEqual(image_chunks[0]["asset_id"], "asset-one")
            self.assertEqual(image_chunks[0]["source_pages"], [1])
            self.assertTrue((root / image_chunks[0]["asset_path"]).is_file())

    def test_long_sections_split_with_overlap_and_settings_are_validated(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            long_text = "Materials\n" + "This sentence describes the material list. " * 80
            make_repo(root, [[long_text]])
            result = build_chunks(root, max_chars=180, overlap_chars=25, min_chars=1)
            text_chunks = [row for row in result.chunks if row["chunk_type"] == "text"]
            self.assertGreater(len(text_chunks), 1)
            self.assertTrue(all(len(row["text"]) <= 180 for row in text_chunks))
            self.assertTrue(all(row["section_title"] == "Materials" for row in text_chunks))
            self.assertTrue(any(
                previous["text"][-20:] in current["text"]
                for previous, current in zip(text_chunks, text_chunks[1:])
            ))
        with self.assertRaises(ValueError):
            validate_chunk_settings(100, 100, 1)
        with self.assertRaises(ValueError):
            validate_chunk_settings(63, 2, 1)

    def test_invalid_enriched_pages_fall_back_to_valid_native_pages(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            make_repo(root, [["A page with useful native words."]])
            (root / "data/processed/pages_enriched.jsonl").write_text("not-json\n", encoding="utf-8")
            docs = read_manifest(root / "data/processed/manifest.jsonl", root / "Dataset")
            loaded = load_page_input(root, docs)
            self.assertEqual(loaded.source_name, "pages.jsonl")
            self.assertIn("Malformed enriched pages JSON", loaded.fallback_reason)


class VectorSearchTests(unittest.TestCase):
    def temporary_directory(self):
        return tempfile.TemporaryDirectory(dir=ROOT)

    def _make_index_source(self, root: Path) -> Path:
        chunks_path = root / "data/processed/chunks.jsonl"
        write_jsonl(chunks_path, [
            vector_chunk("chunk-materials", "Materials include cotton yarn and a hook."),
            vector_chunk("chunk-finishing", "Finishing instructions say sew and fasten ends."),
            vector_chunk("chunk-image", "A diagram of a snowman ornament.", doc="doc-2", product="prod-2", kind="image_description"),
        ])
        return chunks_path

    def test_embeddings_are_float32_finite_normalized_and_batchable(self):
        matrix = encode_texts(FakeEncoder(), ["cotton yarn", "finish sewing"], batch_size=1, show_progress=False)
        self.assertEqual(matrix.dtype, np.float32)
        self.assertEqual(matrix.shape, (2, 3))
        self.assertTrue(np.isfinite(matrix).all())
        np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), [1.0, 1.0])
        with self.assertRaises(EmbeddingError):
            encode_texts(FakeEncoder(), [""], show_progress=False)

    def test_index_build_search_filters_and_asset_sources(self):
        with self.temporary_directory() as tmp:
            root = Path(tmp)
            self._make_index_source(root)
            with patch.dict(os.environ, {"RAG_EMBEDDING_MODEL": "test-model"}):
                built = build_vector_index(root, encoder=FakeEncoder(), batch_size=2)
                self.assertEqual(built["vector_count"], 3)
                paths = index_paths(root)
                mapping = [json.loads(line) for line in paths["mapping"].read_text(encoding="utf-8").splitlines()]
                self.assertEqual([row["vector_id"] for row in mapping], [0, 1, 2])
                self.assertEqual(len({row["chunk_id"] for row in mapping}), 3)
                results = search(root, "cotton yarn materials", top_k=5, encoder=FakeEncoder())
                self.assertLessEqual(len(results), 3)
                self.assertEqual(results[0].chunk_id, "chunk-materials")
                filtered = search(root, "diagram snowman", top_k=1, chunk_type="image_description", encoder=FakeEncoder())
                self.assertEqual(filtered[0].chunk_id, "chunk-image")
                self.assertEqual(filtered[0].asset_id, "asset-chunk-image")
                self.assertEqual(filtered[0].source_pages, (2,))
                document_filtered = search(root, "materials", top_k=5, document_id="doc-2", encoder=FakeEncoder())
                self.assertTrue(all(row.document_id == "doc-2" for row in document_filtered))
                self.assertEqual(len(document_filtered), 1)
                product_filtered = search(root, "finishing", top_k=5, product_id="prod-1", encoder=FakeEncoder())
                self.assertTrue(all(row.product_id == "prod-1" for row in product_filtered))

    def test_empty_query_model_mismatch_and_stale_chunks_are_clear(self):
        with self.temporary_directory() as tmp:
            root = Path(tmp)
            chunk_path = self._make_index_source(root)
            with patch.dict(os.environ, {"RAG_EMBEDDING_MODEL": "test-model"}):
                build_vector_index(root, encoder=FakeEncoder())
                with self.assertRaisesRegex(ValueError, "non-empty"):
                    search(root, "  ", encoder=FakeEncoder())
                with self.assertRaisesRegex(VectorStoreError, "does not match"):
                    search(root, "materials", model_name="other-model", encoder=FakeEncoder())
                chunk_path.write_text(chunk_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
                with self.assertRaisesRegex(VectorStoreError, "changed after indexing"):
                    search(root, "materials", encoder=FakeEncoder())

    def test_failed_rebuild_preserves_previous_index_files(self):
        with self.temporary_directory() as tmp:
            root = Path(tmp)
            self._make_index_source(root)
            with patch.dict(os.environ, {"RAG_EMBEDDING_MODEL": "test-model"}):
                build_vector_index(root, encoder=FakeEncoder())
                paths = index_paths(root)
                previous = {key: paths[key].read_bytes() for key in ("index", "mapping", "info")}
                with self.assertRaisesRegex(VectorStoreError, "Embedding batch"):
                    build_vector_index(root, encoder=FakeEncoder(fail=True))
                self.assertEqual(previous, {key: paths[key].read_bytes() for key in previous})

    def test_empty_chunk_dataset_is_rejected(self):
        with self.temporary_directory() as tmp:
            root = Path(tmp)
            write_jsonl(root / "data/processed/chunks.jsonl", [])
            with self.assertRaisesRegex(VectorStoreError, "No valid non-empty chunks"):
                build_vector_index(root, encoder=FakeEncoder())

    def test_model_cache_cannot_be_placed_inside_the_repository(self):
        with self.temporary_directory() as tmp:
            root = Path(tmp)
            self._make_index_source(root)
            with patch.dict(os.environ, {"HF_HOME": str(root / "Dataset" / "models")}):
                with self.assertRaisesRegex(VectorStoreError, "inside the repository"):
                    build_vector_index(root, encoder=FakeEncoder(), model_name="test-model")


if __name__ == "__main__":
    unittest.main()
