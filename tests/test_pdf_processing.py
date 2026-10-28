"""Unit tests for text extraction, image extraction, and page rendering."""

from __future__ import annotations

import contextlib
import io
import json
import struct
import sys
import tempfile
import unittest
import uuid
import zlib
from pathlib import Path
from unittest.mock import patch

import pymupdf


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from digital_product_rag.ingestion.common import (  # noqa: E402
    ASSET_FIELDS,
    ManifestDocument,
    ManifestError,
    read_manifest,
)
from digital_product_rag.ingestion import image_extractor, page_renderer  # noqa: E402
from digital_product_rag.ingestion.image_extractor import extract_images  # noqa: E402
from digital_product_rag.ingestion.page_renderer import render_pages  # noqa: E402
from digital_product_rag.ingestion.text_extractor import extract_pages  # noqa: E402
from scripts.process_pdfs import run_asset_pipeline, run_pipeline  # noqa: E402


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    payload = chunk_type + data
    return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)


def solid_png() -> bytes:
    """Create a small RGB PNG using only the standard library."""
    width = height = 2
    pixels = b"\x00" + bytes((220, 40, 70)) * width
    raw = pixels * height
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(raw))
        + _png_chunk(b"IEND", b"")
    )


def create_pdf(path: Path, *, with_images: bool = True) -> None:
    """Create a small two-page PDF with known text and optional repeated images."""
    document = pymupdf.open()
    first_page = document.new_page(width=300, height=200)
    first_page.insert_text((20, 25), "Known synthetic page one text for extraction.", fontsize=11)
    image_bytes = solid_png()
    if with_images:
        first_page.insert_image(pymupdf.Rect(20, 45, 70, 95), stream=image_bytes)
        first_page.insert_image(pymupdf.Rect(90, 45, 140, 95), stream=image_bytes)

    second_page = document.new_page(width=300, height=200)
    if with_images:
        second_page.insert_image(pymupdf.Rect(40, 50, 100, 110), stream=image_bytes)
    document.save(path)
    document.close()


def manifest_document(path: Path, dataset_dir: Path, *, page_count: int = 2) -> ManifestDocument:
    relative_path = path.relative_to(dataset_dir).as_posix()
    return ManifestDocument(
        document_id=str(uuid.uuid5(uuid.NAMESPACE_URL, relative_path)),
        filename=path.name,
        relative_path=relative_path,
        status="valid",
        page_count=page_count,
        source_path=path,
    )


def write_test_manifest(dataset_dir: Path, processed_dir: Path, pdf: Path, pages: int) -> None:
    document_id = str(uuid.uuid5(uuid.NAMESPACE_URL, pdf.relative_to(dataset_dir).as_posix()))
    row = {
        "document_id": document_id,
        "filename": pdf.name,
        "relative_path": pdf.relative_to(dataset_dir).as_posix(),
        "file_size_bytes": pdf.stat().st_size,
        "page_count": pages,
        "sha256": "test-hash",
        "status": "valid",
        "product_title": pdf.stem,
        "error_message": None,
    }
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "manifest.jsonl").write_text(
        json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8"
    )


class PdfProcessingTests(unittest.TestCase):
    def temporary_directory(self):
        return tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT)

    def test_text_extraction_preserves_page_order_and_flags_text_poor_pages(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            dataset = root / "Dataset"
            dataset.mkdir()
            pdf_path = dataset / "two-pages.pdf"
            create_pdf(pdf_path, with_images=False)
            document = manifest_document(pdf_path, dataset)

            result = extract_pages([document], root / "pages.jsonl", pymupdf)

            self.assertEqual(result.expected_page_count, 2)
            self.assertEqual([record["page_number"] for record in result.records], [1, 2])
            self.assertEqual(
                result.records[0]["text"], "Known synthetic page one text for extraction."
            )
            self.assertEqual(result.records[0]["text_char_count"], len(result.records[0]["text"]))
            self.assertFalse(result.records[0]["ocr_required"])
            self.assertEqual(result.records[1]["text"], "")
            self.assertTrue(result.records[1]["ocr_required"])
            self.assertEqual(result.records[0]["extraction_method"], "native_text")
            saved_records = [
                json.loads(line)
                for line in (root / "pages.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(saved_records), 2)
            self.assertTrue(
                all(record["text_char_count"] == len(record["text"]) for record in saved_records)
            )

    def test_page_extraction_error_is_recorded_and_later_pages_continue(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            dataset = root / "Dataset"
            dataset.mkdir()
            pdf_path = dataset / "two-pages.pdf"
            create_pdf(pdf_path, with_images=False)
            document = manifest_document(pdf_path, dataset)
            original_get_text = pymupdf.Page.get_text

            def fail_first_page(page, *args, **kwargs):
                if page.number == 0:
                    raise RuntimeError("synthetic extraction failure")
                return original_get_text(page, *args, **kwargs)

            with patch.object(pymupdf.Page, "get_text", new=fail_first_page):
                result = extract_pages([document], root / "pages.jsonl", pymupdf)

            self.assertEqual([record["status"] for record in result.records], ["error", "ok"])
            self.assertEqual(result.records[0]["text"], "")
            self.assertEqual(result.records[0]["extraction_method"], "failed")
            self.assertTrue(result.records[0]["ocr_required"])
            self.assertIsNotNone(result.records[0]["error_message"])

    def test_unopenable_document_is_reported_and_other_documents_continue(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            dataset = root / "Dataset"
            dataset.mkdir()
            malformed = dataset / "malformed.pdf"
            malformed.write_bytes(b"not a PDF")
            valid = dataset / "valid.pdf"
            create_pdf(valid, with_images=False)
            documents = [
                manifest_document(malformed, dataset),
                manifest_document(valid, dataset),
            ]

            result = extract_pages(documents, root / "pages.jsonl", pymupdf)

            self.assertEqual(len(result.records), 2)
            self.assertEqual(result.records[0]["document_id"], documents[1].document_id)
            self.assertEqual(len(result.file_errors), 1)
            self.assertIn("malformed.pdf", result.file_errors[0])

    def test_manifest_rejects_malformed_json_and_path_traversal(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            dataset = root / "Dataset"
            dataset.mkdir()
            manifest = root / "manifest.jsonl"
            manifest.write_text("not json\n", encoding="utf-8")
            with self.assertRaisesRegex(ManifestError, "Malformed inventory JSON"):
                read_manifest(manifest, dataset)

            row = {
                "document_id": str(uuid.uuid4()),
                "filename": "outside.pdf",
                "relative_path": "../outside.pdf",
                "status": "valid",
                "page_count": 1,
            }
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ManifestError, "must not escape"):
                read_manifest(manifest, dataset)

            row["relative_path"] = "inside.pdf"
            row["filename"] = "inside.pdf"
            del row["status"]
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ManifestError, "missing or invalid required fields"):
                read_manifest(manifest, dataset)

    def test_image_occurrences_are_extracted_and_deduplicated_per_document(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            dataset = root / "Dataset"
            dataset.mkdir()
            pdf_path = dataset / "images.pdf"
            create_pdf(pdf_path)
            document = manifest_document(pdf_path, dataset)

            result = extract_images([document], root, root / "data" / "assets", pymupdf=pymupdf)

            self.assertEqual(len(result.records), 3)
            self.assertTrue(all(record["status"] == "ok" for record in result.records))
            self.assertEqual([record["page_number"] for record in result.records], [1, 1, 2])
            self.assertEqual(len({record["asset_id"] for record in result.records}), 3)
            self.assertEqual(len({record["asset_path"] for record in result.records}), 1)
            self.assertEqual(
                [record["image_index"] for record in result.records if record["page_number"] == 1],
                [1, 2],
            )
            self.assertEqual(
                [record["image_index"] for record in result.records if record["page_number"] == 2],
                [1],
            )
            for record in result.records:
                self.assertEqual(record["asset_type"], "embedded_image")
                self.assertTrue((root / record["asset_path"]).is_file())
                self.assertTrue(Path(record["asset_path"]).as_posix().startswith("data/assets/images/"))

    def test_missing_image_is_regenerated_and_existing_image_is_reused(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            dataset = root / "Dataset"
            dataset.mkdir()
            pdf_path = dataset / "images.pdf"
            create_pdf(pdf_path)
            document = manifest_document(pdf_path, dataset)
            assets_root = root / "data" / "assets"

            initial = extract_images([document], root, assets_root, pymupdf=pymupdf)
            missing_path = root / initial.records[0]["asset_path"]
            asset_ids = [record["asset_id"] for record in initial.records]
            missing_path.unlink()
            repaired = extract_images([document], root, assets_root, pymupdf=pymupdf)

            self.assertTrue(missing_path.is_file())
            self.assertEqual(asset_ids, [record["asset_id"] for record in repaired.records])
            self.assertTrue(all(record["status"] == "ok" for record in repaired.records))

            with patch.object(image_extractor, "atomic_write_bytes", side_effect=AssertionError):
                reused = extract_images([document], root, assets_root, pymupdf=pymupdf)
            self.assertEqual(reused.reused_files, 1)
            self.assertTrue(all(record["status"] == "ok" for record in reused.records))

    def test_failed_direct_image_extraction_falls_back_to_page_crop(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            dataset = root / "Dataset"
            dataset.mkdir()
            pdf_path = dataset / "fallback.pdf"
            create_pdf(pdf_path)
            document = manifest_document(pdf_path, dataset)

            with patch.object(pymupdf.Document, "extract_image", side_effect=RuntimeError("test")):
                result = extract_images(
                    [document], root, root / "data" / "assets", dpi=72, pymupdf=pymupdf
                )

            self.assertEqual(len(result.records), 3)
            self.assertTrue(all(record["status"] == "ok" for record in result.records))
            self.assertTrue(all(record["asset_type"] == "image_crop_fallback" for record in result.records))
            self.assertTrue(all(record["extraction_method"] == "page_crop_fallback" for record in result.records))

    def test_image_page_error_does_not_stop_later_page_assets(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            dataset = root / "Dataset"
            dataset.mkdir()
            pdf_path = dataset / "page-error.pdf"
            create_pdf(pdf_path)
            document = manifest_document(pdf_path, dataset)
            original_get_image_info = pymupdf.Page.get_image_info

            def fail_first_page(page, *args, **kwargs):
                if page.number == 0:
                    raise RuntimeError("synthetic image metadata failure")
                return original_get_image_info(page, *args, **kwargs)

            with patch.object(pymupdf.Page, "get_image_info", new=fail_first_page):
                result = extract_images(
                    [document], root, root / "data" / "assets", pymupdf=pymupdf
                )

            self.assertEqual(len(result.records), 2)
            self.assertEqual(result.records[0]["status"], "error")
            self.assertIsNone(result.records[0]["image_index"])
            self.assertEqual(result.records[1]["status"], "ok")
            self.assertEqual(result.records[1]["page_number"], 2)

    def test_page_renderer_writes_one_png_record_per_page(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            dataset = root / "Dataset"
            dataset.mkdir()
            pdf_path = dataset / "render.pdf"
            create_pdf(pdf_path, with_images=False)
            document = manifest_document(pdf_path, dataset)

            result = render_pages(
                [document], root, root / "data" / "assets", dpi=72, pymupdf=pymupdf
            )

            self.assertEqual(len(result.records), 2)
            self.assertTrue(all(record["status"] == "ok" for record in result.records))
            self.assertEqual([record["page_number"] for record in result.records], [1, 2])
            for record in result.records:
                path = root / record["asset_path"]
                self.assertTrue(path.is_file())
                self.assertTrue(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
                self.assertEqual(record["asset_type"], "page_preview")
                self.assertEqual(record["image_format"], "png")

    def test_missing_preview_is_regenerated_and_valid_previews_are_reused(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            dataset = root / "Dataset"
            dataset.mkdir()
            pdf_path = dataset / "preview.pdf"
            create_pdf(pdf_path, with_images=False)
            document = manifest_document(pdf_path, dataset)
            assets_root = root / "data" / "assets"

            initial = render_pages([document], root, assets_root, dpi=72, pymupdf=pymupdf)
            missing_path = root / initial.records[0]["asset_path"]
            asset_ids = [record["asset_id"] for record in initial.records]
            missing_path.unlink()
            repaired = render_pages([document], root, assets_root, dpi=72, pymupdf=pymupdf)

            self.assertTrue(missing_path.is_file())
            self.assertEqual(asset_ids, [record["asset_id"] for record in repaired.records])
            self.assertEqual(repaired.reused_files, 1)
            self.assertTrue(all(record["status"] == "ok" for record in repaired.records))

            with (
                patch.object(page_renderer, "atomic_write_bytes", side_effect=AssertionError),
                patch.object(pymupdf.Page, "get_pixmap", side_effect=AssertionError),
            ):
                reused = render_pages([document], root, assets_root, dpi=72, pymupdf=pymupdf)
            self.assertEqual(reused.reused_files, 2)

    def test_render_error_is_recorded_and_later_pages_continue(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            dataset = root / "Dataset"
            dataset.mkdir()
            pdf_path = dataset / "render-error.pdf"
            create_pdf(pdf_path, with_images=False)
            document = manifest_document(pdf_path, dataset)
            original_get_pixmap = pymupdf.Page.get_pixmap

            def fail_first_page(page, *args, **kwargs):
                if page.number == 0:
                    raise RuntimeError("synthetic rendering failure")
                return original_get_pixmap(page, *args, **kwargs)

            with patch.object(pymupdf.Page, "get_pixmap", new=fail_first_page):
                result = render_pages(
                    [document], root, root / "data" / "assets", dpi=72, pymupdf=pymupdf
                )

            self.assertEqual([record["status"] for record in result.records], ["error", "ok"])
            self.assertIsNone(result.records[0]["asset_path"])
            self.assertTrue((root / result.records[1]["asset_path"]).is_file())

    def test_pipeline_combines_assets_and_is_deterministic_on_rerun(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            dataset = root / "Dataset"
            dataset.mkdir()
            pdf_path = dataset / "pipeline.pdf"
            create_pdf(pdf_path)
            write_test_manifest(dataset, root / "data" / "processed", pdf_path, pages=2)

            with contextlib.redirect_stdout(io.StringIO()):
                first = run_pipeline(root, dpi=72, pymupdf=pymupdf)
                pages_first = (root / "data" / "processed" / "pages.jsonl").read_bytes()
                assets_first = (root / "data" / "processed" / "assets.jsonl").read_bytes()
                second = run_pipeline(root, dpi=72, pymupdf=pymupdf)

            pages_second = (root / "data" / "processed" / "pages.jsonl").read_bytes()
            assets_second = (root / "data" / "processed" / "assets.jsonl").read_bytes()
            self.assertEqual(pages_first, pages_second)
            self.assertEqual(assets_first, assets_second)
            self.assertEqual(len(first["text_records"]), 2)
            self.assertEqual(len(second["asset_records"]), 5)
            loaded_assets = [json.loads(line) for line in assets_second.splitlines()]
            self.assertEqual(
                {record["asset_type"] for record in loaded_assets},
                {"embedded_image", "page_preview"},
            )
            for record in loaded_assets:
                self.assertEqual(tuple(record), ASSET_FIELDS)
                self.assertTrue((root / record["asset_path"]).is_file())

    def test_pipeline_never_marks_an_unsaved_file_as_successful(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            dataset = root / "Dataset"
            dataset.mkdir()
            pdf_path = dataset / "unsaved.pdf"
            create_pdf(pdf_path)
            write_test_manifest(dataset, root / "data" / "processed", pdf_path, pages=2)

            with (
                patch.object(image_extractor, "atomic_write_bytes", return_value=None),
                patch.object(page_renderer, "atomic_write_bytes", return_value=None),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = run_pipeline(root, dpi=72, pymupdf=pymupdf)

            self.assertTrue(result["asset_records"])
            self.assertTrue(all(record["status"] == "error" for record in result["asset_records"]))
            self.assertTrue(all(record["asset_path"] is None for record in result["asset_records"]))
            saved = [
                json.loads(line)
                for line in (root / "data" / "processed" / "assets.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertTrue(all(record["status"] == "error" for record in saved))

    def test_asset_only_rerun_repairs_missing_files_without_touching_page_text(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            dataset = root / "Dataset"
            dataset.mkdir()
            pdf_path = dataset / "repair.pdf"
            create_pdf(pdf_path)
            write_test_manifest(dataset, root / "data" / "processed", pdf_path, pages=2)

            with contextlib.redirect_stdout(io.StringIO()):
                initial = run_pipeline(root, dpi=72, pymupdf=pymupdf)
            pages_path = root / "data" / "processed" / "pages.jsonl"
            page_bytes = pages_path.read_bytes()
            previous_ids = {record["asset_id"] for record in initial["asset_records"]}
            missing_image = root / initial["image_records"][0]["asset_path"]
            missing_preview = root / initial["preview_records"][0]["asset_path"]
            missing_image.unlink()
            missing_preview.unlink()

            with contextlib.redirect_stdout(io.StringIO()):
                repaired = run_asset_pipeline(root, dpi=72, pymupdf=pymupdf)

            self.assertTrue(missing_image.is_file())
            self.assertTrue(missing_preview.is_file())
            self.assertEqual(pages_path.read_bytes(), page_bytes)
            self.assertEqual(
                previous_ids, {record["asset_id"] for record in repaired["asset_records"]}
            )
            self.assertEqual(repaired["preview_result"].reused_files, 1)
            self.assertTrue(all(record["status"] == "ok" for record in repaired["asset_records"]))

    def test_asset_repair_can_recover_ids_when_inventory_manifest_is_absent(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            dataset = root / "Dataset"
            dataset.mkdir()
            pdf_path = dataset / "recovery.pdf"
            create_pdf(pdf_path)
            processed = root / "data" / "processed"
            write_test_manifest(dataset, processed, pdf_path, pages=2)

            with contextlib.redirect_stdout(io.StringIO()):
                initial = run_pipeline(root, dpi=72, pymupdf=pymupdf)
            expected_ids = {record["asset_id"] for record in initial["asset_records"]}
            (processed / "manifest.jsonl").unlink()
            (processed / "pages.jsonl").unlink()

            with contextlib.redirect_stdout(io.StringIO()):
                repaired = run_asset_pipeline(root, dpi=72, pymupdf=pymupdf)

            self.assertFalse((processed / "manifest.jsonl").exists())
            self.assertFalse((processed / "pages.jsonl").exists())
            self.assertEqual(expected_ids, {record["asset_id"] for record in repaired["asset_records"]})
            self.assertTrue(all(record["status"] == "ok" for record in repaired["asset_records"]))


if __name__ == "__main__":
    unittest.main()
