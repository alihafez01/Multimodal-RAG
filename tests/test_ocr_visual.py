"""Focused unit tests for selective OCR and visual metadata generation."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import pymupdf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from digital_product_rag.ingestion.ocr import (  # noqa: E402
    TesseractStatus,
    _extract_ocr_only_text,
    evaluate_page,
    merge_native_and_ocr,
    process_ocr,
)
from digital_product_rag.ingestion.visual_describer import (  # noqa: E402
    OUTPUT_FIELDS,
    process_unselected_page_previews,
    process_visual_assets,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _new_image(path: Path, color: int = 128) -> tuple[int, int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 8, 8), False)
    pixmap.clear_with(color)
    pixmap.save(str(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return pixmap.width, pixmap.height, digest


def _make_repo(root: Path, *, page_text: list[str]) -> tuple[str, Path, list[dict]]:
    dataset = root / "Dataset"
    processed = root / "data" / "processed"
    dataset.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    pdf_path = dataset / "sample.pdf"
    pdf = pymupdf.open()
    for number, text in enumerate(page_text):
        page = pdf.new_page(width=300, height=300)
        if text:
            page.insert_text((15, 30), text, fontsize=11)
    pdf.save(pdf_path)
    pdf.close()
    document_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "sample.pdf"))
    manifest_row = {
        "document_id": document_id,
        "filename": pdf_path.name,
        "relative_path": "sample.pdf",
        "file_size_bytes": pdf_path.stat().st_size,
        "page_count": len(page_text),
        "sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        "status": "valid",
        "product_title": "sample",
        "error_message": None,
    }
    _write_jsonl(processed / "manifest.jsonl", [manifest_row])
    page_rows = []
    for page_number, text in enumerate(page_text, start=1):
        page_rows.append({
            "document_id": document_id,
            "filename": "sample.pdf",
            "relative_path": "sample.pdf",
            "page_number": page_number,
            "text": text,
            "text_char_count": len(text),
            "extraction_method": "native_text",
            "ocr_required": False,
            "status": "ok",
            "error_message": None,
        })
    _write_jsonl(processed / "pages.jsonl", page_rows)
    return document_id, processed, page_rows


def _asset_row(document_id: str, page_number: int, asset_id: str, path: str | None,
               width: int | None, height: int | None, digest: str | None,
               *, status: str = "ok", asset_type: str = "embedded_image") -> dict:
    return {
        "asset_id": asset_id,
        "document_id": document_id,
        "filename": "sample.pdf",
        "relative_path": "sample.pdf",
        "page_number": page_number,
        "asset_type": asset_type,
        "asset_path": path,
        "image_format": "png" if path else None,
        "width_px": width,
        "height_px": height,
        "bbox": [0.0, 0.0, 10.0, 10.0] if asset_type != "page_preview" else None,
        "image_xref": 0 if asset_type == "page_preview" else 5,
        "image_index": 1 if asset_type != "page_preview" else None,
        "sha256": digest,
        "extraction_method": "source_image",
        "status": status,
        "error_message": None if status == "ok" else "source extraction failed",
    }


class SelectiveOcrTests(unittest.TestCase):
    def temporary_directory(self):
        return tempfile.TemporaryDirectory(dir=ROOT)

    def test_text_heavy_page_is_not_selected_and_textless_page_is(self) -> None:
        long_text = "A sufficiently long native text paragraph that should not require OCR."
        self.assertEqual(evaluate_page(long_text)[0], [])
        self.assertIn("very_low_native_text", evaluate_page("")[0])

    def test_large_overlapping_image_boxes_trigger_once_without_double_count(self) -> None:
        class Rect:
            x0, y0, x1, y1 = 0, 0, 100, 100

        class FakePage:
            rect = Rect()

            def get_image_info(self, *, xrefs: bool):
                self.assert_xrefs = xrefs
                return [
                    {"bbox": (0, 0, 60, 100)},
                    {"bbox": (40, 0, 100, 100)},
                ]

        reasons, coverage = evaluate_page("short", FakePage())
        self.assertAlmostEqual(coverage, 1.0)
        self.assertIn("large_image_with_limited_text", reasons)

    def test_legacy_false_ocr_flag_does_not_suppress_heuristic(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            document_id, processed, page_rows = _make_repo(root, page_text=[""])
            original = (processed / "pages.jsonl").read_bytes()
            result = process_ocr(
                root, dry_run=True, pymupdf=pymupdf,
                tesseract_status=TesseractStatus(None, None, (), "not installed"),
            )
            self.assertEqual(len(result["candidates"]), 1)
            self.assertEqual(result["candidates"][0]["page"]["document_id"], document_id)
            self.assertFalse(page_rows[0]["ocr_required"])
            self.assertEqual((processed / "pages.jsonl").read_bytes(), original)

    def test_native_text_is_preserved_and_duplicate_ocr_lines_are_filtered(self) -> None:
        native = "Native Text\nKeep this line."
        combined = merge_native_and_ocr(native, "native   text!\nNew OCR line")
        self.assertTrue(combined.startswith(native))
        self.assertEqual(combined, "Native Text\nKeep this line.\nNew OCR line")

    def test_partial_ocr_textpage_keeps_only_glyphless_ocr_spans(self) -> None:
        class FakePage:
            def get_text(self, kind, *, textpage, sort):
                self.assert_call = (kind, textpage, sort)
                return {"blocks": [{"lines": [{"spans": [
                    {"font": "Helvetica", "text": "Native words"},
                    {"font": "GlyphLessFont", "text": "OCR label"},
                ]}]}]}

        result = _extract_ocr_only_text(FakePage(), object(), "Native words")
        self.assertEqual(result, "OCR label")

    def test_unavailable_tesseract_writes_explicit_skip_and_preserves_baseline(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            _, processed, _ = _make_repo(root, page_text=[""])
            baseline = (processed / "pages.jsonl").read_bytes()
            result = process_ocr(
                root, pymupdf=pymupdf,
                tesseract_status=TesseractStatus(None, None, (), "not on PATH"),
            )
            enriched = [json.loads(line) for line in (processed / "pages_enriched.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(result["successes"], 0)
            self.assertEqual(enriched[0]["ocr_status"], "skipped_tesseract_unavailable")
            self.assertEqual(enriched[0]["ocr_text"], "")
            self.assertEqual(enriched[0]["text_char_count"], len(enriched[0]["text"]))
            self.assertEqual((processed / "pages.jsonl").read_bytes(), baseline)

    def test_one_ocr_failure_does_not_stop_later_candidate_page(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            _, processed, _ = _make_repo(root, page_text=["", ""])
            original_ocr = pymupdf.Page.get_textpage_ocr

            def mocked_ocr(page, *args, **kwargs):
                if page.number == 0:
                    raise RuntimeError("synthetic OCR failure")
                return page.get_textpage()

            with patch.object(pymupdf.Page, "get_textpage_ocr", new=mocked_ocr):
                process_ocr(
                    root, force=True, pymupdf=pymupdf,
                    tesseract_status=TesseractStatus("tesseract", "tesseract 5", ("eng",)),
                )
            enriched = [json.loads(line) for line in (processed / "pages_enriched.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["ocr_status"] for row in enriched], ["failed", "completed"])


class VisualMetadataTests(unittest.TestCase):
    def temporary_directory(self):
        return tempfile.TemporaryDirectory(dir=ROOT)

    def _fixture(self, root: Path):
        document_id, processed, page_rows = _make_repo(root, page_text=["Native page text."])
        page_rows[0].update({
            "native_text": "Native page text.",
            "ocr_text": "OCR detected label",
            "text": "Native page text.\nOCR detected label",
            "text_char_count": len("Native page text.\nOCR detected label"),
            "ocr_required": True,
        })
        _write_jsonl(processed / "pages_enriched.jsonl", page_rows)
        image_path = root / "data" / "assets" / "images" / "sample.png"
        width, height, digest = _new_image(image_path)
        relative = image_path.relative_to(root).as_posix()
        rows = [
            _asset_row(document_id, 1, "asset-1", relative, width, height, digest),
            _asset_row(document_id, 1, "asset-2", relative, width, height, digest),
            _asset_row(document_id, 1, "asset-missing", "data/assets/images/missing.png", 10, 10, "bad"),
        ]
        _write_jsonl(processed / "assets.jsonl", rows)
        return document_id, processed, rows, image_path

    def test_all_assets_get_records_and_ocr_excerpt_is_linked_to_page(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            document_id, processed, assets, _ = self._fixture(root)
            result = process_visual_assets(root, vision_mode="off", pymupdf=pymupdf)
            records = [json.loads(line) for line in (processed / "visual_metadata.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), len(assets))
            self.assertEqual([row["asset_id"] for row in records], [row["asset_id"] for row in assets])
            self.assertTrue(all(tuple(row) == OUTPUT_FIELDS for row in records))
            self.assertEqual(records[0]["document_id"], document_id)
            self.assertEqual(records[0]["page_number"], 1)
            self.assertEqual(records[0]["page_ocr_text_excerpt"], "OCR detected label")
            self.assertEqual(records[2]["description_status"], "invalid_asset")
            self.assertEqual(result["status_counts"]["invalid_asset"], 1)

    def test_no_local_model_skips_descriptions_without_fabricating_them(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            _, processed, assets, _ = self._fixture(root)
            with patch(
                "digital_product_rag.ingestion.visual_describer.discover_ollama_models",
                return_value=([], "Ollama unavailable"),
            ):
                process_visual_assets(root, vision_mode="selected", pymupdf=pymupdf)
            records = [json.loads(line) for line in (processed / "visual_metadata.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(records[0]["description_status"], "skipped_model_unavailable")
            self.assertIsNone(records[0]["description"])
            self.assertEqual(records[2]["description_status"], "invalid_asset")

    def test_verified_duplicate_hash_reuses_one_genuine_model_description(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            _, processed, assets, _ = self._fixture(root)
            with (
                patch("digital_product_rag.ingestion.visual_describer.discover_ollama_models", return_value=(['vision:latest'], None)),
                patch("digital_product_rag.ingestion.visual_describer.describe_image", return_value="A small plain image.") as describe,
            ):
                result = process_visual_assets(
                    root, vision_mode="selected", vision_model="vision:latest", pymupdf=pymupdf
                )
            records = [json.loads(line) for line in (processed / "visual_metadata.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(describe.call_count, 1)
            self.assertEqual(result["unique_vision_requests"], 1)
            self.assertEqual([record["asset_id"] for record in records[:2]], [assets[0]["asset_id"], assets[1]["asset_id"]])
            self.assertEqual([record["description_status"] for record in records[:2]], ["generated", "generated"])
            self.assertTrue(all(record["description_source"] == "vision_model" for record in records[:2]))

    def test_bad_model_response_does_not_stop_later_unique_asset(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            document_id, processed, rows, _ = self._fixture(root)
            second_path = root / "data" / "assets" / "images" / "another.png"
            width, height, digest = _new_image(second_path, color=210)
            rows.append(_asset_row(
                document_id, 1, "asset-unique", second_path.relative_to(root).as_posix(),
                width, height, digest,
            ))
            _write_jsonl(processed / "assets.jsonl", rows)
            with (
                patch("digital_product_rag.ingestion.visual_describer.discover_ollama_models", return_value=(['vision:latest'], None)),
                patch(
                    "digital_product_rag.ingestion.visual_describer.describe_image",
                    side_effect=[ValueError("malformed model response"), "A second distinct image."],
                ) as describe,
            ):
                process_visual_assets(root, vision_mode="selected", vision_model="vision:latest", pymupdf=pymupdf)
            records = [json.loads(line) for line in (processed / "visual_metadata.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(describe.call_count, 2)
            self.assertEqual(records[0]["description_status"], "failed")
            self.assertEqual(records[1]["description_status"], "failed")
            self.assertEqual(records[3]["description_status"], "generated")

    def test_previous_descriptions_are_reused_when_refreshing_ocr_excerpts(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            _, processed, _, _ = self._fixture(root)
            page_file = processed / "pages_enriched.jsonl"
            page = json.loads(page_file.read_text(encoding="utf-8").splitlines()[0])
            page["ocr_text"] = ""
            page["ocr_text_char_count"] = 0
            page["text"] = page["native_text"]
            page["text_char_count"] = len(page["text"])
            _write_jsonl(page_file, [page])
            with (
                patch("digital_product_rag.ingestion.visual_describer.discover_ollama_models", return_value=(['vision:latest'], None)),
                patch("digital_product_rag.ingestion.visual_describer.describe_image", return_value="A verified local caption.") as describe,
            ):
                process_visual_assets(
                    root, vision_mode="selected", vision_model="vision:latest", pymupdf=pymupdf
                )
            self.assertEqual(describe.call_count, 1)

            page["ocr_text"] = "New OCR label"
            page["ocr_text_char_count"] = len(page["ocr_text"])
            page["text"] = page["native_text"] + "\n" + page["ocr_text"]
            page["text_char_count"] = len(page["text"])
            _write_jsonl(page_file, [page])
            with (
                patch("digital_product_rag.ingestion.visual_describer.discover_ollama_models", return_value=([], "Ollama unavailable")),
                patch("digital_product_rag.ingestion.visual_describer.describe_image", side_effect=AssertionError("cached caption should be reused")),
            ):
                refreshed = process_visual_assets(
                    root, vision_mode="selected", vision_model="vision:latest", pymupdf=pymupdf
                )
            records = [json.loads(line) for line in (processed / "visual_metadata.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(refreshed["reused_descriptions"], 2)
            self.assertEqual([record["description_status"] for record in records[:2]], ["generated", "generated"])
            self.assertEqual([record["description"] for record in records[:2]], ["A verified local caption."] * 2)
            self.assertEqual([record["page_ocr_text_excerpt"] for record in records[:2]], ["New OCR label"] * 2)

    def test_targeted_page_preview_run_preserves_other_records_and_records_asset_errors(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            document_id, processed, assets, _ = self._fixture(root)
            preview_path = root / "data" / "assets" / "page_previews" / "sample.png"
            width, height, digest = _new_image(preview_path, color=200)
            preview = _asset_row(
                document_id, 1, "preview-target", preview_path.relative_to(root).as_posix(),
                width, height, digest, asset_type="page_preview",
            )
            missing = _asset_row(
                document_id, 1, "preview-missing", "data/assets/page_previews/missing.png",
                width, height, "missing-digest", asset_type="page_preview",
            )
            assets.extend([preview, missing])
            _write_jsonl(processed / "assets.jsonl", assets)

            def metadata(asset, description, status, source="vision_model", model="qwen3-vl:4b"):
                row = {
                    **asset,
                    "page_text_excerpt": "Existing page excerpt",
                    "page_ocr_text_excerpt": "",
                    "related_chunk_ids": [],
                    "description": description,
                    "description_source": source,
                    "description_status": status,
                    "description_model": model if status == "generated" else None,
                    "description_error": None,
                }
                return {field: row.get(field) for field in OUTPUT_FIELDS}

            rows = [
                metadata(assets[0], "Keep this embedded caption.", "generated"),
                metadata(assets[1], "Keep the second embedded caption.", "generated"),
                metadata(assets[2], None, "invalid_asset", "none"),
                metadata(preview, "Keep this existing preview caption.", "generated"),
                metadata(preview, None, "not_selected", "none"),
                metadata(missing, None, "not_selected", "none"),
            ]
            # Keep the generated and unselected records on distinct page-preview IDs.
            generated_preview = {**preview, "asset_id": "preview-generated"}
            generated_preview_path = root / "data" / "assets" / "page_previews" / "generated.png"
            generated_width, generated_height, generated_digest = _new_image(generated_preview_path, color=60)
            generated_preview.update({
                "asset_path": generated_preview_path.relative_to(root).as_posix(),
                "width_px": generated_width,
                "height_px": generated_height,
                "sha256": generated_digest,
            })
            rows[3] = metadata(generated_preview, "Keep this existing preview caption.", "generated")
            assets.append(generated_preview)
            rows[4] = metadata(preview, None, "not_selected", "none")
            _write_jsonl(processed / "assets.jsonl", assets)
            _write_jsonl(processed / "visual_metadata.jsonl", rows)
            baseline = [dict(row) for row in rows]

            with (
                patch(
                    "digital_product_rag.ingestion.visual_describer.discover_ollama_models",
                    return_value=(["qwen3-vl:4b"], None),
                ),
                patch(
                    "digital_product_rag.ingestion.visual_describer.describe_image",
                    return_value="A page shows a labeled crochet diagram with arrows indicating stitch direction.",
                ) as describe,
            ):
                result = process_unselected_page_previews(
                    root, vision_model="qwen3-vl:4b", pymupdf=pymupdf, show_progress=False
                )

            saved = [
                json.loads(line)
                for line in (processed / "visual_metadata.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(result["target_count"], 2)
            self.assertEqual(result["attempted_requests"], 1)
            self.assertEqual(describe.call_count, 1)
            self.assertEqual(len(saved), len(rows))
            self.assertEqual(len({row["asset_id"] for row in saved}), len(rows))
            self.assertEqual([row["asset_id"] for row in saved], [row["asset_id"] for row in baseline])
            self.assertEqual(saved[0], baseline[0])
            self.assertEqual(saved[1], baseline[1])
            self.assertEqual(saved[3], baseline[3])
            self.assertEqual(saved[4]["description_status"], "generated")
            self.assertEqual(saved[4]["description_model"], "qwen3-vl:4b")
            self.assertIn("labeled crochet diagram", saved[4]["description"])
            self.assertEqual(saved[4]["asset_path"], preview["asset_path"])
            self.assertEqual(saved[4]["sha256"], preview["sha256"])
            self.assertEqual(saved[5]["description_status"], "invalid_asset")
            self.assertIn("missing or unreadable", saved[5]["description_error"])


if __name__ == "__main__":
    unittest.main()
