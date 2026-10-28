"""Tests for the local PDF inventory script."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import inspect_dataset


def synthetic_pdf(page_count: int) -> bytes:
    """Build a small structurally valid PDF without external test dependencies."""
    page_objects = [
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 10 10] >>".encode("ascii")
        for _ in range(page_count)
    ]
    kids = " ".join(f"{index + 3} 0 R" for index in range(page_count))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode("ascii"),
        *page_objects,
    ]

    content = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(content))
        content.extend(f"{number} 0 obj\n".encode("ascii"))
        content.extend(body)
        content.extend(b"\nendobj\n")

    xref_offset = len(content)
    content.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    content.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        content.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    content.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(content)


class _FakeDocument:
    def __init__(self, payload: bytes) -> None:
        self.is_pdf = True
        self.needs_pass = payload.startswith(b"%PDF-ENCRYPTED")
        self.page_count = payload.count(b"/Type /Page ")

    def close(self) -> None:
        pass


class _FakePyMuPDF:
    @staticmethod
    def open(path: Path) -> _FakeDocument:
        payload = path.read_bytes()
        if not payload.startswith(b"%PDF-"):
            raise RuntimeError("malformed PDF")
        return _FakeDocument(payload)


class InspectDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if importlib.util.find_spec("pymupdf") is None:
            cls.pdf_api = _FakePyMuPDF
        else:
            cls.pdf_api = inspect_dataset.load_pymupdf()

    def temporary_directory(self):
        # Keep test writes inside the workspace writable root on restricted Windows setups.
        return tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])

    def run_inventory(self, dataset: Path, output: Path) -> list[dict]:
        with contextlib.redirect_stdout(io.StringIO()):
            return inspect_dataset.generate_manifest(
                dataset, output, pymupdf=self.__class__.pdf_api
            )

    def test_valid_synthetic_pdf_has_page_count_and_manifest_schema(self) -> None:
        with self.temporary_directory() as directory:
            root = Path(directory)
            dataset = root / "Dataset"
            dataset.mkdir()
            pdf = dataset / "crochet_baby_blanket-pattern.pdf"
            pdf.write_bytes(synthetic_pdf(2))
            output = root / "data" / "processed" / "manifest.jsonl"

            records = self.run_inventory(dataset, output)
            lines = output.read_text(encoding="utf-8").splitlines()
            loaded = [json.loads(line) for line in lines]

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["page_count"], 2)
            self.assertEqual(records[0]["status"], "valid")
            self.assertEqual(records[0]["product_title"], "crochet baby blanket pattern")
            self.assertEqual(tuple(loaded[0]), inspect_dataset.MANIFEST_FIELDS)
            self.assertEqual(loaded[0]["filename"], pdf.name)
            self.assertEqual(loaded[0]["relative_path"], pdf.name)
            self.assertEqual(loaded[0]["file_size_bytes"], pdf.stat().st_size)
            self.assertEqual(loaded[0]["sha256"], hashlib.sha256(pdf.read_bytes()).hexdigest())

    def test_malformed_pdf_does_not_stop_valid_pdf_processing(self) -> None:
        with self.temporary_directory() as directory:
            root = Path(directory)
            dataset = root / "Dataset"
            dataset.mkdir()
            (dataset / "a-malformed.pdf").write_bytes(b"not a PDF")
            (dataset / "b-valid.pdf").write_bytes(synthetic_pdf(1))

            records = self.run_inventory(dataset, root / "manifest.jsonl")

            self.assertEqual([record["status"] for record in records], ["unreadable", "valid"])
            self.assertIsNone(records[0]["page_count"])
            self.assertEqual(records[1]["page_count"], 1)

    def test_duplicate_hashes_keep_separate_records_and_ids(self) -> None:
        with self.temporary_directory() as directory:
            root = Path(directory)
            dataset = root / "Dataset"
            (dataset / "nested").mkdir(parents=True)
            payload = synthetic_pdf(1)
            (dataset / "one.pdf").write_bytes(payload)
            (dataset / "nested" / "copy.PDF").write_bytes(payload)

            records = self.run_inventory(dataset, root / "manifest.jsonl")
            groups = inspect_dataset.find_duplicate_groups(records)

            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["sha256"], records[1]["sha256"])
            self.assertNotEqual(records[0]["document_id"], records[1]["document_id"])
            self.assertEqual(groups, [["nested/copy.PDF", "one.pdf"]])

    def test_encrypted_pdf_is_reported_without_a_page_count(self) -> None:
        with self.temporary_directory() as directory:
            root = Path(directory)
            dataset = root / "Dataset"
            dataset.mkdir()
            (dataset / "locked.pdf").write_bytes(b"%PDF-ENCRYPTED")

            with contextlib.redirect_stdout(io.StringIO()):
                records = inspect_dataset.generate_manifest(
                    dataset, root / "manifest.jsonl", pymupdf=_FakePyMuPDF
                )

            self.assertEqual(records[0]["status"], "encrypted")
            self.assertIsNone(records[0]["page_count"])
            self.assertIn("password", records[0]["error_message"])

    def test_document_ids_are_stable_across_runs(self) -> None:
        with self.temporary_directory() as directory:
            root = Path(directory)
            dataset = root / "Dataset"
            dataset.mkdir()
            (dataset / "product.pdf").write_bytes(synthetic_pdf(1))

            first = self.run_inventory(dataset, root / "first.jsonl")
            second = self.run_inventory(dataset, root / "second.jsonl")

            self.assertEqual(first[0]["document_id"], second[0]["document_id"])

    def test_missing_dataset_returns_nonzero_with_actionable_error(self) -> None:
        with self.temporary_directory() as directory:
            missing = Path(directory) / "missing"
            output = io.StringIO()
            with contextlib.redirect_stderr(output):
                result = inspect_dataset.main(
                    ["--dataset", str(missing), "--output", str(Path(directory) / "out.jsonl")]
                )

            self.assertNotEqual(result, 0)
            self.assertIn("Dataset directory not found", output.getvalue())
            self.assertIn("--dataset PATH", output.getvalue())

    def test_empty_dataset_writes_empty_manifest_and_summary(self) -> None:
        with self.temporary_directory() as directory:
            root = Path(directory)
            dataset = root / "Dataset"
            dataset.mkdir()
            output = root / "manifest.jsonl"
            stdout = io.StringIO()
            with patch.object(inspect_dataset, "load_pymupdf", side_effect=AssertionError):
                with contextlib.redirect_stdout(stdout):
                    records = inspect_dataset.generate_manifest(dataset, output)

            self.assertEqual(records, [])
            self.assertEqual(output.read_bytes(), b"")
            self.assertIn("Warning: no PDF files", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
