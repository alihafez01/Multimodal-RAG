"""Tests for labelled dataset integrity and retrieval evaluation metrics."""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from digital_product_rag.application import load_catalog  # noqa: E402
from evaluate_rag import (  # noqa: E402
    _local_answer_evaluation,
    _local_ollama_host,
    calculate_metrics,
    validate_dataset,
)


class EvaluationDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [
            json.loads(line)
            for line in (ROOT / "evaluation" / "datasets" / "rag_questions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        cls.chunk_records = [
            json.loads(line)
            for line in (ROOT / "data" / "processed" / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        cls.chunks = {item["chunk_id"]: item for item in cls.chunk_records}
        cls.products, cls.product_by_document = load_catalog(ROOT)

    def test_all_labels_match_real_product_document_chunk_and_page_records(self):
        self.assertEqual(len(self.rows), 51)
        self.assertEqual(sum(row["answerable"] for row in self.rows), 41)
        self.assertEqual(sum(not row["answerable"] for row in self.rows), 10)
        validate_dataset(self.rows, self.chunks, self.products, self.product_by_document)

    def test_unanswerable_personal_data_scopes_have_no_transaction_markers_in_chunks(self):
        markers = re.compile(
            r"purchase price|amount paid|purchase date|purchaser email|customer email|"
            r"order (?:number|id)|tracking (?:number|id)|refund status|billing address|"
            r"credit card|last four|download url",
            re.IGNORECASE,
        )
        for row in self.rows:
            if row["answerable"]:
                continue
            scope = [
                chunk for chunk in self.chunk_records
                if (row.get("product_id") is None or chunk["product_id"] == row["product_id"])
                and (row.get("document_id") is None or chunk["document_id"] == row["document_id"])
            ]
            self.assertFalse(
                [chunk["chunk_id"] for chunk in scope if markers.search(chunk["text"])],
                row["question_id"],
            )


class RetrievalMetricTests(unittest.TestCase):
    def setUp(self):
        self.chunks = {
            "c1": {"chunk_id": "c1", "document_id": "d1", "product_id": "p1", "filename": "one.pdf", "relative_path": "one.pdf", "source_pages": [1], "chunk_type": "text"},
            "c2": {"chunk_id": "c2", "document_id": "d1", "product_id": "p1", "filename": "one.pdf", "relative_path": "one.pdf", "source_pages": [2], "chunk_type": "text"},
            "c3": {"chunk_id": "c3", "document_id": "d2", "product_id": "p2", "filename": "two.pdf", "relative_path": "two.pdf", "source_pages": [4], "chunk_type": "text"},
            "c4": {"chunk_id": "c4", "document_id": "d3", "product_id": "p3", "filename": "three.pdf", "relative_path": "three.pdf", "source_pages": [9], "chunk_type": "text"},
        }
        self.rows = [
            {"question_id": "a", "query": "query a", "answerable": True, "relevant_chunk_ids": ["c1", "c2"], "relevant_document_ids": ["d1"], "expected_source_pages": [1, 2], "product_id": "p1", "document_id": "d1"},
            {"question_id": "b", "query": "query b", "answerable": True, "relevant_chunk_ids": ["c3"], "relevant_document_ids": ["d2"], "expected_source_pages": [4], "product_id": "p2", "document_id": "d2"},
            {"question_id": "u", "query": "query unanswerable", "answerable": False, "relevant_chunk_ids": [], "relevant_document_ids": [], "expected_source_pages": [], "product_id": "p3", "document_id": "d3"},
        ]

    def _result(self, chunk_id):
        chunk = self.chunks[chunk_id]
        return SimpleNamespace(
            chunk_id=chunk_id,
            document_id=chunk["document_id"],
            product_id=chunk["product_id"],
            filename=chunk["filename"],
            relative_path=chunk["relative_path"],
            source_pages=tuple(chunk["source_pages"]),
            chunk_type=chunk["chunk_type"],
        )

    def test_recall_mrr_page_hit_and_citation_validity_are_distinct_and_correct(self):
        retrieved = {
            "a": [self._result("c1"), self._result("c4")],  # c4 has valid metadata but violates scope.
            "b": [self._result("c4"), self._result("c3")],
            "u": [self._result("c4")],
        }
        metrics = calculate_metrics(self.rows, retrieved, self.chunks, [1, 2])
        k1 = metrics["at_k"]["1"]
        self.assertEqual(k1["recall_at_k"]["value"], 0.25)
        self.assertEqual(k1["mrr_at_k"]["value"], 0.5)
        self.assertEqual(k1["source_page_hit_rate"]["value"], 0.5)
        self.assertEqual(k1["citation_reference_validity"]["value"], 2 / 3)
        self.assertEqual(k1["citation_relevance"]["value"], 0.5)
        k2 = metrics["at_k"]["2"]
        self.assertEqual(k2["recall_at_k"]["value"], 0.75)
        self.assertEqual(k2["mrr_at_k"]["value"], 0.75)
        self.assertEqual(k2["source_page_hit_rate"]["value"], 1.0)
        self.assertEqual(k2["citation_reference_validity"]["value"], 3 / 5)
        self.assertEqual(k2["citation_relevance"]["value"], 0.5)
        self.assertEqual(metrics["unanswerable_retrieval_rate"]["value"], 1.0)
        self.assertEqual(
            [item["question_id"] for item in metrics["failure_analysis"]["product_or_document_scope_violations"]],
            ["a", "b"],
        )

    def test_optional_answer_evaluation_requires_a_configured_model_and_local_host(self):
        response = _local_answer_evaluation(
            [], {}, model=None, host="http://localhost:11434", answer_limit=2
        )
        self.assertEqual(response["status"], "not_evaluated")
        self.assertIn("not configured", response["reason"])
        with self.assertRaisesRegex(ValueError, "local loopback"):
            _local_ollama_host("https://example.com")


if __name__ == "__main__":
    unittest.main()
