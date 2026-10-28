"""Tests for the standalone evidence-grounded answer CLI."""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import answer_question as answer_cli  # noqa: E402
from digital_product_rag.answering.answer_generator import (  # noqa: E402
    AnswerResponse,
    EvidenceCitation,
    SAFE_ABSTENTION_MESSAGE,
)
from digital_product_rag.retrieval.keyword_search import KeywordSearchError  # noqa: E402
from digital_product_rag.retrieval.vector_store import VectorStoreError  # noqa: E402
from digital_product_rag.schemas import SearchResult  # noqa: E402


def _result() -> SearchResult:
    return SearchResult(
        chunk_id="chunk-1",
        document_id="doc-1",
        product_id="prod-1",
        product_title="Sample Product",
        product_category="unknown",
        product_version=None,
        chunk_type="text",
        section_title="Materials",
        text="Use worsted #4 yarn and a crochet hook.",
        filename="sample.pdf",
        relative_path="sample.pdf",
        page_start=3,
        page_end=3,
        source_pages=(3,),
        asset_id=None,
        asset_path=None,
        similarity_score=0.7,
        semantic_score=0.7,
    )


class AnswerQuestionCatalogTests(unittest.TestCase):
    def test_sweater_titles_remain_two_separate_catalog_products(self):
        products, product_by_document = answer_cli._load_catalog(ROOT)
        focus, anchors = answer_cli._product_title_focus(
            "Which materials do I need to make a sweater?", products
        )
        self.assertIn("sweater", anchors)
        self.assertEqual(
            focus,
            frozenset({"product_4a0a46e4b0e054f6", "product_69d77da0c38f524d"}),
        )
        self.assertNotEqual(
            product_by_document["c1d0e3d2-2f91-59c9-84f2-26a4ca159fd9"],
            product_by_document["80f9fbf2-0e41-5db4-9ad6-edc6b4adeaa2"],
        )

    def test_catalog_filter_mismatch_and_malformed_catalog_are_clear(self):
        products, product_by_document = answer_cli._load_catalog(ROOT)
        with self.assertRaisesRegex(ValueError, "Filter mismatch"):
            answer_cli._validate_filters(
                products,
                product_by_document,
                product_id="product_4a0a46e4b0e054f6",
                document_id="80f9fbf2-0e41-5db4-9ad6-edc6b4adeaa2",
            )
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            (root / "config").mkdir()
            (root / "config" / "product_catalog.json").write_text("{not-json", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Could not read product catalog"):
                answer_cli._load_catalog(root)

    def test_product_filter_excludes_other_sweater_product(self):
        focus, _ = answer_cli._product_title_focus(
            "sweater #4 yarn", answer_cli._load_catalog(ROOT)[0]
        )
        self.assertIsNotNone(focus)
        results = answer_cli._retrieve(
            ROOT,
            "sweater #4 yarn",
            retrieval_mode="keyword",
            top_k=5,
            product_id="product_4a0a46e4b0e054f6",
            document_id=None,
            chunk_type=None,
            search_all_for_product_focus=False,
        )
        self.assertTrue(results)
        self.assertTrue(all(row.product_id == "product_4a0a46e4b0e054f6" for row in results))
        self.assertFalse(any(row.product_id == "product_69d77da0c38f524d" for row in results))

    def test_actual_filtered_purchase_price_question_abstains_without_calling_ollama(self):
        query = "What exact purchase price did I personally pay for this PDF?"
        product_id = "product_4a0a46e4b0e054f6"
        results = answer_cli._retrieve(
            ROOT,
            query,
            retrieval_mode="keyword",
            top_k=5,
            product_id=product_id,
            document_id=None,
            chunk_type=None,
            search_all_for_product_focus=False,
        )
        self.assertTrue(results)
        self.assertTrue(all(row.product_id == product_id for row in results))

        with patch(
            "digital_product_rag.answering.answer_generator.urllib.request.urlopen"
        ) as request:
            response = answer_cli.generate_answer(query, results, mode="ollama", model="local-test")

        request.assert_not_called()
        self.assertEqual(response.status, "abstained_insufficient_evidence")
        self.assertTrue(response.abstained)
        self.assertEqual(response.answer, SAFE_ABSTENTION_MESSAGE)
        self.assertEqual(len(response.citations), len(results))

    def test_document_filter_returns_only_the_selected_sweater_document(self):
        document_id = "80f9fbf2-0e41-5db4-9ad6-edc6b4adeaa2"
        results = answer_cli._retrieve(
            ROOT,
            "Pro Tips sweater decorations",
            retrieval_mode="keyword",
            top_k=5,
            product_id=None,
            document_id=document_id,
            chunk_type=None,
            search_all_for_product_focus=False,
        )
        self.assertTrue(results)
        self.assertTrue(all(row.document_id == document_id for row in results))
        self.assertTrue(all(row.product_id == "product_69d77da0c38f524d" for row in results))

    def test_missing_index_and_chunk_dataset_fail_clearly(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            with self.assertRaisesRegex(VectorStoreError, "incomplete or missing"):
                answer_cli._retrieve(
                    root, "yarn", retrieval_mode="semantic", top_k=3,
                    product_id=None, document_id=None, chunk_type=None,
                    search_all_for_product_focus=False,
                )
            with self.assertRaises(KeywordSearchError):
                answer_cli._retrieve(
                    root, "yarn", retrieval_mode="keyword", top_k=3,
                    product_id=None, document_id=None, chunk_type=None,
                    search_all_for_product_focus=False,
                )

    def test_cli_resolves_repository_from_script_not_current_directory(self):
        script = ROOT / "scripts" / "answer_question.py"
        with tempfile.TemporaryDirectory() as working_directory:
            result = subprocess.run(
                [
                    sys.executable, str(script), "--query", "worsted #4 yarn",
                    "--answer-mode", "evidence", "--retrieval-mode", "keyword", "--top-k", "1",
                ],
                cwd=working_directory,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=30,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Evidence-only mode", result.stdout)
        self.assertIn("[E1]", result.stdout)
        self.assertIn("chunk_id=", result.stdout)

    def test_cli_does_not_claim_success_when_generation_fails(self):
        products = {"prod-1": {"title": "Sample Product", "document_ids": ["doc-1"]}}
        documents = {"doc-1": "prod-1"}
        citation = EvidenceCitation(
            reference="E1", chunk_id="chunk-1", document_id="doc-1", product_id="prod-1",
            product_title="Sample Product", filename="sample.pdf", relative_path="sample.pdf",
            page_start=3, page_end=3, source_pages=(3,), section_title="Materials",
            chunk_type="text", text="Use worsted #4 yarn and a crochet hook.", asset_id=None, asset_path=None,
        )
        rejected = AnswerResponse(
            answer=SAFE_ABSTENTION_MESSAGE, status="invalid_citations", abstained=True,
            citations=(citation,), used_citations=(), error="unknown evidence ID E9",
            abstention_reason=SAFE_ABSTENTION_MESSAGE,
        )
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch.object(answer_cli, "_load_catalog", return_value=(products, documents)),
            patch.object(answer_cli, "_product_title_focus", return_value=(None, ())),
            patch.object(answer_cli, "_retrieve", return_value=[_result()]),
            patch.object(answer_cli, "generate_answer", return_value=rejected),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            code = answer_cli.main([
                "--query", "Which yarn?", "--retrieval-mode", "keyword", "--answer-mode", "ollama"
            ])
        self.assertEqual(code, 0)
        self.assertIn(SAFE_ABSTENTION_MESSAGE, stdout.getvalue())
        self.assertIn("Abstained:", stdout.getvalue())
        self.assertIn("[E1]", stdout.getvalue())
        self.assertNotIn("Answer (local Ollama model", stdout.getvalue())
        self.assertIn("Passage:", stdout.getvalue())
        self.assertNotIn("unknown evidence ID", stdout.getvalue())
        self.assertIn("unknown evidence ID", rejected.error)

    def test_cli_applies_top_k_after_product_title_focus(self):
        products = {"prod-1": {"title": "Sample Sweater", "document_ids": ["doc-1"]}}
        documents = {"doc-1": "prod-1"}
        results = [
            SearchResult(
                **{
                    **_result().__dict__,
                    "chunk_id": f"chunk-{number}",
                    "text": f"Use worsted #4 yarn. Passage {number}.",
                    "page_start": number,
                    "page_end": number,
                    "source_pages": (number,),
                }
            )
            for number in range(1, 5)
        ]
        output = io.StringIO()
        with (
            patch.object(answer_cli, "_load_catalog", return_value=(products, documents)),
            patch.object(answer_cli, "_product_title_focus", return_value=(frozenset({"prod-1"}), ("sweater",))),
            patch.object(answer_cli, "_retrieve", return_value=results),
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            code = answer_cli.main([
                "--query", "sweater yarn", "--retrieval-mode", "semantic", "--answer-mode", "evidence", "--top-k", "2"
            ])
        self.assertEqual(code, 0)
        self.assertIn("Retrieved 2 evidence passage(s)", output.getvalue())
        self.assertIn("[E1]", output.getvalue())
        self.assertIn("[E2]", output.getvalue())
        self.assertNotIn("[E3]", output.getvalue())


if __name__ == "__main__":
    unittest.main()
