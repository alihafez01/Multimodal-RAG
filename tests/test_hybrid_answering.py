"""Tests for BM25/hybrid retrieval and evidence-grounded answer safeguards."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from urllib.error import URLError
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from digital_product_rag.answering.answer_generator import (  # noqa: E402
    AnswerGenerationError,
    SAFE_ABSTENTION_MESSAGE,
    answer_query,
    answer_question,
    validate_answer_citations,
)
from digital_product_rag.retrieval.hybrid_search import (  # noqa: E402
    reciprocal_rank_fusion,
    search_hybrid,
)
from digital_product_rag.retrieval.keyword_search import KeywordSearchIndex, tokenize  # noqa: E402
from digital_product_rag.schemas import SearchResult  # noqa: E402


def _chunk(
    chunk_id: str,
    text: str,
    *,
    doc: str = "doc-1",
    product: str = "prod-1",
    kind: str = "text",
    page: int = 1,
) -> dict[str, object]:
    image = kind == "image_description"
    return {
        "chunk_id": chunk_id,
        "document_id": doc,
        "product_id": product,
        "product_title": f"Product {product}",
        "product_category": "craft",
        "product_version": None,
        "filename": f"{doc}.pdf",
        "relative_path": f"{doc}.pdf",
        "page_start": page,
        "page_end": page,
        "source_pages": [page],
        "section_title": "Materials",
        "chunk_type": kind,
        "text": text,
        "asset_id": f"asset-{chunk_id}" if image else None,
        "asset_path": f"data/assets/{chunk_id}.png" if image else None,
        "metadata_status": None,
    }


def _result(chunk_id: str, *, mode: str = "semantic", page: int = 1) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        product_id="prod-1",
        product_title="Product prod-1",
        product_category="craft",
        product_version=None,
        chunk_type="text",
        section_title="Materials",
        text="Materials include wool yarn and a 4 mm hook.",
        filename="doc-1.pdf",
        relative_path="doc-1.pdf",
        page_start=page,
        page_end=page,
        source_pages=(page,),
        asset_id=None,
        asset_path=None,
        similarity_score=0.7 if mode == "semantic" else None,
        semantic_score=0.7 if mode == "semantic" else None,
        bm25_score=2.5 if mode == "keyword" else None,
        retrieval_mode=mode,
    )


class KeywordSearchTests(unittest.TestCase):
    def test_tokenizer_keeps_crochet_terms_sizes_fractions_and_numbers(self):
        tokens = tokenize("SC, DC and MR use #4 yarn; round-12 has 1/2 inch rows.")
        self.assertIn("sc", tokens)
        self.assertIn("dc", tokens)
        self.assertIn("mr", tokens)
        self.assertIn("#4", tokens)
        self.assertIn("round-12", tokens)
        self.assertIn("1/2", tokens)
        self.assertNotIn("and", tokens)

    def test_bm25_scores_filters_and_metadata(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            path = Path(temp) / "chunks.jsonl"
            rows = [
                _chunk("materials-1", "Worsted #4 yarn is listed for this product."),
                _chunk("image-1", "A diagram shows yarn strands and stitch placement.", kind="image_description", page=2),
                _chunk("materials-2", "Cotton thread and a small hook are listed.", doc="doc-2", product="prod-2"),
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            index = KeywordSearchIndex(path)
            results = index.search("WORSTED #4", top_k=5)
            self.assertEqual([item.chunk_id for item in results], ["materials-1"])
            self.assertGreater(results[0].bm25_score, 0)
            self.assertIsNone(results[0].similarity_score)
            self.assertEqual(results[0].source_pages, (1,))
            image = index.search("diagram yarn", chunk_type="image_description")
            self.assertEqual(image[0].chunk_id, "image-1")
            self.assertEqual(image[0].asset_id, "asset-image-1")
            self.assertEqual(image[0].asset_path, "data/assets/image-1.png")
            self.assertEqual(index.search("cotton", product_id="prod-1", document_id="doc-2"), [])
            self.assertEqual(index.search("the and"), [])
            with self.assertRaisesRegex(ValueError, "Unknown product_id"):
                index.search("yarn", product_id="not-a-product")


class HybridSearchTests(unittest.TestCase):
    def test_rrf_merges_duplicates_keeps_one_list_hits_and_sorts_ties(self):
        semantic = [_result("a"), _result("b")]
        keyword = [
            replace(_result("b", mode="keyword"), bm25_score=4.0),
            replace(_result("a", mode="keyword"), bm25_score=3.0),
            replace(_result("c", mode="keyword"), bm25_score=2.0),
        ]
        fused = reciprocal_rank_fusion(semantic, keyword, top_k=3, rrf_k=60)
        self.assertEqual([item.chunk_id for item in fused], ["a", "b", "c"])
        self.assertAlmostEqual(fused[0].rrf_score, 1 / 61 + 1 / 62)
        self.assertEqual(fused[0].semantic_rank, 1)
        self.assertEqual(fused[0].bm25_rank, 2)
        self.assertEqual(fused[0].semantic_score, 0.7)
        self.assertEqual(fused[0].bm25_score, 3.0)
        self.assertIsNone(fused[2].semantic_rank)
        self.assertIsNone(fused[2].semantic_score)
        self.assertEqual(fused[2].retrieval_mode, "hybrid")

    def test_filtered_hybrid_keeps_keyword_only_hits_and_uses_all_filtered_candidates(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            chunk_path = root / "data" / "processed" / "chunks.jsonl"
            chunk_path.parent.mkdir(parents=True)
            rows = [
                _chunk("yarn-1", "Worsted yarn is recommended."),
                _chunk("yarn-2", "Yarn is used for the second sample.", doc="doc-2", product="prod-2"),
            ]
            chunk_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            calls = []

            def no_semantic_results(*args, **kwargs):
                calls.append(kwargs)
                return []

            results = search_hybrid(
                root,
                "yarn",
                top_k=1,
                candidate_limit=1,
                product_id="prod-1",
                semantic_retriever=no_semantic_results,
            )
        self.assertEqual([result.chunk_id for result in results], ["yarn-1"])
        self.assertIsNone(results[0].semantic_rank)
        self.assertEqual(results[0].bm25_rank, 1)
        self.assertEqual(calls[0]["top_k"], 2)
        self.assertEqual(calls[0]["product_id"], "prod-1")


class EvidenceAnswerTests(unittest.TestCase):
    def _model_answer(self, text, *, source=None, citation_ids=None, query="Which yarn is listed?"):
        result = source or _result("evidence-1")
        raw = json.dumps({
            "answer": text,
            "citation_ids": citation_ids or ["E1"],
            "abstain": False,
            "abstention_reason": None,
        })
        with patch(
            "digital_product_rag.answering.answer_generator._ollama_generate", return_value=raw
        ):
            return answer_question(query, [result], mode="ollama", model="local-test")

    def test_inline_citations_are_validated_without_becoming_factual_terms(self):
        response = self._model_answer("The materials include wool yarn [E1].")
        self.assertEqual(response.status, "answered")
        self.assertEqual(response.used_citations, ("E1",))
        fabricated = self._model_answer("The materials include wool yarn [E99].")
        self.assertEqual(fabricated.status, "invalid_citations")
        self.assertTrue(fabricated.abstained)
        self.assertIn("unknown evidence", fabricated.error)
        omitted = self._model_answer("The materials include wool yarn [E1].\nUse a 4 mm hook.")
        self.assertEqual(omitted.status, "invalid_citations")
        self.assertIn("Every answer sentence", omitted.error)

    def test_names_only_sequence_preserves_source_steps_and_order(self):
        source = replace(
            _result("sequence"), product_title="Sample Bag", section_title="Construction",
            text="The construction sequence is:\nCut fabric → pin edges → sew seam.\nUse wool yarn.",
        )
        query = "Which product and what are the construction step names only?"
        ordered = "Sample Bag [E1]\n1. Cut fabric [E1]\n2. Pin edges [E1]\n3. Sew seam [E1]"
        response = self._model_answer(ordered, source=source, query=query)
        self.assertEqual(response.status, "answered", response.error)
        self.assertEqual(response.answer, ordered)
        for text in (
            "Sample Bag\n1. Pin edges\n2. Cut fabric\n3. Sew seam",
            "Sample Bag\n1. Cut fabric\n2. Sew seam",
            "Sample Bag\n1. Cut fabric\n2. Pin edges\n3. Sew seam\n4. Use wool yarn",
            "Sample Bag\n1. Cut fabric\n2. Sew edges\n3. Pin seam",
        ):
            with self.subTest(text=text):
                rejected = self._model_answer(text, source=source, query=query)
                self.assertEqual(rejected.status, "invalid_citations")
                self.assertTrue(rejected.abstained)
                self.assertIn("documented sequence", rejected.error)

    def test_list_numbering_does_not_hide_unsupported_quantity(self):
        response = self._model_answer("1. Use 400 yards of wool yarn [E1].")
        self.assertEqual(response.status, "invalid_citations")
        self.assertIn("400", response.error)

    def test_negation_cannot_be_added_or_removed_from_source_instructions(self):
        source = replace(_result("polarity"), text="Use wool yarn. Do not use cotton yarn.")
        for text in ("Do not use wool yarn.", "Use cotton yarn."):
            with self.subTest(text=text):
                rejected = self._model_answer(text, source=source)
                self.assertEqual(rejected.status, "invalid_citations")
                self.assertTrue(rejected.abstained)
        accepted = self._model_answer("Do not use cotton yarn.", source=source)
        self.assertEqual(accepted.status, "answered", accepted.error)

    def test_evidence_only_abstention_and_citation_validation(self):
        result = _result("evidence-1")
        evidence = answer_query("Which yarn is listed?", [result])
        self.assertEqual(evidence.status, "evidence_only")
        self.assertFalse(evidence.abstained)
        self.assertEqual(evidence.citations[0].reference, "C1")
        self.assertEqual(evidence.citations[0].source_pages, (1,))
        abstention = answer_query("the and", [result])
        self.assertTrue(abstention.abstained)
        self.assertEqual(abstention.status, "abstained_insufficient_evidence")
        self.assertEqual(validate_answer_citations("Wool yarn is listed [C1].", evidence.citations), ("C1",))
        with self.assertRaisesRegex(AnswerGenerationError, "unknown evidence"):
            validate_answer_citations("Wool yarn is listed [C9].", evidence.citations)
        with self.assertRaisesRegex(AnswerGenerationError, "no evidence citations"):
            validate_answer_citations("Wool yarn is listed.", evidence.citations)
        with self.assertRaisesRegex(AnswerGenerationError, "Every answer sentence"):
            validate_answer_citations("Wool yarn is listed [C1]. The page mentions a hook.", evidence.citations)

    def test_local_llm_answer_uses_only_resolving_citations(self):
        result = _result("evidence-1")

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                structured = {
                    "answer": "The materials include wool yarn.",
                    "citation_ids": ["C1"],
                    "abstain": False,
                    "abstention_reason": None,
                }
                return json.dumps({"response": json.dumps(structured)}).encode("utf-8")

        with patch(
            "digital_product_rag.answering.answer_generator.urllib.request.urlopen",
            return_value=FakeResponse(),
        ) as request:
            response = answer_query(
                "Which yarn is listed?", [result], mode="llm", model="local-test",
                ollama_host="http://127.0.0.1:11434",
            )
        self.assertEqual(response.status, "answered")
        self.assertFalse(response.abstained)
        self.assertEqual(response.used_citations, ("C1",))
        self.assertIn("wool yarn", response.answer)
        request.assert_called_once()
        self.assertIn("127.0.0.1:11434/api/generate", request.call_args.args[0].full_url)

    def test_invalid_model_citations_are_not_displayed_as_an_answer(self):
        result = _result("evidence-1")

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                structured = {
                    "answer": "The materials include wool yarn.",
                    "citation_ids": ["C9"],
                    "abstain": False,
                    "abstention_reason": None,
                }
                return json.dumps({"response": json.dumps(structured)}).encode("utf-8")

        with patch(
            "digital_product_rag.answering.answer_generator.urllib.request.urlopen",
            return_value=FakeResponse(),
        ):
            response = answer_query("Which yarn is listed?", [result], mode="llm", model="local-test")
        self.assertEqual(response.status, "citation_validation_failed")
        self.assertTrue(response.abstained)
        self.assertEqual(response.answer, SAFE_ABSTENTION_MESSAGE)
        self.assertEqual(response.used_citations, ())
        self.assertIsNotNone(response.error)

    def test_missing_ollama_model_is_reported_without_blocking_evidence(self):
        result = _result("evidence-1")
        with patch.dict("os.environ", {"RAG_GENERATION_MODEL": ""}):
            response = answer_question("Which yarn is listed?", [result], mode="ollama", model=None)
        self.assertEqual(response.status, "ollama_unavailable")
        self.assertFalse(response.abstained)
        self.assertIsNone(response.answer)
        self.assertTrue(response.citations)

    def test_unavailable_ollama_service_is_reported_without_claiming_an_answer(self):
        result = _result("evidence-1")
        with patch(
            "digital_product_rag.answering.answer_generator.urllib.request.urlopen",
            side_effect=URLError("connection refused"),
        ):
            response = answer_question(
                "Which yarn is listed?", [result], mode="ollama", model="local-test",
                ollama_host="http://127.0.0.1:11434",
            )
        self.assertEqual(response.status, "ollama_unavailable")
        self.assertIsNone(response.answer)
        self.assertTrue(response.citations)
        self.assertIn("unavailable", response.error)

    def test_structured_json_validation_abstention_and_malformed_responses(self):
        result = _result("evidence-1")

        class FakeResponse:
            def __init__(self, response_text):
                self.response_text = response_text

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"response": self.response_text}).encode("utf-8")

        abstain_json = json.dumps({
            "answer": "The passages do not establish a single material choice.",
            "citation_ids": ["E1"],
            "abstain": True,
            "abstention_reason": "The passages give different yarn recommendations.",
        })
        with patch(
            "digital_product_rag.answering.answer_generator.urllib.request.urlopen",
            return_value=FakeResponse(abstain_json),
        ):
            response = answer_question("Which yarn is listed?", [result], mode="ollama", model="local-test")
        self.assertEqual(response.status, "model_abstained")
        self.assertTrue(response.abstained)
        self.assertEqual(response.abstention_reason, SAFE_ABSTENTION_MESSAGE)
        self.assertEqual(response.answer, SAFE_ABSTENTION_MESSAGE)
        self.assertIn("different yarn recommendations", response.error)
        self.assertEqual(response.citations[0].reference, "E1")

        with patch(
            "digital_product_rag.answering.answer_generator.urllib.request.urlopen",
            return_value=FakeResponse("not-json"),
        ):
            malformed = answer_question("Which yarn is listed?", [result], mode="ollama", model="local-test")
        self.assertEqual(malformed.status, "invalid_model_response")
        self.assertTrue(malformed.abstained)
        self.assertEqual(malformed.answer, SAFE_ABSTENTION_MESSAGE)
        self.assertTrue(malformed.citations)
        self.assertIn("valid JSON", malformed.error)

    def test_unsupported_numeric_answer_is_rejected_even_with_a_known_citation(self):
        result = _result("evidence-1")

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                body = {
                    "answer": "Use 400 yards of red yarn.",
                    "citation_ids": ["E1"],
                    "abstain": False,
                    "abstention_reason": None,
                }
                return json.dumps({"response": json.dumps(body)}).encode("utf-8")

        with patch(
            "digital_product_rag.answering.answer_generator.urllib.request.urlopen",
            return_value=FakeResponse(),
        ):
            response = answer_question("Which yarn is listed?", [result], mode="ollama", model="local-test")
        self.assertEqual(response.status, "invalid_citations")
        self.assertTrue(response.abstained)
        self.assertEqual(response.answer, SAFE_ABSTENTION_MESSAGE)
        self.assertEqual(response.used_citations, ())
        self.assertIn("absent from its cited", response.error)

    def test_answer_with_an_uncited_factual_detail_is_rejected(self):
        result = _result("evidence-1")

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                body = {
                    "answer": "The materials include wool yarn and a tapestry needle.",
                    "citation_ids": ["E1"],
                    "abstain": False,
                    "abstention_reason": None,
                }
                return json.dumps({"response": json.dumps(body)}).encode("utf-8")

        with patch(
            "digital_product_rag.answering.answer_generator.urllib.request.urlopen",
            return_value=FakeResponse(),
        ):
            response = answer_question("Which yarn is listed?", [result], mode="ollama", model="local-test")
        self.assertEqual(response.status, "invalid_citations")
        self.assertTrue(response.abstained)
        self.assertEqual(response.answer, SAFE_ABSTENTION_MESSAGE)
        self.assertIn("tapestry", response.error)

    def test_evidence_mode_assigns_request_scoped_ids_without_http(self):
        image = replace(
            _result("evidence-image"),
            chunk_type="image_description",
            asset_id="asset-1",
            asset_path="data/assets/asset-1.png",
        )
        with patch("digital_product_rag.answering.answer_generator.urllib.request.urlopen") as request:
            response = answer_question("Which yarn is listed?", [image], mode="evidence")
        request.assert_not_called()
        self.assertEqual(response.status, "evidence_only")
        self.assertEqual(response.citations[0].reference, "E1")
        self.assertEqual(response.citations[0].document_id, "doc-1")
        self.assertEqual(response.citations[0].page_start, 1)
        self.assertEqual(response.citations[0].asset_id, "asset-1")
        self.assertEqual(response.citations[0].asset_path, "data/assets/asset-1.png")


if __name__ == "__main__":
    unittest.main()
