"""Streamlit AppTest smoke and integration tests for the local UI."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import streamlit  # noqa: F401
    from streamlit.testing.v1 import AppTest
except ImportError:  # pragma: no cover - only used when the optional UI dependency is absent
    AppTest = None

from digital_product_rag.ui import app as rag_app  # noqa: E402
from digital_product_rag.answering.answer_generator import (  # noqa: E402
    SAFE_ABSTENTION_MESSAGE,
    answer_question,
)
from digital_product_rag.retrieval.keyword_search import KeywordSearchIndex  # noqa: E402


@unittest.skipIf(AppTest is None, "Streamlit AppTest is not installed")
class StreamlitApplicationTests(unittest.TestCase):
    def _app(self):
        return AppTest.from_file(str(ROOT / "src" / "digital_product_rag" / "ui" / "app.py"), default_timeout=60)

    def test_app_initializes_with_actual_catalog_and_dynamic_product_options(self):
        app = self._app().run()
        self.assertEqual(app.exception, [])
        self.assertIn("Digital Product Assistant", [item.value for item in app.title])
        product_box = app.selectbox(key="product_filter")
        self.assertIsNone(product_box.value)
        self.assertIn("All products", product_box.options)
        self.assertTrue(any("product_5bac9b0d35c65bd4" in item for item in product_box.options))
        self.assertEqual(app.selectbox(key="chunk_type").value, "auto")
        self.assertEqual(len(app.chat_message), 0)
        self.assertEqual(len(app.get("imgs")), 0)
        self.assertTrue(any("ask a question to begin" in item.value for item in app.info))

    def test_initial_session_drops_legacy_answer_and_preview_state(self):
        app = self._app()
        app.session_state["chat_history"] = [{"query": "old query", "answer": "old answer", "display_evidence": [{"asset_path": "stale.png"}]}]
        app.session_state["last_answer"] = "old answer"
        app.session_state["selected_preview"] = "stale.png"
        app.session_state["preview_legacy"] = True
        app.run()
        self.assertEqual(app.exception, [])
        self.assertEqual(app.session_state["chat_history"], [])
        self.assertEqual(len(app.chat_message), 0)
        self.assertEqual(len(app.get("imgs")), 0)
        self.assertNotIn("last_answer", app.session_state)
        self.assertNotIn("selected_preview", app.session_state)
        self.assertNotIn("preview_legacy", app.session_state)

    def test_product_filter_chat_evidence_citations_previews_and_session_history(self):
        product_id = "product_5bac9b0d35c65bd4"
        app = self._app().run()
        app.selectbox(key="retrieval_mode").set_value("keyword")
        app.selectbox(key="product_filter").set_value(product_id)
        app.run()
        document_id = "c6ba34ec-a43f-50a2-b256-23cbcf22f2ea"
        document_box = app.selectbox(key="document_filter")
        self.assertTrue(any(document_id in item for item in document_box.options))
        document_box.set_value(document_id)
        app.chat_input[0].set_value("What yarn weight and fiber are recommended for the Christmas hanging ornaments?")
        app.run()
        self.assertEqual(app.exception, [])
        history = app.session_state["chat_history"]
        self.assertEqual(len(history), 1)
        entry = history[0]
        self.assertEqual(entry["status"], "evidence_only")
        self.assertEqual(entry["answer_mode"], "evidence")
        self.assertTrue(entry["retrieved_evidence"])
        self.assertTrue(entry["citations"])
        self.assertTrue(all(item["product_id"] == product_id for item in entry["retrieved_evidence"]))
        self.assertTrue(all(item["document_id"] == document_id for item in entry["retrieved_evidence"]))
        self.assertTrue(all(item["document_id"] and item["filename"] and item["source_pages"] and item["chunk_id"] for item in entry["retrieved_evidence"]))
        self.assertTrue(any("Evidence-only mode" in item.value for item in app.info))
        assets = [
            json.loads(line)
            for line in (ROOT / "data" / "processed" / "assets.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        _, assets_by_page = rag_app._asset_maps(assets)
        preview_paths = []
        for item in entry["retrieved_evidence"]:
            if item["chunk_type"] == "image_description":
                continue
            for page in item["source_pages"][: rag_app.PAGE_PREVIEW_LIMIT]:
                preview = next((
                    asset for asset in assets_by_page.get((item["document_id"], page), [])
                    if asset.get("asset_type") == "page_preview"
                ), None)
                if preview:
                    path, error = rag_app._safe_image_path(ROOT, preview)
                    if path and not error:
                        preview_paths.append((page, path))
        self.assertTrue(preview_paths)
        self.assertEqual(len(app.get("imgs")), 0)
        app.checkbox(key=f"preview_{entry['request_key']}").check().run()
        self.assertEqual(app.exception, [])
        self.assertGreater(len(app.get("imgs")), 0)
        self.assertLessEqual(len(app.get("imgs")), rag_app.PAGE_PREVIEW_LIMIT)
        app.run()
        self.assertEqual(len(app.session_state["chat_history"]), 1)
        app.chat_input[0].set_value("Which yarn is recommended for the ornaments?").run()
        self.assertEqual(app.exception, [])
        self.assertEqual(len(app.session_state["chat_history"]), 2)
        self.assertEqual(len(app.get("imgs")), 0, "Old opted-in previews must not reappear for a new answer.")
        self.assertEqual(len(app.checkbox), 1, "Only the latest submitted answer may enable previews.")

    def test_exact_question_answer_sources_and_opt_in_page_preview(self):
        query = "tell me in which product the panguien has and what are the costruction step names to make this panguine sweater tell me names only"
        product_id = "product_4a0a46e4b0e054f6"
        sequence_id = "c64779d1-7957-5075-ad21-6bab0dd90e2f"
        steps = [
            "Crochet garment", "Assemble garment", "Make ribbing", "Block garment",
            "Crochet appliqués", "Block appliqués", "Pin layout", "Sew decorations", "Embroider small details",
        ]

        def local_model_response(_query, citations, **_options):
            sequence = next(item for item in citations if item.chunk_id == sequence_id)
            title = next((item for item in citations if item.chunk_id == "99fb5e3a-d25b-5ce4-a43e-a121eb20e00f"), sequence)
            answer = f"{sequence.product_title} [{title.reference}]\n\n" + "\n".join(
                f"{number}. {step} [{sequence.reference}]" for number, step in enumerate(steps, 1)
            )
            return json.dumps({
                "answer": answer, "citation_ids": list(dict.fromkeys([title.reference, sequence.reference])),
                "abstain": False, "abstention_reason": None,
            })

        with patch.dict("os.environ", {"RAG_GENERATION_MODEL": "local-test"}), patch(
            "digital_product_rag.answering.answer_generator._ollama_generate", side_effect=local_model_response,
        ) as model:
            app = self._app().run()
            app.selectbox(key="retrieval_mode").set_value("keyword")
            app.selectbox(key="answer_mode").set_value("ollama")
            app.selectbox(key="product_filter").set_value(product_id)
            app.run()
            self.assertTrue(any("80f9fbf2-0e41-5db4-9ad6-edc6b4adeaa2" in item for item in app.selectbox(key="document_filter").options))
            app.chat_input[0].set_value(query).run()
            self.assertEqual(app.exception, [])
            entry = app.session_state["chat_history"][0]
            self.assertEqual(entry["status"], "answered", entry.get("diagnostic"))
            sequence = next(item for item in entry["citations"] if item["chunk_id"] == sequence_id)
            self.assertEqual(sequence["source_pages"], [2])
            self.assertEqual(sequence["filename"], "Beginner DIY Crochet Christmas Penguin Sweater PDF 1.pdf")
            self.assertEqual(len(app.get("imgs")), 0)
            answer_elements = list(app.chat_message[-1])
            answer_position = next(i for i, item in enumerate(answer_elements) if item.type == "markdown" and item.value == entry["answer"])
            citations_position = next(i for i, item in enumerate(answer_elements) if item.type == "markdown" and item.value == "**Sources cited in the answer**")
            evidence_position = next(i for i, item in enumerate(answer_elements) if item.type == "expander" and item.label.startswith("Supporting evidence"))
            self.assertLess(answer_position, citations_position)
            self.assertLess(citations_position, evidence_position)
            self.assertTrue(any(sequence["filename"] in item.value and "page(s) 2" in item.value for item in app.caption))
            app.checkbox(key=f"preview_{entry['request_key']}").check().run()
            self.assertEqual(app.exception, [])
            images = app.get("imgs")
            self.assertGreater(len(images), 0)
            captions = [image.caption for item in images for image in item.proto.imgs]
            self.assertTrue(any(sequence["filename"] in text and "page 2" in text for text in captions))
            self.assertTrue(all("Penguin Sweater" in text for text in captions))
            model.assert_called_once()

    def test_rejected_ollama_answer_shows_safe_abstention_and_keeps_retrieved_passages(self):
        product_id = "product_5bac9b0d35c65bd4"
        document_id = "c6ba34ec-a43f-50a2-b256-23cbcf22f2ea"
        query = "What yarn weight and fiber are recommended for the Christmas hanging ornaments?"
        keyword_index = KeywordSearchIndex(ROOT / "data" / "processed" / "chunks.jsonl")
        results = keyword_index.search(query, top_k=5, product_id=product_id, document_id=document_id)
        response = answer_question(query, results, mode="evidence")
        response = replace(
            response,
            answer=SAFE_ABSTENTION_MESSAGE,
            status="invalid_citations",
            abstained=True,
            error="synthetic invalid citation diagnostic",
            abstention_reason=SAFE_ABSTENTION_MESSAGE,
        )

        with patch(
            "digital_product_rag.application.query_knowledge_base",
            return_value=(results, response, (), None),
        ):
            app = self._app().run()
            app.selectbox(key="retrieval_mode").set_value("keyword")
            app.selectbox(key="product_filter").set_value(product_id)
            app.selectbox(key="document_filter").set_value(document_id)
            app.selectbox(key="answer_mode").set_value("ollama")
            app.chat_input[0].set_value(query)
            app.run()

        self.assertEqual(app.exception, [])
        entry = app.session_state["chat_history"][0]
        self.assertTrue(entry["abstained"])
        self.assertEqual(entry["answer"], SAFE_ABSTENTION_MESSAGE)
        self.assertEqual(entry["status"], "invalid_citations")
        self.assertTrue(entry["retrieved_evidence"])
        self.assertEqual(len(entry["display_evidence"]), len(results))
        self.assertEqual(entry["diagnostic"], "synthetic invalid citation diagnostic")
        self.assertTrue(any(SAFE_ABSTENTION_MESSAGE in item.value for item in app.warning))

    def test_missing_local_model_warns_and_keeps_the_evidence(self):
        product_id = "product_5bac9b0d35c65bd4"
        document_id = "c6ba34ec-a43f-50a2-b256-23cbcf22f2ea"
        query = "What yarn weight and fiber are recommended for the Christmas hanging ornaments?"
        keyword_index = KeywordSearchIndex(ROOT / "data" / "processed" / "chunks.jsonl")
        results = keyword_index.search(query, top_k=5, product_id=product_id, document_id=document_id)
        with patch.dict("os.environ", {"RAG_GENERATION_MODEL": ""}):
            response = answer_question(query, results, mode="ollama", model="")
            self.assertEqual(response.status, "ollama_unavailable")
            with patch(
                "digital_product_rag.application.query_knowledge_base",
                return_value=(results, response, (), None),
            ):
                app = self._app().run()
                app.selectbox(key="retrieval_mode").set_value("keyword")
                app.selectbox(key="product_filter").set_value(product_id)
                app.selectbox(key="document_filter").set_value(document_id)
                app.selectbox(key="answer_mode").set_value("ollama")
                app.chat_input[0].set_value(query)
                app.run()
        self.assertEqual(app.exception, [])
        entry = app.session_state["chat_history"][0]
        self.assertEqual(entry["status"], "ollama_unavailable")
        self.assertTrue(entry["retrieved_evidence"])
        self.assertTrue(any("Ollama is unavailable" in item.value for item in app.warning))

    def test_clear_history_button_clears_session_state(self):
        app = self._app().run()
        app.session_state["chat_history"] = [{
            "query": "kept in session",
            "answer_mode": "evidence",
            "retrieval_mode": "keyword",
            "scope_label": "All products",
            "status": "evidence_only",
            "answer": "Evidence-only mode.",
            "abstained": False,
            "citations": [],
            "retrieved_evidence": [],
            "display_evidence": [],
            "diagnostic": None,
        }]
        app.run()
        app.button(key="clear_history").click().run()
        self.assertEqual(app.exception, [])
        self.assertEqual(app.session_state["chat_history"], [])

    def test_existing_page_preview_is_resolved_and_missing_image_is_a_warning(self):
        assets = [
            json.loads(line)
            for line in (ROOT / "data" / "processed" / "assets.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        preview = next(item for item in assets if item.get("asset_type") == "page_preview" and item.get("status") == "ok")
        path, error = rag_app._safe_image_path(ROOT, preview)
        self.assertIsNone(error)
        self.assertTrue(path and path.is_file())

        script = (
            "import sys\n"
            f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
            "from pathlib import Path\n"
            "from digital_product_rag.ui import app as rag_app\n"
            f"rag_app._show_asset(Path({str(ROOT)!r}), {{'status': 'ok', 'asset_path': 'data/assets/missing-preview.png'}}, caption='test')\n"
        )
        missing = AppTest.from_string(script).run()
        self.assertEqual(missing.exception, [])
        self.assertTrue(any("missing" in item.value.lower() for item in missing.warning))

    def test_missing_configuration_and_invalid_index_have_actionable_ui_states(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            missing_config_script = (
                "import sys\n"
                f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
                "from pathlib import Path\n"
                "from digital_product_rag.ui import app as rag_app\n"
                f"rag_app.REPOSITORY_ROOT = Path({temporary!r})\n"
                "rag_app.main()\n"
            )
            missing_config = AppTest.from_string(missing_config_script).run()
            self.assertEqual(missing_config.exception, [])
            self.assertTrue(any("knowledge base could not be loaded" in item.value.lower() for item in missing_config.error))

        invalid_index_script = (
            "import sys\n"
            f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
            "from pathlib import Path\n"
            "from digital_product_rag.ui import app as rag_app\n"
            f"rag_app.REPOSITORY_ROOT = Path({str(ROOT)!r})\n"
            "def invalid_bundle(*args, **kwargs):\n"
            "    raise ValueError('synthetic stale index')\n"
            "rag_app._cached_index_bundle = invalid_bundle\n"
            "rag_app.main()\n"
        )
        invalid_index = AppTest.from_string(invalid_index_script).run()
        self.assertEqual(invalid_index.exception, [])
        self.assertTrue(any(
            "valid, current FAISS index" in item.value
            for item in invalid_index.sidebar.warning
        ))


if __name__ == "__main__":
    unittest.main()
