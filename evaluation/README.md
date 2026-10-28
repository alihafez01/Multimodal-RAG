# Retrieval evaluation

`datasets/rag_questions.jsonl` contains a repository-specific labelled sample curated from the current chunk dataset. Every answerable label is checked at run time against the current catalog and chunk IDs, document IDs, and pages; evaluation fails closed if those references drift. The unanswerable cases ask for private transaction or account details absent from their scoped PDF text.

Run from the repository root:

```powershell
python scripts/evaluate_rag.py --top-k 5
python scripts/evaluate_rag.py --k-values 1,3,5,10
```

The normal run evaluates semantic, keyword, and hybrid retrieval and makes no LLM calls. `Recall@K` is macro-averaged per answerable question, with recall equal to labelled relevant chunk IDs returned divided by all labelled relevant chunk IDs. `MRR@K` is averaged over answerable questions and assigns zero when no labelled chunk appears in the cutoff. Source-page hit rate checks whether any retrieved source page matches a labelled document/page reference. Citation-reference validity checks the retrieved chunk's canonical document, product, filename, path, pages, chunk type, and requested scope. Citation relevance is exact relevant chunk-ID precision on answerable questions only. Metrics include their question/reference denominators in the JSON output.

Unanswerable retrieval hits are reported as diagnostic passages, not as failed abstention tests. Abstention is evaluated only when generation actually runs. To optionally run a bounded generation and local Ollama-as-judge sample, configure `RAG_GENERATION_MODEL` to an already installed model and use:

```powershell
python scripts/evaluate_rag.py --top-k 5 --answer-evaluation --answer-limit 10
```

The evaluator checks the configured model against the local `/api/tags` endpoint and will not download one. The local model judge receives the answer and the exact cited evidence, has a strict JSON schema, and is advisory rather than ground truth. Its results and short review examples are stored in `data/processed/evaluation_results.json` and `evaluation_report.md`.
