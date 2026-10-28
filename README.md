# Multimodal-RAG

## Build product chunks and semantic search index

Use the repository's existing global Python interpreter. The first build may download
`sentence-transformers/all-MiniLM-L6-v2` to the normal Hugging Face cache outside this
repository. The index uses normalized Sentence Transformers vectors with FAISS inner
product, which produces cosine similarity scores.

```powershell
python scripts/build_chunks.py
python scripts/build_vector_index.py
python scripts/search_knowledge_base.py --query "Which materials do I need?" --top-k 5
```

The same scripts can be run from `scripts/` as `python build_chunks.py`,
`python build_vector_index.py`, and `python search_knowledge_base.py --query "..."`.
Edit `config/product_catalog.json` to maintain product titles, categories, versions,
and document groupings. Automatically created catalog entries start with category
`unknown` and `metadata_status` set to `needs_review`.

Search embeds both text chunks and genuine image descriptions as text. Image results
include their original asset reference; the raw image files are not embedded or searched
directly in this stage.

## Keyword, hybrid, and evidence-grounded search

The search CLI defaults to hybrid retrieval: normalized Sentence Transformers/FAISS
results are combined with BM25 keyword results using reciprocal rank fusion. Use
`--mode semantic` or `--mode keyword` to select one retriever, or `--compare` to print
all three rankings. Product, document, and chunk-type filters can be combined.

```powershell
python scripts/search_knowledge_base.py --query "worsted #4 yarn" --mode keyword --top-k 5
python scripts/search_knowledge_base.py --query "Which yarn size is recommended?" --compare --top-k 5
python scripts/search_knowledge_base.py --query "Which materials are listed?" --mode hybrid --product-id <product-id>
```

`scripts/answer_question.py` is the evidence-grounded answer CLI. Evidence-only mode is
the default, returns the actual retrieved passages, and does not call Ollama. For example,
the following command works from the repository root or, using `python answer_question.py`,
from the `scripts/` directory:

```powershell
python scripts/answer_question.py --query "Which materials do I need to make a sweater?" --answer-mode evidence --retrieval-mode hybrid --top-k 5
```

Optional generation uses only a local Ollama service. Set `RAG_GENERATION_MODEL` to an
already installed model name; the CLI validates JSON citations against the retrieved
chunks and falls back to the passages if generation fails validation.

```powershell
$env:RAG_GENERATION_MODEL = "qwen2.5:7b-instruct"
python scripts/answer_question.py --query "Which yarn size is recommended?" --answer-mode ollama --retrieval-mode hybrid
```

## Local Streamlit application and evaluation

Start the browser UI from the repository root. It uses the existing catalog, document
inventory, retrieval indexes, validated answer generator, and local image assets.
Conversation history stays in the browser session.

```powershell
streamlit run src/digital_product_rag/ui/app.py
```

Run the automated suite and compare retrieval on the labelled 50-question corpus:

```powershell
python -m unittest discover -s tests
python scripts/evaluate_rag.py --top-k 5
python scripts/evaluate_rag.py --k-values 1,3,5,10
```

Retrieval-only evaluation makes no Ollama calls. Optional local answer and faithfulness
evaluation requires `RAG_GENERATION_MODEL` to name a model already installed in the
local Ollama service; it never downloads one:

```powershell
python scripts/evaluate_rag.py --top-k 5 --answer-evaluation --answer-limit 10
```

The evaluation dataset and metric definitions are in `evaluation/README.md`. Generated
results and the readable report are written to `data/processed/evaluation_results.json`
and `data/processed/evaluation_report.md`.
