# AnchorRAG Prototype

Research-oriented AnchorRAG framework for PDF documents, with a production-friendly module layout.

## Architecture

```text
PDF
  -> layout-first PDF parser
  -> heading/section/paragraph/sentence chunking
  -> table/figure/scanned-page aware chunking
  -> rule-based anchor detection
  -> SQLite tree store + FAISS vector index

Query
  -> optional Qwen query rewrite
  -> anchor retrieval
  -> tree expansion
  -> local chunk expansion
  -> budget selection
  -> Qwen answer generation
```

## Why SQLite + FAISS?

AnchorRAG needs tree traversal more than a plain vector database does. This prototype stores document structure in SQLite with `parent_id`, `prev_id`, `next_id`, and `hierarchy_path`, then uses FAISS only for dense retrieval. It is easy to upgrade later to PostgreSQL `ltree`, Neo4j, Qdrant, or a hybrid production service.

## PDF Chunking

PDF ingestion parses layout before chunking:

- Headings become section parent chunks.
- Paragraphs become child prose chunks and split by sentence/semantic breakpoints when long.
- Tables are preserved as atomic chunks when possible; large tables split by rows while repeating the header.
- Figures are represented as visual placeholders with page/bounding-box metadata.
- Scanned pages and visual crops are OCRed with PaddleOCR when `ANRAG_ENABLE_OCR=true`.
- Figures can include OCR text and optional vision captions when `ANRAG_ENABLE_VISION_CAPTION=true` and `ANRAG_VISION_MODEL` points to an Ollama vision model.
- Fixed token windows are used only as a fallback for oversized text units.

Every child chunk keeps `parent_id`, `hierarchy_path`, page range, block kind, chunk role, and optional bounding box metadata. This keeps retrieval precise while allowing AnchorRAG to expand back to section-level context.

## Run

Start Ollama separately if you want local generation:

```powershell
ollama serve
ollama pull qwen3:8b
```

Start the app:

```powershell
python -m uvicorn anrag.api:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

## CLI

```powershell
python -m anrag.cli ingest path\to\paper.pdf
python -m anrag.cli query "What is the method?" --mode anchor
python -m anrag.cli query "What is the method?" --mode baseline
```

## OCR And Vision

PaddleOCR is enabled by default:

```powershell
$env:ANRAG_ENABLE_OCR="true"
$env:ANRAG_OCR_LANG="en"
```

If your PaddleOCR models are local, configure them explicitly:

```powershell
$env:ANRAG_OCR_TEXT_DETECTION_MODEL_DIR="C:\path\to\detection_model"
$env:ANRAG_OCR_TEXT_RECOGNITION_MODEL_DIR="C:\path\to\recognition_model"
```

Optional figure captioning needs an Ollama vision model:

```powershell
$env:ANRAG_ENABLE_VISION_CAPTION="true"
$env:ANRAG_VISION_MODEL="qwen3-vl:8b"
```

Without a vision model, figures and scanned pages still contribute OCR text, page ranges, bounding boxes, and visual crop previews.

## Notes

- Default embedding mode is `auto`: try a multilingual sentence-transformer model, then fall back to a local hashing encoder if model loading fails.
- Anchor detection is rule-based.
- Query rewrite and answer generation use Ollama with `qwen3:8b` by default.
- Cross-encoder reranking is intentionally optional and isolated for later experiments.
- OCR is enabled by default through PaddleOCR. If your PaddleOCR models are stored locally, point `ANRAG_OCR_TEXT_DETECTION_MODEL_DIR` and `ANRAG_OCR_TEXT_RECOGNITION_MODEL_DIR` at them.
- True image captioning requires a vision-capable Ollama model and is disabled by default.
