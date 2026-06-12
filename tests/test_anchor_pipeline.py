from anrag.anchors import mark_anchors
from anrag.chunking import semantic_chunking
from anrag.models import ParsedBlock
from anrag.ocr import _parse_paddle_result


def test_rule_based_anchor_detection_bilingual():
    chunks = [
        ParsedBlock(id="h1", text="AnchorRAG", page=1, kind="heading", level=1, hierarchy_path=["AnchorRAG"]),
        ParsedBlock(
            id="p1",
            text="AnchorRAG is defined as a structure-aware retrieval method.",
            page=1,
            kind="paragraph",
            parent_id="h1",
            hierarchy_path=["AnchorRAG"],
        ),
        ParsedBlock(
            id="h2",
            text="Algorithm 1 Anchor Retrieval",
            page=2,
            kind="heading",
            level=2,
            parent_id="h1",
            hierarchy_path=["AnchorRAG", "Algorithm 1 Anchor Retrieval"],
        ),
    ]
    result = mark_anchors(semantic_chunking("doc_test", chunks))
    anchor_types = {chunk.anchor_type for chunk in result if chunk.anchor_type}
    assert "TITLE" in anchor_types
    assert "DEFINITION" in anchor_types
    assert "METHOD" in anchor_types


def test_hierarchical_layout_chunking_keeps_parent_child_links():
    blocks = [
        ParsedBlock(id="h1", text="1 Introduction", page=1, kind="heading", level=1, hierarchy_path=["1 Introduction"]),
        ParsedBlock(
            id="p1",
            text="This section introduces AnchorRAG. It retrieves anchors before expanding context.",
            page=1,
            kind="paragraph",
            parent_id="h1",
            hierarchy_path=["1 Introduction"],
        ),
        ParsedBlock(
            id="t1",
            text="Metric | Value\nAccuracy | 0.90\nLatency | 120ms",
            page=1,
            kind="table",
            parent_id="h1",
            hierarchy_path=["1 Introduction"],
            metadata={"layout_role": "table"},
        ),
    ]

    chunks = semantic_chunking("doc_test", blocks, max_tokens=80)
    section = next(chunk for chunk in chunks if chunk.metadata["chunk_role"] == "section")
    children = [chunk for chunk in chunks if chunk.parent_id == section.id]
    assert {chunk.metadata["chunk_role"] for chunk in children} == {"prose", "table"}
    assert next(chunk for chunk in children if chunk.metadata["chunk_role"] == "table").metadata["table_preserved"]


def test_semantic_prose_chunking_splits_long_arguments_by_sentence():
    text = (
        "Anchor retrieval finds section-level signals. "
        "These anchors preserve hierarchy during expansion. "
        "The method improves context selection for long PDFs. "
        "Evaluation uses a plain RAG baseline. "
        "Latency is measured separately from answer quality. "
        "Production systems also need observability and versioning."
    )
    blocks = [
        ParsedBlock(id="h1", text="Method", page=1, kind="heading", level=1, hierarchy_path=["Method"]),
        ParsedBlock(id="p1", text=text, page=1, kind="paragraph", parent_id="h1", hierarchy_path=["Method"]),
    ]

    chunks = semantic_chunking(
        "doc_test",
        blocks,
        max_tokens=18,
        min_semantic_tokens=8,
        semantic_break_threshold=0.2,
    )
    prose = [chunk for chunk in chunks if chunk.metadata["chunk_role"] == "prose"]
    assert len(prose) > 1
    assert all(chunk.token_count <= 18 for chunk in prose)


def test_paddle_ocr_result_parser_supports_v3_and_legacy_shapes():
    v3 = [{"rec_texts": ["Anchor", "Retrieval"], "rec_scores": [0.99, 0.97]}]
    legacy = [[[[0, 0], [1, 0], [1, 1], [0, 1]], ("Tree Context", 0.95)]]

    assert _parse_paddle_result(v3, min_confidence=0.35).text == "Anchor\nRetrieval"
    assert _parse_paddle_result(legacy, min_confidence=0.35).text == "Tree Context"
