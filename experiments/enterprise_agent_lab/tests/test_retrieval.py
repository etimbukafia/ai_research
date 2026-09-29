from enterprise_agent_lab.catalog import build_documents
from enterprise_agent_lab.retrieval import HybridRetriever


def test_common_retrieval_interface_returns_typed_catalog_context():
    retriever = HybridRetriever(build_documents("semantic_catalog"))
    results = retriever.search_business_context("used seats and billing rule", top_k=8)
    assert results
    assert results[0].condition == "semantic_catalog"
    assert any(result.catalog_entry and result.catalog_entry.concept == "billable_seats" for result in results)


def test_retrieval_is_deterministic_and_supports_type_filter():
    documents = build_documents("prose_rag")
    first = HybridRetriever(documents).search_business_context("service credit", top_k=4)
    second = HybridRetriever(documents).search_business_context("service credit", top_k=4)
    assert [item.context_id for item in first] == [item.context_id for item in second]
    typed = HybridRetriever(build_documents("semantic_catalog")).search_business_context(
        "service credit", concept_type="policy", top_k=8
    )
    assert typed
    assert all(item.catalog_entry.concept_type == "policy" for item in typed)

