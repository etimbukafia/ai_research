import pytest

from enterprise_agent_lab.catalog import build_documents
from enterprise_agent_lab.policies import PolicyDenied, PolicyEngine, ScopeDenied
from enterprise_agent_lab.retrieval import HybridRetriever
from enterprise_agent_lab.storage import SourceStore
from enterprise_agent_lab.tools import ToolRegistry


def make_registry(account_scope="account-01"):
    store = SourceStore(":memory:")
    retriever = HybridRetriever(build_documents("semantic_catalog"))
    policy = PolicyEngine(store, account_scope=account_scope, case_time="2026-05-31")
    return store, ToolRegistry(store, retriever, policy, account_scope=account_scope), policy


def test_registry_exposes_stable_tool_names_and_reads_contract():
    store, registry, _ = make_registry()
    assert registry.names() == [
        "find_account",
        "get_active_contract",
        "get_subscription",
        "get_invoice",
        "get_usage_record",
        "get_support_tickets",
        "search_business_concepts",
        "check_policy",
        "draft_service_credit",
        "draft_plan_change",
        "draft_support_ticket",
        "request_human_approval",
    ]
    result = registry.call(
        "get_active_contract", {"account_id": "account-01", "at": "2026-05-31"}
    )
    assert result.contract.contract_id == "contract-02"
    store.close()


def test_draft_requires_evidence_and_does_not_change_source_records():
    store, registry, _ = make_registry()
    before = store.source_counts()
    with pytest.raises(PolicyDenied):
        registry.call(
            "draft_service_credit",
            {
                "account_id": "account-01",
                "amount": 750,
                "reason": "billing mismatch",
                "evidence_ids": [],
            },
        )
    draft = registry.call(
        "draft_service_credit",
        {
            "account_id": "account-01",
            "amount": 750,
            "reason": "billing mismatch",
            "evidence_ids": ["invoice-01-2026-05", "contract-02"],
        },
    )
    assert draft.action.status == "pending_approval"
    assert draft.action.requires_approval is True
    assert store.source_counts() == before
    store.close()


def test_scope_gate_blocks_other_account():
    store, registry, _ = make_registry()
    with pytest.raises(ScopeDenied):
        registry.call("get_active_contract", {"account_id": "account-02"})
    store.close()

