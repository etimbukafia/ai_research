from enterprise_agent_lab.catalog import build_documents
from enterprise_agent_lab.policies import PolicyEngine
from enterprise_agent_lab.retrieval import HybridRetriever
from enterprise_agent_lab.storage import SourceStore


def test_large_credit_passes_policy_but_requires_approval():
    store = SourceStore(":memory:")
    engine = PolicyEngine(store, account_scope="account-01", case_time="2026-05-31")
    checks, approval_required = engine.check_action(
        "service_credit",
        "account-01",
        amount=750,
        evidence=["invoice-01-2026-05", "contract-02"],
    )
    assert approval_required is True
    assert all(check.passed for check in checks)
    store.close()


def test_missing_credit_evidence_fails_policy():
    store = SourceStore(":memory:")
    engine = PolicyEngine(store, account_scope="account-01")
    checks, _ = engine.check_action("service_credit", "account-01", amount=100)
    assert any(check.policy_id == "policy-02" and not check.passed for check in checks)
    store.close()

