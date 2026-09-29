from pydantic import ValidationError
import pytest

from enterprise_agent_lab.catalog import build_catalog
from enterprise_agent_lab.models import AgentDecision
from enterprise_agent_lab.seed import seed_counts


def test_seed_and_catalog_sizes_are_deterministic():
    assert seed_counts() == {
        "accounts": 8,
        "contracts": 12,
        "subscriptions": 12,
        "invoices": 24,
        "usage": 24,
        "tickets": 16,
        "policies": 10,
    }
    assert len(build_catalog()) == 36


def test_decision_rejects_unknown_status_and_fields():
    with pytest.raises(ValidationError):
        AgentDecision(status="unknown", request="x")
    with pytest.raises(ValidationError):
        AgentDecision(status="answer", request="x", extra_field="bad")

