"""Policy gates that run outside the model."""

from __future__ import annotations

from typing import Iterable

from .models import EvidenceRef, PolicyCheck
from .storage import SourceStore


class ScopeDenied(RuntimeError):
    """Raised when a tool call leaves the configured account scope."""


class PolicyDenied(RuntimeError):
    """Raised when a draft action does not have the required evidence."""

    def __init__(self, message: str, checks: list[PolicyCheck] | None = None) -> None:
        super().__init__(message)
        self.checks = checks or []


class PolicyEngine:
    def __init__(
        self,
        store: SourceStore,
        *,
        account_scope: str | None = None,
        case_time: str = "2026-05-31",
    ) -> None:
        self.store = store
        self.account_scope = account_scope
        self.case_time = case_time

    def check_scope(self, account_id: str | None) -> PolicyCheck:
        passed = not self.account_scope or account_id == self.account_scope
        return PolicyCheck(
            policy_id="policy-10",
            name="Account scope",
            passed=passed,
            reason=(
                "account is inside the configured scope"
                if passed
                else f"account {account_id} is outside scope {self.account_scope}"
            ),
            source_id="policy-10",
        )

    def check_time(self, at: str | None = None) -> PolicyCheck:
        effective_time = at or self.case_time
        return PolicyCheck(
            policy_id="policy-08",
            name="Case time",
            passed=bool(effective_time),
            reason=f"records must be valid at {effective_time}",
            source_id="policy-08",
        )

    def check_action(
        self,
        action: str,
        account_id: str,
        *,
        amount: float | None = None,
        evidence: Iterable[EvidenceRef | str] = (),
        source_ids: Iterable[str] = (),
        at: str | None = None,
        reason: str | None = None,
        priority: str | None = None,
    ) -> tuple[list[PolicyCheck], bool]:
        """Return checks and approval state. Failed checks block the draft."""

        evidence_ids = {
            item.evidence_id if isinstance(item, EvidenceRef) else str(item) for item in evidence
        }
        sources = set(source_ids) | evidence_ids
        checks = [self.check_scope(account_id), self.check_time(at)]
        if action == "service_credit":
            contract = self.store.get_active_contract(account_id, at or self.case_time)
            allowed = bool(contract and contract.service_credit_allowed)
            checks.append(
                PolicyCheck(
                    policy_id="policy-02",
                    name="Service credit evidence",
                    passed=(
                        allowed
                        and any(item.startswith("invoice-") for item in sources)
                        and any(
                            item.startswith(("contract-", "usage-")) for item in sources
                        )
                    ),
                    reason=(
                        "invoice plus contract or usage evidence is present"
                        if allowed
                        and any(item.startswith("invoice-") for item in sources)
                        and any(item.startswith(("contract-", "usage-")) for item in sources)
                        else "service credit needs an active contract, invoice, and contract or usage evidence"
                    ),
                    source_id="policy-02",
                )
            )
            threshold = next(
                (
                    policy.threshold
                    for policy in self.store.get_policy_rules("service_credit", at or self.case_time)
                    if policy.policy_id == "policy-01"
                ),
                500.0,
            )
            requires_approval = bool(amount is not None and amount > threshold)
            checks.append(
                PolicyCheck(
                    policy_id="policy-01",
                    name="Service credit approval",
                    passed=True,
                    reason=(
                        f"credit amount {amount or 0:.2f} is within the approval limit"
                        if not requires_approval
                        else f"credit amount {amount or 0:.2f} needs human approval"
                    ),
                    approval_required=requires_approval,
                    source_id="policy-01",
                )
            )
        elif action == "plan_change":
            checks.append(
                PolicyCheck(
                    policy_id="policy-04",
                    name="Plan change contract",
                    passed=any(item.startswith("contract-") for item in sources),
                    reason=(
                        "active contract evidence is present"
                        if any(item.startswith("contract-") for item in sources)
                        else "plan change needs active contract evidence"
                    ),
                    source_id="policy-04",
                )
            )
            requires_approval = True
            checks.append(
                PolicyCheck(
                    policy_id="policy-03",
                    name="Plan change approval",
                    passed=True,
                    reason="plan changes need human approval",
                    approval_required=True,
                    source_id="policy-03",
                )
            )
        elif action == "support_ticket":
            has_reason = bool(reason and reason.strip())
            high_priority = priority == "high"
            checks.append(
                PolicyCheck(
                    policy_id="policy-05",
                    name="Priority ticket",
                    passed=has_reason,
                    reason="support reason is present" if has_reason else "support reason is required",
                    source_id="policy-05",
                )
            )
            requires_approval = high_priority
            checks.append(
                PolicyCheck(
                    policy_id="policy-06",
                    name="High priority review",
                    passed=True,
                    reason=(
                        "high priority ticket needs human review"
                        if high_priority
                        else "normal priority ticket does not need approval"
                    ),
                    approval_required=high_priority,
                    source_id="policy-06",
                )
            )
        else:
            requires_approval = False

        checks.append(
            PolicyCheck(
                policy_id="policy-07",
                name="Source authority",
                passed=not any(item.startswith("ticket-") for item in sources)
                or action == "support_ticket",
                reason=(
                    "source authority matches the requested action"
                    if not any(item.startswith("ticket-") for item in sources)
                    or action == "support_ticket"
                    else "support tickets cannot define contract or billing terms"
                ),
                source_id="policy-07",
            )
        )
        return checks, requires_approval

    @staticmethod
    def assert_allowed(checks: list[PolicyCheck]) -> None:
        failed = [check for check in checks if not check.passed]
        if failed:
            reason = "; ".join(f"{check.name}: {check.reason}" for check in failed)
            raise PolicyDenied(reason, checks)

