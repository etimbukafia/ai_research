"""The hand-authored Aster Cloud semantic catalog and document views."""

from __future__ import annotations

from typing import Any

from .models import CatalogEntry


def _entry(
    catalog_id: str,
    concept: str,
    concept_type: str,
    definition: str,
    *,
    entity: str,
    grain: str,
    maps_to: list[str],
    authority: str,
    time_basis: str = "case_time",
    synonyms: list[str] | None = None,
    allowed_joins: list[str] | None = None,
    source_ids: list[str] | None = None,
    preconditions: list[str] | None = None,
    do_not_use: list[str] | None = None,
) -> CatalogEntry:
    return CatalogEntry(
        catalog_id=catalog_id,
        concept=concept,
        concept_type=concept_type,
        definition=definition,
        synonyms=synonyms or [],
        entity=entity,
        grain=grain,
        maps_to=maps_to,
        allowed_joins=allowed_joins or [],
        time_basis=time_basis,
        authority=authority,
        valid_from="2026-01-01",
        source_ids=source_ids or [],
        preconditions=preconditions or [],
        do_not_use=do_not_use or [],
    )


def build_catalog() -> list[CatalogEntry]:
    """Build exactly 36 deterministic catalog records."""

    core = [
        _entry(
            "catalog-active-account",
            "active_account",
            "concept",
            "An account with active CRM status at the case time.",
            entity="account",
            grain="account",
            maps_to=["crm.accounts.status"],
            authority="crm",
            synonyms=["live account", "customer"],
            source_ids=["account-01"],
        ),
        _entry(
            "catalog-active-contract",
            "active_contract",
            "concept",
            "The contract whose effective dates include the case time.",
            entity="contract",
            grain="account_at_time",
            maps_to=["contracts.effective_from", "contracts.effective_to"],
            authority="contracts",
            synonyms=["current contract", "contract in force"],
            allowed_joins=["contract.account_id = account.account_id"],
            source_ids=["contract-02"],
        ),
        _entry(
            "catalog-billable-seats",
            "billable_seats",
            "metric",
            "The highest seat count in the invoice period when the active contract permits peak usage billing.",
            entity="subscription",
            grain="account_month",
            maps_to=["usage.monthly_peak_seats", "contracts.seat_billing_rule"],
            authority="contracts",
            synonyms=["used seats", "licensed seats", "seat usage"],
            time_basis="invoice_period",
            allowed_joins=[
                "subscription.account_id = usage.account_id",
                "subscription.contract_id = contract.contract_id",
            ],
            source_ids=["contract-02", "usage-01-2026-05"],
            preconditions=["an active contract", "a usage record for the period"],
            do_not_use=["support.ticket_count", "invoice.seat_count alone"],
        ),
        _entry(
            "catalog-usage-peak",
            "usage_peak",
            "metric",
            "The peak number of seats recorded in one account month.",
            entity="usage",
            grain="account_month",
            maps_to=["usage.peak_seats"],
            authority="usage",
            synonyms=["peak seats", "maximum seats"],
            time_basis="usage_period",
            source_ids=["usage-01-2026-05"],
        ),
        _entry(
            "catalog-invoice-period",
            "invoice_period",
            "concept",
            "The start and end dates printed on the invoice.",
            entity="invoice",
            grain="invoice",
            maps_to=["billing.invoices.period_start", "billing.invoices.period_end"],
            authority="billing",
            synonyms=["billing period", "invoice month"],
            source_ids=["invoice-01-2026-05"],
        ),
        _entry(
            "catalog-net-invoice-amount",
            "net_invoice_amount",
            "metric",
            "The invoice amount after invoice-level credits recorded by billing.",
            entity="invoice",
            grain="invoice",
            maps_to=["billing.invoices.amount"],
            authority="billing",
            synonyms=["net bill", "invoice total"],
            time_basis="invoice_period",
            source_ids=["invoice-01-2026-05"],
            do_not_use=["usage.peak_seats"],
        ),
        _entry(
            "catalog-service-credit",
            "service_credit",
            "policy",
            "A billing credit linked to an invoice and supported by contract or usage evidence.",
            entity="credit",
            grain="account_invoice",
            maps_to=["billing.credits"],
            authority="finance",
            synonyms=["billing credit", "credit"],
            source_ids=["policy-01", "policy-02"],
            preconditions=["an invoice", "supporting evidence"],
            do_not_use=["drafting a credit without review"],
        ),
        _entry(
            "catalog-contract-renewal",
            "contract_renewal",
            "concept",
            "The next date on which the active contract can renew.",
            entity="contract",
            grain="account",
            maps_to=["contracts.renewal_date"],
            authority="contracts",
            synonyms=["renewal date", "next renewal"],
            source_ids=["contract-02"],
        ),
        _entry(
            "catalog-account-health",
            "account_health",
            "concept",
            "A support view based on account status, open tickets, and recent usage.",
            entity="account",
            grain="account",
            maps_to=["crm.accounts.status", "support.tickets", "usage.monthly"],
            authority="support",
            synonyms=["customer health", "account state"],
            source_ids=["account-01", "ticket-01-1"],
            allowed_joins=["ticket.account_id = account.account_id"],
        ),
        _entry(
            "catalog-priority-ticket",
            "priority_ticket",
            "concept",
            "A support ticket whose priority is high and whose status is open.",
            entity="support_ticket",
            grain="ticket",
            maps_to=["support.tickets.priority", "support.tickets.status"],
            authority="support",
            synonyms=["urgent ticket", "high priority issue"],
            source_ids=["ticket-01-1"],
        ),
        _entry(
            "catalog-approved-plan-change",
            "approved_plan_change",
            "policy",
            "A plan change with an active contract and a human approval record.",
            entity="subscription",
            grain="account_change",
            maps_to=["billing.subscriptions", "policy.plan_change"],
            authority="sales_operations",
            synonyms=["approved upgrade", "plan change"],
            source_ids=["policy-03", "policy-04"],
            preconditions=["an active contract", "human approval"],
        ),
        _entry(
            "catalog-source-authority",
            "source_authority",
            "relationship",
            "The system that owns a business fact for a given concept.",
            entity="source",
            grain="concept_source",
            maps_to=["catalog.authority"],
            authority="platform",
            synonyms=["system of record", "authoritative source"],
            source_ids=["policy-07"],
        ),
    ]

    supporting_specs = [
        ("account_identifier", "field", "The stable ID for an account.", "account", "account", ["crm.accounts.account_id"]),
        ("contract_plan", "field", "The plan named by the active contract.", "contract", "account_at_time", ["contracts.plan"]),
        ("contract_seat_limit", "field", "The committed seat limit in the active contract.", "contract", "account_at_time", ["contracts.seat_limit"]),
        ("contract_billing_rule", "rule", "The contract rule that controls how usage becomes billable.", "contract", "account_at_time", ["contracts.seat_billing_rule"]),
        ("subscription_status", "field", "The subscription state at the case time.", "subscription", "account_at_time", ["billing.subscriptions.status"]),
        ("subscription_seats", "field", "The committed seats on a subscription.", "subscription", "account_at_time", ["billing.subscriptions.seats"]),
        ("invoice_seat_count", "field", "The seat count printed on an invoice.", "invoice", "invoice", ["billing.invoices.seat_count"]),
        ("invoice_amount", "field", "The amount printed on an invoice.", "invoice", "invoice", ["billing.invoices.amount"]),
        ("usage_active_users", "field", "The number of active users in a usage period.", "usage", "account_month", ["usage.active_users"]),
        ("usage_period", "field", "The dates that bound a usage record.", "usage", "account_month", ["usage.period_start", "usage.period_end"]),
        ("credit_amount", "field", "The proposed amount for a service credit.", "credit", "account_invoice", ["billing.credits.amount"]),
        ("credit_approval_limit", "rule", "A credit above 500 USD needs human approval.", "credit", "account_invoice", ["policy-01"]),
        ("credit_evidence", "rule", "A credit needs an invoice and supporting contract or usage evidence.", "credit", "account_invoice", ["policy-02"]),
        ("plan_change_target", "field", "The target plan or seat count for a plan change.", "subscription", "account_change", ["billing.subscriptions.plan"]),
        ("ticket_priority", "field", "The priority stored on a support ticket.", "support_ticket", "ticket", ["support.tickets.priority"]),
        ("ticket_status", "field", "The state stored on a support ticket.", "support_ticket", "ticket", ["support.tickets.status"]),
        ("ticket_account", "relationship", "A support ticket belongs to one account.", "support_ticket", "ticket", ["ticket.account_id"]),
        ("time_validity", "rule", "Use a record only when its validity range contains the case time.", "source", "record_at_time", ["policy-08"]),
        ("account_contract_link", "relationship", "An account links to its contract through account_id.", "account", "account_contract", ["contract.account_id = account.account_id"]),
        ("contract_subscription_link", "relationship", "A subscription links to its contract through contract_id.", "contract", "contract_subscription", ["subscription.contract_id = contract.contract_id"]),
        ("subscription_invoice_link", "relationship", "An invoice links to a subscription through subscription_id.", "subscription", "subscription_invoice", ["invoice.subscription_id = subscription.subscription_id"]),
        ("subscription_usage_link", "relationship", "Usage links to a subscription through subscription_id.", "subscription", "subscription_usage", ["usage.subscription_id = subscription.subscription_id"]),
        ("policy_missing_evidence", "negative_rule", "Do not draft an action when required evidence is missing.", "policy", "action", ["policy-02"]),
        ("forbidden_ticket_as_contract", "negative_rule", "Do not use a support ticket as authority for contract terms.", "source", "concept_source", ["policy-07"]),
    ]
    for number, (concept, concept_type, definition, entity, grain, maps_to) in enumerate(
        supporting_specs, start=1
    ):
        core.append(
            _entry(
                f"catalog-support-{number:02d}",
                concept,
                concept_type,
                definition,
                entity=entity,
                grain=grain,
                maps_to=maps_to,
                authority="platform" if concept_type == "negative_rule" else "source",
                source_ids=["policy-07"] if concept_type == "negative_rule" else [],
            )
        )
    return core


def _typed_text(entry: CatalogEntry) -> str:
    values = [
        entry.concept,
        entry.definition,
        "synonyms: " + ", ".join(entry.synonyms),
        f"entity: {entry.entity}",
        f"grain: {entry.grain}",
        "maps_to: " + ", ".join(entry.maps_to),
        "allowed_joins: " + ", ".join(entry.allowed_joins),
        f"time_basis: {entry.time_basis}",
        f"authority: {entry.authority}",
        "preconditions: " + ", ".join(entry.preconditions),
        "do_not_use: " + ", ".join(entry.do_not_use),
    ]
    return " | ".join(values)


def build_documents(condition: str) -> list[Any]:
    """Return one of the three context views used by the evaluator."""

    from .retrieval import RetrievalDocument

    entries = build_catalog()
    if condition == "semantic_catalog":
        return [
            RetrievalDocument(
                context_id=entry.catalog_id,
                condition=condition,
                kind="catalog",
                text=_typed_text(entry),
                source_ids=entry.source_ids,
                catalog_entry=entry,
                metadata={"concept_type": entry.concept_type, "authority": entry.authority},
            )
            for entry in entries
        ]
    if condition == "prose_rag":
        return [
            RetrievalDocument(
                context_id=f"prose-{entry.catalog_id.removeprefix('catalog-')}",
                condition=condition,
                kind="prose",
                text=(
                    f"{entry.concept.replace('_', ' ')} means {entry.definition} "
                    f"Teams often use the words {', '.join(entry.synonyms) or entry.concept} "
                    "when they discuss this topic."
                ),
                source_ids=entry.source_ids,
                metadata={"concept_type": entry.concept_type},
            )
            for entry in entries
        ]
    if condition == "raw_schema":
        return [
            RetrievalDocument(
                context_id=f"raw-{entry.catalog_id.removeprefix('catalog-')}",
                condition=condition,
                kind="raw_schema",
                text=(
                    f"Table {entry.entity}s has fields {', '.join(entry.maps_to)}. "
                    f"Field description: {entry.definition}"
                ),
                source_ids=entry.source_ids,
                metadata={"table": entry.entity},
            )
            for entry in entries
        ]
    raise ValueError(f"Unknown retrieval condition: {condition}")
