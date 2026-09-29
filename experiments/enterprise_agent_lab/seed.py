"""Deterministic synthetic Aster Cloud source records."""

from __future__ import annotations

from typing import Any

from .models import (
    Account,
    Contract,
    Invoice,
    PolicyRule,
    Subscription,
    SupportTicket,
    UsageRecord,
)


MONTHS = (
    ("2026-04-01", "2026-04-30"),
    ("2026-05-01", "2026-05-31"),
    ("2026-06-01", "2026-06-30"),
)


def seed_records() -> dict[str, list[Any]]:
    """Return the same records on every call."""

    account_names = (
        "Acme Robotics",
        "Beacon Labs",
        "Cedar Health",
        "Delta Freight",
        "Ember Studio",
        "Fjord Retail",
        "Grove Energy",
        "Harbor Legal",
    )
    accounts = [
        Account(
            account_id=f"account-{index:02d}",
            name=name,
            status="active",
            region=("Europe" if index in {1, 4, 7} else "North America"),
            owner=("Maya Chen" if index % 2 else "Jon Bell"),
            valid_from="2024-01-01",
            source_id=f"account-{index:02d}",
        )
        for index, name in enumerate(account_names, start=1)
    ]

    contract_accounts = {
        1: ("account-01", "Growth", 100, "peak usage above committed seats is billable"),
        2: ("account-01", "Growth", 100, "peak usage above committed seats is billable"),
        3: ("account-02", "Scale", 200, "peak usage above committed seats is billable"),
        4: ("account-02", "Scale", 250, "peak usage above committed seats is billable"),
        5: ("account-03", "Starter", 50, "committed seats are billed"),
        6: ("account-03", "Scale", 150, "peak usage above committed seats is billable"),
        7: ("account-04", "Growth", 100, "committed seats are billed"),
        8: ("account-04", "Growth", 120, "peak usage above committed seats is billable"),
        9: ("account-05", "Starter", 50, "committed seats are billed"),
        10: ("account-06", "Growth", 100, "committed seats are billed"),
        11: ("account-07", "Scale", 200, "peak usage above committed seats is billable"),
        12: ("account-08", "Starter", 50, "committed seats are billed"),
    }
    contracts: list[Contract] = []
    for number, (account_id, plan, seat_limit, rule) in contract_accounts.items():
        historical = number in {1, 3, 5, 7}
        contracts.append(
            Contract(
                contract_id=f"contract-{number:02d}",
                account_id=account_id,
                plan=plan,
                seat_limit=seat_limit + (10 if not historical and number in {4, 6, 8} else 0),
                seat_billing_rule=rule,
                effective_from="2025-01-01" if historical else "2026-04-01",
                effective_to="2026-03-31" if historical else None,
                renewal_date="2026-04-01" if historical else "2027-04-01",
                service_credit_allowed=True,
                source_id=f"contract-{number:02d}",
                authority="contracts",
            )
        )

    subscriptions = [
        Subscription(
            subscription_id=f"subscription-{number:02d}",
            account_id=contract.account_id,
            contract_id=contract.contract_id,
            plan=contract.plan,
            status="ended" if contract.effective_to else "active",
            seats=contract.seat_limit,
            started_at=contract.effective_from,
            ended_at=contract.effective_to,
            source_id=f"subscription-{number:02d}",
        )
        for number, contract in enumerate(contracts, start=1)
    ]

    current_subscriptions = {
        account.account_id: next(
            subscription
            for subscription in reversed(subscriptions)
            if subscription.account_id == account.account_id
        )
        for account in accounts
    }
    invoices: list[Invoice] = []
    usage: list[UsageRecord] = []
    for account_number, account in enumerate(accounts, start=1):
        subscription = current_subscriptions[account.account_id]
        for month_number, (start, end) in enumerate(MONTHS, start=1):
            special_peak = account_number == 1 and start == "2026-05-01"
            peak_seats = subscription.seats + (50 if special_peak else 5 * month_number)
            seat_count = subscription.seats
            invoice_number = f"{account_number:02d}-2026-{month_number + 3:02d}"
            invoices.append(
                Invoice(
                    invoice_id=f"invoice-{invoice_number}",
                    account_id=account.account_id,
                    subscription_id=subscription.subscription_id,
                    period_start=start,
                    period_end=end,
                    amount=float(seat_count * 100 + month_number * 25),
                    currency="USD",
                    seat_count=seat_count,
                    status="open" if month_number == 3 else "paid",
                    source_id=f"invoice-{invoice_number}",
                )
            )
            usage_id = f"usage-{account_number:02d}-2026-{month_number + 3:02d}"
            usage.append(
                UsageRecord(
                    usage_id=usage_id,
                    account_id=account.account_id,
                    subscription_id=subscription.subscription_id,
                    period_start=start,
                    period_end=end,
                    peak_seats=peak_seats,
                    active_users=max(1, peak_seats - 8),
                    source_id=usage_id,
                )
            )

    tickets = [
        SupportTicket(
            ticket_id=f"ticket-{account_number:02d}-{ticket_number}",
            account_id=f"account-{account_number:02d}",
            subject=(
                "Invoice seat mismatch"
                if ticket_number == 1
                else "Quarterly account review"
            ),
            priority="high" if ticket_number == 1 and account_number == 1 else "normal",
            status="open" if ticket_number == 1 else "closed",
            created_at="2026-05-20" if ticket_number == 1 else "2026-04-10",
            source_id=f"ticket-{account_number:02d}-{ticket_number}",
        )
        for account_number in range(1, 9)
        for ticket_number in (1, 2)
    ]

    policies = [
        PolicyRule(
            policy_id="policy-01",
            name="Service credit approval",
            action="service_credit",
            condition="credits above 500 USD need human approval",
            threshold=500,
            approval_required=True,
            authority="finance",
            valid_from="2026-01-01",
            source_id="policy-01",
        ),
        PolicyRule(
            policy_id="policy-02",
            name="Service credit evidence",
            action="service_credit",
            condition="a credit needs an invoice and a contract or usage record",
            approval_required=False,
            authority="finance",
            valid_from="2026-01-01",
            source_id="policy-02",
        ),
        PolicyRule(
            policy_id="policy-03",
            name="Plan change approval",
            action="plan_change",
            condition="a plan change needs human approval",
            approval_required=True,
            authority="sales_operations",
            valid_from="2026-01-01",
            source_id="policy-03",
        ),
        PolicyRule(
            policy_id="policy-04",
            name="Plan change contract",
            action="plan_change",
            condition="the active contract must be cited",
            approval_required=False,
            authority="contracts",
            valid_from="2026-01-01",
            source_id="policy-04",
        ),
        PolicyRule(
            policy_id="policy-05",
            name="Priority ticket",
            action="support_ticket",
            condition="high priority tickets need a support reason",
            approval_required=False,
            authority="support",
            valid_from="2026-01-01",
            source_id="policy-05",
        ),
        PolicyRule(
            policy_id="policy-06",
            name="High priority review",
            action="support_ticket",
            condition="high priority tickets need human review",
            threshold=1,
            approval_required=True,
            authority="support",
            valid_from="2026-01-01",
            source_id="policy-06",
        ),
        PolicyRule(
            policy_id="policy-07",
            name="Source authority",
            action="any",
            condition="contract terms come from the contracts source",
            approval_required=False,
            authority="contracts",
            valid_from="2026-01-01",
            source_id="policy-07",
        ),
        PolicyRule(
            policy_id="policy-08",
            name="Case time",
            action="any",
            condition="use records valid at the case time",
            approval_required=False,
            authority="policy",
            valid_from="2026-01-01",
            source_id="policy-08",
        ),
        PolicyRule(
            policy_id="policy-09",
            name="No source mutation",
            action="any",
            condition="the lab can create drafts but cannot change source records",
            approval_required=False,
            authority="platform",
            valid_from="2026-01-01",
            source_id="policy-09",
        ),
        PolicyRule(
            policy_id="policy-10",
            name="Account scope",
            action="any",
            condition="a tool call must stay inside the selected account scope",
            approval_required=False,
            authority="platform",
            valid_from="2026-01-01",
            source_id="policy-10",
        ),
    ]
    return {
        "accounts": accounts,
        "contracts": contracts,
        "subscriptions": subscriptions,
        "invoices": invoices,
        "usage": usage,
        "tickets": tickets,
        "policies": policies,
    }


def seed_counts() -> dict[str, int]:
    """Return the expected deterministic seed counts."""

    return {name: len(records) for name, records in seed_records().items()}

