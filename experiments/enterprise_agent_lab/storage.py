"""SQLite source store for the local Aster Cloud systems."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, TypeVar

from .models import (
    Account,
    Contract,
    Invoice,
    PolicyRule,
    Subscription,
    SupportTicket,
    UsageRecord,
)
from .seed import seed_records

ModelT = TypeVar("ModelT")


class SourceStore:
    """A small read-oriented SQLite store with deterministic seed data."""

    _record_models = {
        "accounts": (Account, "account_id"),
        "contracts": (Contract, "contract_id"),
        "subscriptions": (Subscription, "subscription_id"),
        "invoices": (Invoice, "invoice_id"),
        "usage": (UsageRecord, "usage_id"),
        "tickets": (SupportTicket, "ticket_id"),
        "policies": (PolicyRule, "policy_id"),
    }

    def __init__(self, path: str | Path = ":memory:", seed: bool = True) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # PydanticAI may run synchronous tools in a worker thread. The store
        # remains read-oriented, and the agent calls tools in sequence.
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._create_schema()
        if seed:
            self.seed()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS source_records (
                record_type TEXT NOT NULL,
                record_id TEXT PRIMARY KEY,
                account_id TEXT,
                payload TEXT NOT NULL,
                valid_from TEXT,
                valid_to TEXT,
                authority TEXT,
                source_id TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_source_type_account
                ON source_records(record_type, account_id);
            CREATE INDEX IF NOT EXISTS idx_source_validity
                ON source_records(valid_from, valid_to);
            """
        )
        self.connection.commit()

    def seed(self) -> None:
        self.connection.execute("DELETE FROM source_records")
        for record_type, records in seed_records().items():
            for record in records:
                payload = record.model_dump(mode="json")
                record_id_field = self._record_models[record_type][1]
                record_id = str(payload[record_id_field])
                if isinstance(record, Contract):
                    valid_from, valid_to, authority = (
                        record.effective_from,
                        record.effective_to,
                        record.authority,
                    )
                elif isinstance(record, Account):
                    valid_from, valid_to, authority = (
                        record.valid_from,
                        record.valid_to,
                        "crm",
                    )
                elif isinstance(record, Subscription):
                    valid_from, valid_to, authority = (
                        record.started_at,
                        record.ended_at,
                        "billing",
                    )
                elif isinstance(record, PolicyRule):
                    valid_from, valid_to, authority = (
                        record.valid_from,
                        record.valid_to,
                        record.authority,
                    )
                elif isinstance(record, Invoice):
                    valid_from, valid_to, authority = (
                        record.period_start,
                        record.period_end,
                        "billing",
                    )
                elif isinstance(record, UsageRecord):
                    valid_from, valid_to, authority = (
                        record.period_start,
                        record.period_end,
                        "usage",
                    )
                else:
                    valid_from, valid_to, authority = (
                        record.created_at,
                        None,
                        "support",
                    )
                account_id = payload.get("account_id")
                self.connection.execute(
                    """
                    INSERT INTO source_records
                    (record_type, record_id, account_id, payload, valid_from, valid_to,
                     authority, source_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record_type,
                        record_id,
                        account_id,
                        json.dumps(payload, sort_keys=True),
                        valid_from,
                        valid_to,
                        authority,
                        payload["source_id"],
                    ),
                )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def _parse_rows(self, record_type: str, rows: list[sqlite3.Row]) -> list[Any]:
        model, _ = self._record_models[record_type]
        return [model.model_validate(json.loads(row["payload"])) for row in rows]

    def _rows(
        self,
        record_type: str,
        *,
        account_id: str | None = None,
        record_id: str | None = None,
        at: str | None = None,
    ) -> list[sqlite3.Row]:
        clauses = ["record_type = ?"]
        values: list[Any] = [record_type]
        if account_id:
            clauses.append("account_id = ?")
            values.append(account_id)
        if record_id:
            clauses.append("record_id = ?")
            values.append(record_id)
        if at:
            clauses.extend(
                [
                    "(valid_from IS NULL OR valid_from <= ?)",
                    "(valid_to IS NULL OR valid_to >= ?)",
                ]
            )
            values.extend([at, at])
        query = (
            "SELECT * FROM source_records WHERE "
            + " AND ".join(clauses)
            + " ORDER BY record_id"
        )
        return list(self.connection.execute(query, values))

    def find_account(
        self, *, account_id: str | None = None, name: str | None = None
    ) -> list[Account]:
        if account_id:
            rows = self._rows("accounts", record_id=account_id)
        else:
            rows = list(
                self.connection.execute(
                    "SELECT * FROM source_records WHERE record_type = 'accounts' "
                    "AND lower(json_extract(payload, '$.name')) LIKE lower(?) "
                    "ORDER BY record_id",
                    (f"%{name or ''}%",),
                )
            )
        return self._parse_rows("accounts", rows)

    def get_active_contract(self, account_id: str, at: str) -> Contract | None:
        rows = self._rows("contracts", account_id=account_id, at=at)
        return self._parse_rows("contracts", rows)[-1] if rows else None

    def get_subscription(
        self, account_id: str, subscription_id: str | None = None, at: str | None = None
    ) -> list[Subscription]:
        rows = self._rows(
            "subscriptions", account_id=account_id, record_id=subscription_id, at=at
        )
        return self._parse_rows("subscriptions", rows)

    def get_invoice(
        self,
        account_id: str,
        invoice_id: str | None = None,
        period_start: str | None = None,
        period_end: str | None = None,
    ) -> list[Invoice]:
        rows = self._rows("invoices", account_id=account_id, record_id=invoice_id)
        invoices = self._parse_rows("invoices", rows)
        return [
            invoice
            for invoice in invoices
            if (period_start is None or invoice.period_start == period_start)
            and (period_end is None or invoice.period_end == period_end)
        ]

    def get_usage_record(
        self,
        account_id: str,
        usage_id: str | None = None,
        period_start: str | None = None,
        period_end: str | None = None,
    ) -> list[UsageRecord]:
        rows = self._rows("usage", account_id=account_id, record_id=usage_id)
        usage = self._parse_rows("usage", rows)
        return [
            record
            for record in usage
            if (period_start is None or record.period_start == period_start)
            and (period_end is None or record.period_end == period_end)
        ]

    def get_support_tickets(
        self, account_id: str, priority: str | None = None, status: str | None = None
    ) -> list[SupportTicket]:
        rows = self._rows("tickets", account_id=account_id)
        tickets = self._parse_rows("tickets", rows)
        return [
            ticket
            for ticket in tickets
            if (priority is None or ticket.priority == priority)
            and (status is None or ticket.status == status)
        ]

    def get_policy_rules(self, action: str | None = None, at: str | None = None) -> list[PolicyRule]:
        rows = self._rows("policies", at=at)
        policies = self._parse_rows("policies", rows)
        return [policy for policy in policies if action in (None, "any") or policy.action in {action, "any"}]

    def source_counts(self) -> dict[str, int]:
        rows = self.connection.execute(
            "SELECT record_type, COUNT(*) AS count FROM source_records GROUP BY record_type"
        )
        return {row["record_type"]: int(row["count"]) for row in rows}

    def all_source_ids(self) -> list[str]:
        rows = self.connection.execute("SELECT source_id FROM source_records ORDER BY source_id")
        return [str(row["source_id"]) for row in rows]
