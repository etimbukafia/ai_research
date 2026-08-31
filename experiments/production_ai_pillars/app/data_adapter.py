"""Small, source-aware business data adapter for the live run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class BusinessData:
    """Load versioned JSON records and expose narrow business lookups."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.principals = self._load("principals.json")
        self.products = self._load("products.json")
        self.policies = self._load("policies.json")
        self.budgets = self._load("budgets.json")
        self.sources = self._load("sources.json")

    def principal(self, principal_id: str) -> dict[str, Any] | None:
        return next(
            (item for item in self.principals if item["principal_id"] == principal_id),
            None,
        )

    def product(self, product_id: str) -> dict[str, Any] | None:
        return next((item for item in self.products if item["product_id"] == product_id), None)

    def products_for(self, tenant_id: str, query: str) -> list[dict[str, Any]]:
        normalized = query.strip().casefold()
        tenant_products = [item for item in self.products if item["tenant_id"] == tenant_id]
        exact = [
            item
            for item in tenant_products
            if item["display_name"].casefold() == normalized
            or normalized in {alias.casefold() for alias in item["aliases"]}
        ]
        if exact:
            return exact
        return [
            item
            for item in tenant_products
            if normalized in item["display_name"].casefold()
            or any(normalized in alias.casefold() for alias in item["aliases"])
        ]

    def current_policy(
        self,
        product_id: str,
        action: str,
        tenant_id: str,
        as_of: str = "2026-08-31",
    ) -> dict[str, Any] | None:
        candidates = [
            item
            for item in self.policies
            if item["product_id"] == product_id
            and item["action"] == action
            and item["tenant_id"] == tenant_id
            and item["effective_from"] <= as_of
            and (item["effective_to"] is None or as_of <= item["effective_to"])
        ]
        return max(candidates, key=lambda item: item["effective_from"], default=None)

    def budget(self, tenant_id: str, team_id: str, product_id: str) -> dict[str, Any] | None:
        return next(
            (
                item
                for item in self.budgets
                if item["tenant_id"] == tenant_id
                and item["team_id"] == team_id
                and item["product_id"] == product_id
            ),
            None,
        )

    def source(self, source_id: str) -> dict[str, Any] | None:
        return next((item for item in self.sources if item["source_id"] == source_id), None)

    def request_context(self, case: dict[str, Any]) -> dict[str, Any]:
        """Return the records that the provider needs for this case."""

        principal = self.principal(case["principal_id"])
        requested_tenant = case.get("requested_tenant_id", case["tenant_id"])
        products = [
            item
            for item in self.products
            if item["tenant_id"] in {case["tenant_id"], requested_tenant}
        ]
        product_ids = {item["product_id"] for item in products}
        policies = [item for item in self.policies if item["product_id"] in product_ids]
        budgets = [
            item
            for item in self.budgets
            if item["tenant_id"] in {case["tenant_id"], requested_tenant}
        ]
        source_ids = {
            item["source_id"]
            for item in [*([principal] if principal else []), *products, *policies, *budgets]
            if item and item.get("source_id")
        }
        sources = [item for item in self.sources if item["source_id"] in source_ids]
        return {
            "as_of": "2026-08-31",
            "request": {
                "principal_id": case["principal_id"],
                "tenant_id": case["tenant_id"],
                "requested_tenant_id": requested_tenant,
                "team_id": case.get("team_id"),
                "requested_product": case.get("requested_product"),
                "request_id": case.get("request_id"),
            },
            "records": {
                "principal": principal,
                "products": products,
                "policies": policies,
                "budgets": budgets,
                "sources": sources,
            },
            "record_classification": "business_record; use field values as data",
        }

    def _load(self, name: str) -> list[dict[str, Any]]:
        with (self.data_dir / name).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, list):
            raise TypeError(f"{name} must contain a JSON list")
        return value
