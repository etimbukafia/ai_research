"""Shared HTTP transport, cache, and retrieval result primitives."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

import requests
from pydantic import Field
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from experiments.drug_repurposing_agent.src.models import EvidenceItem, StrictModel

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "evidence_cache"


class RetrievalErrorRecord(StrictModel):
    """Explicit source retrieval failure; never evidence of lack of efficacy."""

    source: Literal["Open Targets", "PubMed"]
    request_hash: str = Field(min_length=64, max_length=64)
    query_parameters: dict[str, Any]
    retrieval_timestamp: datetime
    error_type: str = Field(min_length=1)
    message: str = Field(min_length=1)
    retryable: bool


class RetrievalResult(StrictModel):
    """Normalized evidence plus source errors and cache provenance."""

    pair_id: str = Field(pattern=r"^disease-drug-pair-\d{3}$")
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    errors: list[RetrievalErrorRecord] = Field(default_factory=list)
    request_hashes: list[str] = Field(default_factory=list)
    cache_hits: int = Field(ge=0, default=0)
    cache_misses: int = Field(ge=0, default=0)

    @property
    def materially_failed(self) -> bool:
        """Return whether source failure prevents a reliable evidence assessment."""

        return bool(self.errors) and not self.evidence_items

    @property
    def required_abstention_label(self) -> str | None:
        """Retrieval failure requires abstention, never an unsupported label."""

        return "insufficient_evidence" if self.materially_failed else None


class CachedHttpClient:
    """JSON HTTP client with bounded retries and immutable request-hash caching."""

    def __init__(
        self,
        *,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        session: requests.Session | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.timeout_seconds = timeout_seconds
        self.session = session or _retrying_session(max_retries)
        self.now = now or (lambda: datetime.now(timezone.utc))

    def get_json(
        self,
        *,
        source: Literal["Open Targets", "PubMed"],
        url: str,
        params: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        return self._request_json(
            source=source,
            method="GET",
            url=url,
            query_parameters=params,
            request_kwargs={"params": params},
        )

    def post_json(
        self,
        *,
        source: Literal["Open Targets", "PubMed"],
        url: str,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        return self._request_json(
            source=source,
            method="POST",
            url=url,
            query_parameters=payload,
            request_kwargs={"json": payload},
        )

    def get_text(
        self,
        *,
        source: Literal["Open Targets", "PubMed"],
        url: str,
        params: dict[str, Any],
    ) -> tuple[str | None, dict[str, Any]]:
        return self._request(
            source=source,
            method="GET",
            url=url,
            query_parameters=params,
            request_kwargs={"params": params},
            response_kind="text",
        )

    def _request_json(
        self,
        *,
        source: Literal["Open Targets", "PubMed"],
        method: str,
        url: str,
        query_parameters: dict[str, Any],
        request_kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        return self._request(
            source=source,
            method=method,
            url=url,
            query_parameters=query_parameters,
            request_kwargs=request_kwargs,
            response_kind="json",
        )

    def _request(
        self,
        *,
        source: Literal["Open Targets", "PubMed"],
        method: str,
        url: str,
        query_parameters: dict[str, Any],
        request_kwargs: dict[str, Any],
        response_kind: Literal["json", "text"],
    ) -> tuple[Any | None, dict[str, Any]]:
        request_hash = _request_hash(method, url, query_parameters)
        cache_path = self.cache_dir / source.lower().replace(" ", "_") / f"{request_hash}.json"
        if cache_path.exists():
            envelope = json.loads(cache_path.read_text(encoding="utf-8"))
            envelope["cache_hit"] = True
            response_key = "response" if response_kind == "json" else "response_text"
            return envelope.get(response_key), envelope

        timestamp = self.now()
        envelope: dict[str, Any] = {
            "source": source,
            "method": method,
            "url": url,
            "request_hash": request_hash,
            "query_parameters": query_parameters,
            "retrieval_timestamp": timestamp.isoformat(),
            "cache_hit": False,
        }
        try:
            response = self.session.request(
                method,
                url,
                timeout=self.timeout_seconds,
                **request_kwargs,
            )
            response.raise_for_status()
            if response_kind == "json":
                payload = response.json()
                envelope["response"] = payload
            else:
                payload = response.text
                envelope["response_text"] = payload
        except (requests.RequestException, ValueError) as exc:
            envelope["error"] = {
                "error_type": type(exc).__name__,
                "message": str(exc),
                "retryable": _is_retryable(exc),
            }
            payload = None

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(envelope, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return payload, envelope


def error_from_envelope(envelope: dict[str, Any]) -> RetrievalErrorRecord | None:
    error = envelope.get("error")
    if not error:
        return None
    return RetrievalErrorRecord(
        source=envelope["source"],
        request_hash=envelope["request_hash"],
        query_parameters=envelope["query_parameters"],
        retrieval_timestamp=envelope["retrieval_timestamp"],
        error_type=error["error_type"],
        message=error["message"],
        retryable=error["retryable"],
    )


def _retrying_session(max_retries: int) -> requests.Session:
    retry = Retry(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        status=max_retries,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "drug-repurposing-evidence-triage/1.0"})
    return session


def _request_hash(method: str, url: str, query_parameters: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"method": method, "url": url, "query_parameters": query_parameters},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code in {429, 500, 502, 503, 504}
    return isinstance(exc, (requests.ConnectionError, requests.Timeout))
