from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from experiments.personal_assistant.src.memory import (
    PersonalAssistantEpisodicMemory,
    PersonalAssistantMemory,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


DDCCategory = Literal[
    "person",
    "preference",
    "schedule_rule",
    "communication_rule",
    "decision_rule",
    "recurring_workflow",
    "commitment",
    "boundary",
    "project",
    "tool_procedure",
    "topic_concept",
]

DDCRiskLevel = Literal["low", "medium", "high"]
DDCReviewStatus = Literal["pending", "approved", "rejected"]


class DDCGap(BaseModel):
    missing_context: str = Field(description="The specific personal context the assistant lacked.")
    reason: str = Field(description="Why this context matters for future assistant behavior.")


class DDCReviewItem(BaseModel):
    review_id: str = Field(default_factory=lambda: f"ddc_{uuid4().hex[:12]}")
    profile_id: UUID
    category: DDCCategory
    risk: DDCRiskLevel = "medium"
    status: DDCReviewStatus = "pending"
    source_task: str
    missing_context: str
    proposed_memory: str
    reason: str
    session_id: Optional[UUID] = None
    created_at: str = Field(default_factory=_utc_now)
    updated_at: str = Field(default_factory=_utc_now)


class DDCValidatedContext(BaseModel):
    review_id: str
    category: DDCCategory
    approved_memory: str


class DDCCycleLog(BaseModel):
    cycle_id: str = Field(default_factory=lambda: f"cycle_{uuid4().hex[:12]}")
    review_id: str
    profile_id: UUID
    source_task: str
    category: DDCCategory
    action: Literal["approved", "rejected", "revised"]
    promoted_memory: Optional[str] = None
    created_at: str = Field(default_factory=_utc_now)


class DDCGapProposal(BaseModel):
    category: DDCCategory
    risk: DDCRiskLevel = "medium"
    missing_context: str
    proposed_memory: str
    reason: str


class DDCAnalysisResult(BaseModel):
    review_items: list[DDCGapProposal] = Field(default_factory=list)


class DDCReviewService:
    def __init__(self, db) -> None:
        self.db = db

    def create_review_items(
        self,
        *,
        profile_id: UUID,
        source_task: str,
        proposals: list[DDCGapProposal],
        session_id: Optional[UUID] = None,
    ) -> list[DDCReviewItem]:
        existing_pending = self.pending(profile_id)
        existing_keys = {
            _proposal_key(item.category, item.proposed_memory, item.source_task)
            for item in existing_pending
        }
        items = [
            DDCReviewItem(
                profile_id=profile_id,
                category=proposal.category,
                risk=proposal.risk,
                source_task=source_task,
                missing_context=proposal.missing_context,
                proposed_memory=proposal.proposed_memory,
                reason=proposal.reason,
                session_id=session_id,
            )
            for proposal in proposals
            if proposal.proposed_memory.strip()
            and _proposal_key(proposal.category, proposal.proposed_memory, source_task) not in existing_keys
        ]
        if items:
            self.db.insert_ddc_review_items(items)
        return items

    def pending(self, profile_id: UUID) -> list[DDCReviewItem]:
        return self.db.list_ddc_review_items(profile_id, status="pending")

    def approved(self, profile_id: UUID) -> list[DDCReviewItem]:
        return self.db.list_ddc_review_items(profile_id, status="approved")

    def show(self, profile_id: UUID, review_id: str) -> Optional[DDCReviewItem]:
        return self.db.get_ddc_review_item(profile_id, review_id)

    def approve(self, profile_id: UUID, review_id: str) -> DDCReviewItem:
        item = self._require_pending(profile_id, review_id)
        memory = self.db.get_personal_assistant_memory(profile_id)
        self._promote(memory, item)
        self.db.save_personal_assistant_memory(profile_id, memory)
        approved = self.db.set_ddc_review_status(profile_id, review_id, "approved")
        self.db.insert_ddc_cycle_log(
            DDCCycleLog(
                review_id=review_id,
                profile_id=profile_id,
                source_task=item.source_task,
                category=item.category,
                action="approved",
                promoted_memory=item.proposed_memory,
            )
        )
        self.db.increment_context_revision(profile_id)
        return approved

    def reject(self, profile_id: UUID, review_id: str) -> DDCReviewItem:
        item = self._require_pending(profile_id, review_id)
        rejected = self.db.set_ddc_review_status(profile_id, review_id, "rejected")
        self.db.insert_ddc_cycle_log(
            DDCCycleLog(
                review_id=review_id,
                profile_id=profile_id,
                source_task=item.source_task,
                category=item.category,
                action="rejected",
            )
        )
        return rejected

    def revise(self, profile_id: UUID, review_id: str, proposed_memory: str) -> DDCReviewItem:
        self._require_pending(profile_id, review_id)
        item = self.db.update_ddc_review_memory(profile_id, review_id, proposed_memory)
        memory = self.db.get_personal_assistant_memory(profile_id)
        self._promote(memory, item)
        self.db.save_personal_assistant_memory(profile_id, memory)
        approved = self.db.set_ddc_review_status(profile_id, review_id, "approved")
        self.db.insert_ddc_cycle_log(
            DDCCycleLog(
                review_id=review_id,
                profile_id=profile_id,
                source_task=item.source_task,
                category=item.category,
                action="revised",
                promoted_memory=proposed_memory,
            )
        )
        self.db.increment_context_revision(profile_id)
        return approved

    def logs(self, profile_id: UUID, limit: int = 20) -> list[DDCCycleLog]:
        return self.db.list_ddc_cycle_logs(profile_id, limit=limit)

    def _require_pending(self, profile_id: UUID, review_id: str) -> DDCReviewItem:
        item = self.db.get_ddc_review_item(profile_id, review_id)
        if item is None:
            raise ValueError(f"Unknown DDC review item: {review_id}")
        if item.status != "pending":
            raise ValueError(f"DDC review item is not pending: {review_id}")
        return item

    def _promote(self, memory: PersonalAssistantMemory, item: DDCReviewItem) -> None:
        text = item.proposed_memory.strip()
        if item.category in {
            "preference",
            "schedule_rule",
            "communication_rule",
            "decision_rule",
            "boundary",
            "person",
            "topic_concept",
        }:
            _append_unique(memory.semantic.preferences, f"{item.category}: {text}")
        elif item.category in {"recurring_workflow", "tool_procedure"}:
            _append_unique(memory.procedural.routines_that_worked, f"{item.category}: {text}")
        elif item.category in {"commitment", "project"}:
            memory.episodic.append(
                PersonalAssistantEpisodicMemory(
                    title=f"DDC {item.category}",
                    summary=text,
                    category="commitment" if item.category == "commitment" else "goal",
                    follow_ups=[text] if item.category == "commitment" else [],
                    salience=0.8 if item.risk == "high" else 0.6,
                )
            )
        memory.update(memory.render())


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _proposal_key(category: str, proposed_memory: str, source_task: str) -> tuple[str, str, str]:
    return (
        category.strip().lower(),
        " ".join(proposed_memory.lower().split()),
        " ".join(source_task.lower().split()),
    )
