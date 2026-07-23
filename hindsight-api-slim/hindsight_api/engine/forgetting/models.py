from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class RetentionState:
    memory_id: str
    importance: float
    stability_days: float
    last_reinforced_at: datetime
    reinforcement_count: int
    access_count: int
    forgetting_exempt: bool
    lifecycle_state: str
    below_threshold_since: datetime | None = None
    archived_at: datetime | None = None


@dataclass(frozen=True)
class LifecycleSweepResult:
    archived: int
    prune_memory_ids: list[str]


class RetentionDetails(BaseModel):
    memory_id: str
    importance: float
    stability_days: float
    last_reinforced_at: datetime
    last_recalled_at: datetime | None = None
    reinforcement_count: int
    access_count: int
    retrievability: float
    lifecycle_score: float
    lifecycle_state: str
    forgetting_exempt: bool
    exemption_reason: str | None = None
    below_threshold_since: datetime | None = None
    archived_at: datetime | None = None
    archive_reason: str | None = None


class RetentionPolicyUpdate(BaseModel):
    importance: float | None = Field(default=None, ge=0, le=1)
    stability_days: float | None = Field(default=None, gt=0)
    forgetting_exempt: bool | None = None
    exemption_reason: str | None = None


class ReinforceMemoryRequest(BaseModel):
    idempotency_key: str = Field(min_length=1, max_length=256)
    reason: str | None = Field(default=None, max_length=1000)


class ForgettingStats(BaseModel):
    active: int
    archived: int
    exempt: int
    pending_events: int
    below_threshold: int


class ArchivePreview(BaseModel):
    eligible_count: int
    sample_memory_ids: list[str]
