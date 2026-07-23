from dataclasses import dataclass
from datetime import datetime


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
