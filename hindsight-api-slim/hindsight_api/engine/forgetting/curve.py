"""Pure functions for Ebbinghaus-style memory decay.

Retrievability is deliberately computed at query time. Persisting a value that
changes continuously would require periodic writes to every memory and would
still be stale between sweeps.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp

UTC = timezone.utc


@dataclass(frozen=True)
class ForgettingSignal:
    """The explainable outputs of the forgetting curve."""

    retrievability: float
    signal: float
    boost: float


def compute_forgetting_signal(
    *,
    now: datetime,
    last_reinforced_at: datetime | None,
    stability_days: float,
    enabled: bool,
    apply_to_ranking: bool,
    score_floor: float,
    score_gamma: float,
    score_alpha: float,
    exempt: bool = False,
) -> ForgettingSignal:
    """Compute ``R = exp(-elapsed_days / stability_days)`` and its rank boost.

    Missing state is neutral. This is important during gradual rollout: old
    rows must not be penalised before a future state backfill has established
    a meaningful learning/reinforcement timestamp.
    """
    if not enabled or exempt or last_reinforced_at is None:
        return ForgettingSignal(retrievability=1.0, signal=1.0, boost=1.0)

    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    if last_reinforced_at.tzinfo is None:
        last_reinforced_at = last_reinforced_at.replace(tzinfo=UTC)

    elapsed_days = max(0.0, (now - last_reinforced_at).total_seconds() / 86400.0)
    stability = max(stability_days, 1e-9)
    retrievability = min(1.0, max(0.0, exp(-elapsed_days / stability)))
    signal = score_floor + (1.0 - score_floor) * retrievability**score_gamma
    boost = 1.0 + score_alpha * (signal - 0.5) if apply_to_ranking else 1.0
    return ForgettingSignal(retrievability=retrievability, signal=signal, boost=boost)
