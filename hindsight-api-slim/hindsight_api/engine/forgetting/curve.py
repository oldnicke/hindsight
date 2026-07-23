"""Pure functions for Ebbinghaus-style memory decay.

Retrievability is deliberately computed at query time. Persisting a value that
changes continuously would require periodic writes to every memory and would
still be stale between sweeps.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp, log1p

UTC = timezone.utc


@dataclass(frozen=True)
class ForgettingSignal:
    """The explainable outputs of the forgetting curve."""

    retrievability: float
    signal: float
    boost: float


@dataclass(frozen=True)
class ReinforcementResult:
    stability_days: float
    retrievability_before: float


def compute_initial_stability(
    *, base_days: float, importance: float, proof_count: int, fact_type: str, min_days: float, max_days: float
) -> float:
    type_factor = {"world": 1.5, "experience": 1.0, "observation": 2.0}.get(fact_type, 1.0)
    importance_factor = 0.5 + min(1.0, max(0.0, importance))
    evidence_factor = 1.0 + min(log1p(max(0, proof_count)) / 4.0, 1.0)
    return min(max_days, max(min_days, base_days * type_factor * importance_factor * evidence_factor))


def reinforce_stability(
    *,
    now: datetime,
    last_reinforced_at: datetime,
    stability_days: float,
    reinforcement_count: int,
    source_weight: float,
    gain: float,
    max_days: float,
    minimum_spacing_quality: float = 0.05,
) -> ReinforcementResult:
    signal = compute_forgetting_signal(
        now=now,
        last_reinforced_at=last_reinforced_at,
        stability_days=stability_days,
        enabled=True,
        apply_to_ranking=False,
        score_floor=0.0,
        score_gamma=1.0,
        score_alpha=0.0,
    )
    spacing_quality = min(1.0, max(minimum_spacing_quality, 1.0 - signal.retrievability))
    effective_gain = gain * min(1.0, max(0.0, source_weight)) * spacing_quality / (1 + reinforcement_count) ** 0.5
    return ReinforcementResult(
        stability_days=min(max_days, max(stability_days, stability_days * (1.0 + effective_gain))),
        retrievability_before=signal.retrievability,
    )


def lifecycle_score(retrievability: float, importance: float) -> float:
    return min(1.0, max(0.0, retrievability)) * (0.5 + 0.5 * min(1.0, max(0.0, importance)))


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
