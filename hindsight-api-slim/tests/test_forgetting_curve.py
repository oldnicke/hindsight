"""Tests for opt-in Ebbinghaus retrievability scoring."""

from datetime import datetime, timedelta, timezone
from math import exp

from hindsight_api.engine.forgetting import (
    compute_forgetting_signal,
    compute_initial_stability,
    lifecycle_score,
    reinforce_stability,
)
from hindsight_api.engine.search.reranking import apply_combined_scoring
from hindsight_api.engine.search.types import MergedCandidate, RetrievalResult, ScoredResult

UTC = timezone.utc
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _scored(*, created_at: datetime | None, score: float = 0.5) -> ScoredResult:
    return ScoredResult(
        candidate=MergedCandidate(
            retrieval=RetrievalResult(id="memory", text="memory", fact_type="world", created_at=created_at),
            rrf_score=0.1,
        ),
        cross_encoder_score_normalized=score,
    )


def test_curve_is_e_to_minus_one_after_one_stability_period() -> None:
    result = compute_forgetting_signal(
        now=NOW,
        last_reinforced_at=NOW - timedelta(days=30),
        stability_days=30,
        enabled=True,
        apply_to_ranking=True,
        score_floor=0.2,
        score_gamma=1.0,
        score_alpha=0.2,
    )

    assert abs(result.retrievability - exp(-1)) < 1e-12
    assert result.boost < 1.0


def test_observe_mode_reports_decay_without_changing_score() -> None:
    scored = _scored(created_at=NOW - timedelta(days=90))
    apply_combined_scoring(
        [scored],
        now=NOW,
        forgetting_enabled=True,
        forgetting_apply_to_ranking=False,
        forgetting_base_stability_days=30,
    )

    assert scored.retrievability < 0.1
    assert scored.forgetting_boost == 1.0
    assert scored.weight == 0.5


def test_rank_mode_penalizes_old_memory_but_keeps_nonzero_score() -> None:
    scored = _scored(created_at=NOW - timedelta(days=90))
    apply_combined_scoring(
        [scored],
        now=NOW,
        forgetting_enabled=True,
        forgetting_apply_to_ranking=True,
        forgetting_base_stability_days=30,
        forgetting_score_floor=0.2,
        forgetting_score_alpha=0.2,
    )

    assert 0 < scored.weight < 0.5


def test_missing_state_is_neutral_for_gradual_rollout() -> None:
    scored = _scored(created_at=None)
    apply_combined_scoring([scored], now=NOW, forgetting_enabled=True, forgetting_apply_to_ranking=True)

    assert scored.retrievability == 1.0
    assert scored.forgetting_boost == 1.0
    assert scored.weight == 0.5


def test_feature_disabled_preserves_existing_scoring() -> None:
    old = _scored(created_at=NOW - timedelta(days=365))
    apply_combined_scoring([old], now=NOW, forgetting_enabled=False)

    assert old.forgetting_boost == 1.0
    assert old.weight == 0.5


def test_future_learning_timestamp_clamps_to_full_retrievability() -> None:
    result = compute_forgetting_signal(
        now=NOW,
        last_reinforced_at=NOW + timedelta(days=1),
        stability_days=30,
        enabled=True,
        apply_to_ranking=True,
        score_floor=0.2,
        score_gamma=1.0,
        score_alpha=0.2,
    )

    assert result.retrievability == 1.0


def test_initial_stability_uses_type_importance_and_evidence() -> None:
    world = compute_initial_stability(
        base_days=30, importance=0.7, proof_count=2, fact_type="world", min_days=1, max_days=3650
    )
    experience = compute_initial_stability(
        base_days=30, importance=0.5, proof_count=0, fact_type="experience", min_days=1, max_days=3650
    )
    assert world > experience


def test_reinforcement_has_spacing_and_diminishing_returns() -> None:
    first = reinforce_stability(
        now=NOW,
        last_reinforced_at=NOW - timedelta(days=30),
        stability_days=30,
        reinforcement_count=0,
        source_weight=0.8,
        gain=0.5,
        max_days=3650,
    )
    repeated = reinforce_stability(
        now=NOW,
        last_reinforced_at=NOW - timedelta(days=30),
        stability_days=30,
        reinforcement_count=9,
        source_weight=0.8,
        gain=0.5,
        max_days=3650,
    )
    assert first.stability_days > repeated.stability_days > 30


def test_lifecycle_score_protects_important_memories() -> None:
    assert lifecycle_score(0.1, 1.0) > lifecycle_score(0.1, 0.0)
