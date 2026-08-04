"""Regression coverage for reranker failures during recall."""

import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.cancellation import OperationCancelledError
from hindsight_api.config import (
    DEFAULT_RERANKER_REQUIRED,
    ENV_RERANKER_REQUIRED,
    HindsightConfig,
    clear_config_cache,
)
from hindsight_api.engine.response_models import MinScores


@pytest.fixture(autouse=True)
def _strict_reranker_by_default(monkeypatch):
    """Keep module tests isolated from local env and the process-wide config cache."""
    monkeypatch.setenv(ENV_RERANKER_REQUIRED, "true")
    clear_config_cache()
    yield
    clear_config_cache()


def test_reranker_required_defaults_on(monkeypatch):
    monkeypatch.delenv(ENV_RERANKER_REQUIRED, raising=False)

    assert HindsightConfig.from_env().reranker_required is DEFAULT_RERANKER_REQUIRED is True


def test_reranker_required_can_be_disabled(monkeypatch):
    monkeypatch.setenv(ENV_RERANKER_REQUIRED, "false")

    assert HindsightConfig.from_env().reranker_required is False


def test_reranker_required_defaults_for_legacy_constructor_input(monkeypatch):
    monkeypatch.delenv(ENV_RERANKER_REQUIRED, raising=False)
    legacy_values = vars(HindsightConfig.from_env()).copy()
    legacy_values.pop("reranker_required")

    assert HindsightConfig(**legacy_values).reranker_required is True


@pytest.mark.parametrize("value", ["", "yes", "tru", "disabled"])
def test_reranker_required_rejects_ambiguous_values(monkeypatch, value):
    monkeypatch.setenv(ENV_RERANKER_REQUIRED, value)

    with pytest.raises(ValueError, match=ENV_RERANKER_REQUIRED):
        HindsightConfig.from_env()


def test_reranker_required_is_static_server_config():
    assert "reranker_required" in HindsightConfig.get_static_fields()
    assert "reranker_required" not in HindsightConfig.get_configurable_fields()


@pytest.mark.asyncio
async def test_recall_reranker_error_does_not_raise_unbound_local(memory, request_context):
    """Recall must propagate the reranker's exception, not an UnboundLocalError."""
    bank_id = f"test_reranker_err_{datetime.now(timezone.utc).timestamp()}"

    try:
        await memory.retain_async(
            bank_id=bank_id,
            content="Paris is the capital of France",
            request_context=request_context,
        )

        # Simulate a reranker failure (e.g. Cohere API error on empty/small candidate set)
        rerank_mock = AsyncMock(side_effect=RuntimeError("reranker API error"))
        memory._cross_encoder_reranker._initialized = True  # skip ensure_initialized

        with patch.object(memory._cross_encoder_reranker, "rerank", rerank_mock):
            with pytest.raises(Exception, match="reranker API error"):
                await memory.recall_async(
                    bank_id=bank_id,
                    query="capital of France",
                    request_context=request_context,
                )

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_point", ["ensure_initialized", "rerank"])
async def test_optional_reranker_failure_falls_back_to_rrf(
    memory,
    request_context,
    monkeypatch,
    caplog,
    failure_point,
):
    bank_id = f"test_reranker_fallback_{failure_point}_{datetime.now(timezone.utc).timestamp()}"

    try:
        for content in (
            "Paris is the capital of France",
            "Lyon is a large city in France",
            "Berlin is the capital of Germany",
        ):
            await memory.retain_async(bank_id=bank_id, content=content, request_context=request_context)

        question_date = datetime.now(timezone.utc)
        rrf_result = await memory.recall_async(
            bank_id=bank_id,
            query="cities in France",
            question_date=question_date,
            reranking="rrf",
            request_context=request_context,
        )
        assert rrf_result.results

        monkeypatch.setenv(ENV_RERANKER_REQUIRED, "false")
        clear_config_cache()
        caplog.clear()
        caplog.set_level(logging.WARNING, logger="hindsight_api.engine.memory_engine")

        reranker = memory._cross_encoder_reranker
        reranker._initialized = failure_point == "rerank"
        failure = AsyncMock(side_effect=RuntimeError("private reranker payload"))
        otel_span = MagicMock()
        otel_tracer = MagicMock()
        otel_tracer.start_span.return_value = otel_span

        with (
            patch.object(reranker, failure_point, failure),
            patch("hindsight_api.tracing.get_tracer", return_value=otel_tracer),
        ):
            degraded_result = await memory.recall_async(
                bank_id=bank_id,
                query="cities in France",
                question_date=question_date,
                enable_trace=True,
                min_scores=MinScores(reranker=1.0),
                request_context=request_context,
            )

        assert [result.id for result in degraded_result.results] == [result.id for result in rrf_result.results]
        assert all(result.scores is not None and result.scores.reranker is None for result in degraded_result.results)

        assert degraded_result.trace is not None
        reranking_phase = next(
            phase for phase in degraded_result.trace["summary"]["phase_metrics"] if phase["phase_name"] == "reranking"
        )
        assert reranking_phase["details"] == {
            "reranker_type": "rrf-fallback",
            "candidates_reranked": len(degraded_result.results),
            "degraded": True,
            "error_type": "RuntimeError",
            "reranker_min_score_skipped": True,
        }

        span_attributes = {call.args[0]: call.args[1] for call in otel_span.set_attribute.call_args_list}
        assert span_attributes["hindsight.reranker_type"] == "rrf-fallback"
        assert span_attributes["hindsight.reranker_degraded"] is True
        assert span_attributes["hindsight.reranker_error_type"] == "RuntimeError"

        assert f"because {ENV_RERANKER_REQUIRED}=false" in caplog.text
        assert "private reranker payload" not in caplog.text
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_optional_reranker_does_not_swallow_cancellation(memory, request_context, monkeypatch):
    bank_id = f"test_reranker_cancel_{datetime.now(timezone.utc).timestamp()}"

    try:
        await memory.retain_async(
            bank_id=bank_id,
            content="Paris is the capital of France",
            request_context=request_context,
        )

        monkeypatch.setenv(ENV_RERANKER_REQUIRED, "false")
        clear_config_cache()
        memory._cross_encoder_reranker._initialized = True
        rerank_mock = AsyncMock(side_effect=OperationCancelledError("client disconnected"))

        with patch.object(memory._cross_encoder_reranker, "rerank", rerank_mock):
            with pytest.raises(OperationCancelledError, match="client disconnected"):
                await memory.recall_async(
                    bank_id=bank_id,
                    query="capital of France",
                    request_context=request_context,
                )
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_recall_reranker_init_error_does_not_raise_unbound_local(memory, request_context):
    """Same regression when ensure_initialized() raises (before pre_filtered_count is set)."""
    bank_id = f"test_reranker_init_err_{datetime.now(timezone.utc).timestamp()}"

    try:
        await memory.retain_async(
            bank_id=bank_id,
            content="Paris is the capital of France",
            request_context=request_context,
        )

        init_mock = AsyncMock(side_effect=RuntimeError("reranker init failed"))
        memory._cross_encoder_reranker._initialized = False

        with patch.object(memory._cross_encoder_reranker, "ensure_initialized", init_mock):
            with pytest.raises(Exception, match="reranker init failed"):
                await memory.recall_async(
                    bank_id=bank_id,
                    query="capital of France",
                    request_context=request_context,
                )

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
