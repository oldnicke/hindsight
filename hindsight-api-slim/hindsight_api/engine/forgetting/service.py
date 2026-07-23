import logging
import uuid
from datetime import datetime, timedelta, timezone

from ..schema import fq_table
from .curve import compute_forgetting_signal, compute_initial_stability, lifecycle_score, reinforce_stability
from .models import LifecycleSweepResult, RetentionState

logger = logging.getLogger(__name__)
UTC = timezone.utc


def default_importance(fact_type: str) -> float:
    return {"world": 0.7, "experience": 0.5, "observation": 0.8}.get(fact_type, 0.5)


async def initialize_states(conn, bank_id: str, memory_ids: list[str], fact_types: list[str], config) -> None:
    for memory_id, fact_type in zip(memory_ids, fact_types, strict=True):
        importance = default_importance(fact_type)
        stability = compute_initial_stability(
            base_days=config.forgetting_base_stability_days,
            importance=importance,
            proof_count=0,
            fact_type=fact_type,
            min_days=config.forgetting_min_stability_days,
            max_days=config.forgetting_max_stability_days,
        )
        await conn.execute(
            f"""INSERT INTO {fq_table("memory_retention_state")}
            (memory_id, bank_id, importance, stability_days, last_reinforced_at)
            VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP) ON CONFLICT (memory_id) DO NOTHING""",
            memory_id,
            bank_id,
            importance,
            stability,
        )


async def load_states(conn, memory_ids: list[str], config=None) -> dict[str, RetentionState]:
    if not memory_ids:
        return {}
    rows = await conn.fetch(
        f"SELECT * FROM {fq_table('memory_retention_state')} WHERE memory_id = ANY($1::uuid[])",
        [uuid.UUID(value) for value in memory_ids],
    )
    states = {
        str(row["memory_id"]): RetentionState(
            memory_id=str(row["memory_id"]),
            importance=float(row["importance"]),
            stability_days=float(row["stability_days"]),
            last_reinforced_at=row["last_reinforced_at"],
            reinforcement_count=int(row["reinforcement_count"]),
            access_count=int(row["access_count"]),
            forgetting_exempt=bool(row["forgetting_exempt"]),
            lifecycle_state=row["lifecycle_state"],
            below_threshold_since=row["below_threshold_since"],
            archived_at=row["archived_at"],
        )
        for row in rows
    }
    missing = [memory_id for memory_id in memory_ids if memory_id not in states]
    if missing and config is not None:
        rows = await conn.fetch(
            f"SELECT id, bank_id, fact_type FROM {fq_table('memory_units')} WHERE id = ANY($1::uuid[])",
            [uuid.UUID(value) for value in missing],
        )
        by_bank: dict[str, list[tuple[str, str]]] = {}
        for row in rows:
            by_bank.setdefault(row["bank_id"], []).append((str(row["id"]), row["fact_type"]))
        for bank_id, entries in by_bank.items():
            await initialize_states(
                conn, bank_id, [entry[0] for entry in entries], [entry[1] for entry in entries], config
            )
        if rows:
            return await load_states(conn, memory_ids)
    return states


async def record_events(conn, bank_id: str, memory_ids: list[str], event_key: str, source: str, weight: float) -> None:
    now = datetime.now(UTC)
    for memory_id in dict.fromkeys(memory_ids):
        await conn.execute(
            f"""INSERT INTO {fq_table("memory_reinforcement_events")}
            (id, bank_id, memory_id, event_key, source, source_weight, occurred_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7) ON CONFLICT (bank_id, memory_id, event_key) DO NOTHING""",
            uuid.uuid4(),
            bank_id,
            memory_id,
            event_key,
            source,
            weight,
            now,
        )


async def process_events(conn, config) -> int:
    rows = await conn.fetch(
        f"""SELECT e.id, e.memory_id, e.source_weight, e.occurred_at, s.stability_days,
        s.last_reinforced_at, s.reinforcement_count FROM {fq_table("memory_reinforcement_events")} e
        JOIN {fq_table("memory_retention_state")} s ON s.memory_id=e.memory_id
        WHERE e.processed_at IS NULL ORDER BY e.memory_id, e.occurred_at LIMIT $1 FOR UPDATE OF e SKIP LOCKED""",
        config.forgetting_event_batch_size,
    )
    for row in rows:
        cooldown = timedelta(hours=config.forgetting_reinforcement_cooldown_hours)
        eligible = row["occurred_at"] - row["last_reinforced_at"] >= cooldown
        if eligible:
            result = reinforce_stability(
                now=row["occurred_at"],
                last_reinforced_at=row["last_reinforced_at"],
                stability_days=float(row["stability_days"]),
                reinforcement_count=int(row["reinforcement_count"]),
                source_weight=float(row["source_weight"]),
                gain=config.forgetting_reinforcement_gain,
                max_days=config.forgetting_max_stability_days,
            )
            await conn.execute(
                f"""UPDATE {fq_table("memory_retention_state")} SET stability_days=$2,
                last_reinforced_at=$3, last_recalled_at=$3, reinforcement_count=reinforcement_count+1,
                access_count=access_count+1, updated_at=CURRENT_TIMESTAMP WHERE memory_id=$1""",
                row["memory_id"],
                result.stability_days,
                row["occurred_at"],
            )
        else:
            await conn.execute(
                f"UPDATE {fq_table('memory_retention_state')} SET access_count=access_count+1, updated_at=CURRENT_TIMESTAMP WHERE memory_id=$1",
                row["memory_id"],
            )
        await conn.execute(
            f"UPDATE {fq_table('memory_reinforcement_events')} SET processed_at=CURRENT_TIMESTAMP WHERE id=$1",
            row["id"],
        )
    return len(rows)


async def lifecycle_sweep(conn, config, *, now: datetime | None = None, dry_run: bool = False) -> LifecycleSweepResult:
    now = now or datetime.now(UTC)
    rows = await conn.fetch(
        f"""SELECT * FROM {fq_table("memory_retention_state")} WHERE forgetting_exempt=FALSE
        AND importance < $1 ORDER BY last_reinforced_at LIMIT $2""",
        config.forgetting_protected_importance,
        config.forgetting_archive_batch_size,
    )
    archived = 0
    prune_memory_ids: list[str] = []
    for row in rows:
        if row["lifecycle_state"] == "active" and config.forgetting_archive_enabled:
            retention = compute_forgetting_signal(
                now=now,
                last_reinforced_at=row["last_reinforced_at"],
                stability_days=float(row["stability_days"]),
                enabled=True,
                apply_to_ranking=False,
                score_floor=0,
                score_gamma=1,
                score_alpha=0,
            ).retrievability
            below = lifecycle_score(retention, float(row["importance"])) < config.forgetting_archive_threshold
            since = row["below_threshold_since"]
            if below and since and now - since >= timedelta(days=config.forgetting_archive_grace_days):
                archived += 1
                if not dry_run:
                    await conn.execute(
                        f"UPDATE {fq_table('memory_retention_state')} SET lifecycle_state='archived', archived_at=$2, archive_reason='decay threshold', updated_at=$2 WHERE memory_id=$1",
                        row["memory_id"],
                        now,
                    )
            elif not dry_run:
                await conn.execute(
                    f"UPDATE {fq_table('memory_retention_state')} SET below_threshold_since=$2, updated_at=$3 WHERE memory_id=$1",
                    row["memory_id"],
                    (since or now) if below else None,
                    now,
                )
        elif (
            row["lifecycle_state"] == "archived"
            and config.forgetting_auto_prune_enabled
            and row["archived_at"]
            and now - row["archived_at"] >= timedelta(days=config.forgetting_prune_after_days)
        ):
            prune_memory_ids.append(str(row["memory_id"]))
    return LifecycleSweepResult(archived=archived, prune_memory_ids=prune_memory_ids)
