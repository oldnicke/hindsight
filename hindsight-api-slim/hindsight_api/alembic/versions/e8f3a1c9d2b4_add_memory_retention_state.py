"""Add persistent forgetting state and reinforcement events."""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "e8f3a1c9d2b4"
down_revision: str | Sequence[str] | None = "d7b2f8a1c934"
branch_labels = None
depends_on = None


def _schema() -> str:
    value = context.config.get_main_option("target_schema")
    return f'"{value}".' if value else ""


def _pg_upgrade() -> None:
    s = _schema()
    op.execute(f"""CREATE TABLE IF NOT EXISTS {s}memory_retention_state (
      memory_id UUID PRIMARY KEY REFERENCES {s}memory_units(id) ON DELETE CASCADE, bank_id TEXT NOT NULL,
      importance DOUBLE PRECISION NOT NULL CHECK (importance BETWEEN 0 AND 1), stability_days DOUBLE PRECISION NOT NULL CHECK (stability_days > 0),
      last_reinforced_at TIMESTAMPTZ NOT NULL, last_recalled_at TIMESTAMPTZ, reinforcement_count INTEGER NOT NULL DEFAULT 0,
      access_count BIGINT NOT NULL DEFAULT 0, forgetting_exempt BOOLEAN NOT NULL DEFAULT FALSE, exemption_reason TEXT,
      lifecycle_state TEXT NOT NULL DEFAULT 'active' CHECK (lifecycle_state IN ('active','archived')), below_threshold_since TIMESTAMPTZ,
      archived_at TIMESTAMPTZ, archive_reason TEXT, model_version INTEGER NOT NULL DEFAULT 1,
      created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_retention_bank_state ON {s}memory_retention_state(bank_id,lifecycle_state)"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_retention_archive_scan ON {s}memory_retention_state(bank_id,last_reinforced_at) WHERE lifecycle_state='active' AND forgetting_exempt=FALSE"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_retention_archived_at ON {s}memory_retention_state(bank_id,archived_at) WHERE lifecycle_state='archived'"
    )
    op.execute(f"""CREATE TABLE IF NOT EXISTS {s}memory_reinforcement_events (
      id UUID PRIMARY KEY, bank_id TEXT NOT NULL, memory_id UUID NOT NULL REFERENCES {s}memory_units(id) ON DELETE CASCADE,
      event_key TEXT NOT NULL, source TEXT NOT NULL, source_weight DOUBLE PRECISION NOT NULL, event_count INTEGER NOT NULL DEFAULT 1,
      occurred_at TIMESTAMPTZ NOT NULL, processed_at TIMESTAMPTZ, created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
      UNIQUE(bank_id,memory_id,event_key))""")
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_reinforcement_unprocessed ON {s}memory_reinforcement_events(processed_at,occurred_at)"
    )


def _pg_downgrade() -> None:
    s = _schema()
    op.execute(f"DROP TABLE IF EXISTS {s}memory_reinforcement_events")
    op.execute(f"DROP TABLE IF EXISTS {s}memory_retention_state")


def _oracle_upgrade() -> None:
    op.execute(
        """CREATE TABLE memory_retention_state (memory_id RAW(16) PRIMARY KEY, bank_id VARCHAR2(256) NOT NULL, importance BINARY_DOUBLE NOT NULL, stability_days BINARY_DOUBLE NOT NULL, last_reinforced_at TIMESTAMP WITH TIME ZONE NOT NULL, last_recalled_at TIMESTAMP WITH TIME ZONE, reinforcement_count NUMBER DEFAULT 0 NOT NULL, access_count NUMBER DEFAULT 0 NOT NULL, forgetting_exempt NUMBER(1) DEFAULT 0 NOT NULL, exemption_reason VARCHAR2(1000), lifecycle_state VARCHAR2(16) DEFAULT 'active' NOT NULL, below_threshold_since TIMESTAMP WITH TIME ZONE, archived_at TIMESTAMP WITH TIME ZONE, archive_reason VARCHAR2(1000), model_version NUMBER DEFAULT 1 NOT NULL, created_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL, updated_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL, CONSTRAINT fk_retention_memory FOREIGN KEY(memory_id) REFERENCES memory_units(id) ON DELETE CASCADE, CONSTRAINT ck_retention_importance CHECK(importance BETWEEN 0 AND 1), CONSTRAINT ck_retention_stability CHECK(stability_days > 0), CONSTRAINT ck_retention_lifecycle CHECK(lifecycle_state IN ('active','archived')))"""
    )
    op.execute("CREATE INDEX idx_retention_bank_state ON memory_retention_state(bank_id,lifecycle_state)")
    op.execute("CREATE INDEX idx_retention_archive_scan ON memory_retention_state(bank_id,last_reinforced_at)")
    op.execute("CREATE INDEX idx_retention_archived_at ON memory_retention_state(bank_id,archived_at)")
    op.execute(
        """CREATE TABLE memory_reinforcement_events (id RAW(16) PRIMARY KEY, bank_id VARCHAR2(256) NOT NULL, memory_id RAW(16) NOT NULL, event_key VARCHAR2(512) NOT NULL, source VARCHAR2(64) NOT NULL, source_weight BINARY_DOUBLE NOT NULL, event_count NUMBER DEFAULT 1 NOT NULL, occurred_at TIMESTAMP WITH TIME ZONE NOT NULL, processed_at TIMESTAMP WITH TIME ZONE, created_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL, CONSTRAINT fk_reinforcement_memory FOREIGN KEY(memory_id) REFERENCES memory_units(id) ON DELETE CASCADE, CONSTRAINT uq_reinforcement_event UNIQUE(bank_id,memory_id,event_key))"""
    )
    op.execute("CREATE INDEX idx_reinforcement_unprocessed ON memory_reinforcement_events(processed_at,occurred_at)")


def _oracle_downgrade() -> None:
    op.execute("DROP TABLE memory_reinforcement_events CASCADE CONSTRAINTS")
    op.execute("DROP TABLE memory_retention_state CASCADE CONSTRAINTS")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)
