from datetime import UTC, datetime
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

POSTGRES_NAMING_CONVENTION: dict[str, str] = {
    "ix": "idx_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "chk_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class UtcDateTime(sa.TypeDecorator[datetime]):
    """Store all timestamps in PostgreSQL TIMESTAMPTZ normalized to UTC."""

    impl = sa.TIMESTAMP(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: sa.Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: sa.Dialect) -> datetime | None:
        if value is None:
            return None
        return value.astimezone(UTC)


class Base(DeclarativeBase):
    """Declarative base shared across Domains 0-5."""

    metadata = sa.MetaData(naming_convention=POSTGRES_NAMING_CONVENTION)
    type_annotation_map = {datetime: UtcDateTime()}


class PkUUIDMixin:
    id: Mapped[UUID] = mapped_column(
        sa.Uuid,
        primary_key=True,
        default=uuid4,
        server_default=sa.text("gen_random_uuid()"),
    )


class CreatedMixin:
    created_at: Mapped[datetime] = mapped_column(
        server_default=sa.text("clock_timestamp()"),
        nullable=False,
    )


class DatedMixin(CreatedMixin):
    updated_at: Mapped[datetime] = mapped_column(
        server_default=sa.text("clock_timestamp()"),
        onupdate=sa.text("clock_timestamp()"),
        nullable=False,
    )


class ActivatableMixin:
    is_active: Mapped[bool] = mapped_column(
        sa.Boolean,
        default=True,
        server_default=sa.true(),
        nullable=False,
    )
