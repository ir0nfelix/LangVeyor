from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from src.config import Settings, get_settings
from src.db.session import build_session_factory


@pytest.fixture(scope="session")
def settings() -> Settings:
    return get_settings()


@pytest.fixture(scope="session")
async def db_engine(settings: Settings) -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(settings.database_url, poolclass=pool.NullPool)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture(scope="session")
async def session_factory(
    db_engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    return build_session_factory(db_engine)
