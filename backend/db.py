from collections.abc import AsyncGenerator
from pathlib import Path

from alembic.config import Config
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from alembic import command

engine: AsyncEngine | None = None
SessionLocal: async_sessionmaker[AsyncSession] | None = None


def run_migrations() -> None:
    ini_path = Path(__file__).resolve().parent / "alembic.ini"
    config = Config(str(ini_path))
    command.upgrade(config, "head")


def init_db(database_url: str) -> None:
    global engine, SessionLocal
    engine = create_async_engine(database_url, pool_pre_ping=True)
    SessionLocal = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )


async def close_db() -> None:
    global engine, SessionLocal
    if engine is not None:
        await engine.dispose()
    engine = None
    SessionLocal = None


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    if SessionLocal is None:
        raise RuntimeError("Database is not initialized")
    async with SessionLocal() as session:
        yield session
