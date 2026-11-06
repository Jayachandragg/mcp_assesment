import asyncio
import hashlib
import math
import sqlite3
import time
from contextlib import closing
from pathlib import Path


DATABASE = Path(__file__).resolve().parent / "rate_limits.db"
WINDOW_SECONDS = 60
TOKEN_LIMIT = 50_000


class RateLimitExceeded(Exception):
    def __init__(self, retry_after: int):
        super().__init__("Token rate limit exceeded")
        self.retry_after = retry_after


def hash_key(api_key: str) -> str:
    # API keys are secrets, so only a hash goes in the database.
    return hashlib.sha256(api_key.encode()).hexdigest()


class RateLimiter:
    """
    Sliding-window token limiter backed by SQLite.

    Tokens get reserved before the model call and corrected once the real usage
    is known. If we only counted afterwards, a burst of parallel requests could
    all pass the check and go over the limit together.

    BEGIN IMMEDIATE keeps the check and the insert together, even with several
    gateway workers on one file. An asyncio.Lock would only cover one process.
    sqlite3 calls block, so they run in worker threads.
    """

    def __init__(
        self,
        database: str | Path = DATABASE,
        limit: int = TOKEN_LIMIT,
        window: int = WINDOW_SECONDS,
        clock=time.time,
    ):
        self.database = database
        self.limit = limit
        self.window = window
        self.clock = clock

        with closing(self.connect()) as connection:
            # WAL so reads don't get stuck behind writes.
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS token_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_key TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    tokens INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_token_usage_tenant
                ON token_usage (tenant_key, timestamp)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_token_usage_timestamp
                ON token_usage (timestamp)
                """
            )

    def connect(self) -> sqlite3.Connection:
        # isolation_level=None, because transactions are started by hand.
        return sqlite3.connect(self.database, timeout=5, isolation_level=None)

    async def reserve(self, api_key: str, tokens: int) -> int:
        return await asyncio.to_thread(self.reserve_sync, hash_key(api_key), tokens)

    async def reconcile(self, reservation_id: int, actual_tokens: int) -> None:
        await asyncio.to_thread(self.reconcile_sync, reservation_id, actual_tokens)

    async def release(self, reservation_id: int) -> None:
        await asyncio.to_thread(self.release_sync, reservation_id)

    def reserve_sync(self, tenant_key: str, tokens: int) -> int:
        now = self.clock()

        with closing(self.connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")

            # Clear expired rows for everyone, or idle tenants would leave
            # theirs behind forever.
            connection.execute(
                "DELETE FROM token_usage WHERE timestamp <= ?",
                (now - self.window,),
            )

            used, oldest = connection.execute(
                """
                SELECT COALESCE(SUM(tokens), 0), MIN(timestamp)
                FROM token_usage
                WHERE tenant_key = ?
                """,
                (tenant_key,),
            ).fetchone()

            # Over the limit, so tell the caller when to retry.
            if used + tokens > self.limit:
                connection.execute("COMMIT")

                waited = self.window if oldest is None else oldest + self.window - now
                raise RateLimitExceeded(retry_after=max(1, math.ceil(waited)))

            cursor = connection.execute(
                """
                INSERT INTO token_usage (tenant_key, timestamp, tokens)
                VALUES (?, ?, ?)
                """,
                (tenant_key, now, tokens),
            )
            connection.execute("COMMIT")

            return cursor.lastrowid

    def reconcile_sync(self, reservation_id: int, actual_tokens: int) -> None:
        with closing(self.connect()) as connection:
            connection.execute(
                "UPDATE token_usage SET tokens = ? WHERE id = ?",
                (actual_tokens, reservation_id),
            )

    def release_sync(self, reservation_id: int) -> None:
        with closing(self.connect()) as connection:
            connection.execute(
                "DELETE FROM token_usage WHERE id = ?",
                (reservation_id,),
            )
