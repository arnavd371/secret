"""
SQLite persistence for the MVP: a single file-based database (no
separate server process to run, unlike the Redis-backed
RedisSessionStateStore this codebase already has) so every store's state
survives a process restart. Every table is a thin `key columns + JSON
blob` shape - the Pydantic model is the real schema; SQLite just needs
to find a row by key and hand back the JSON for the model to
re-validate, the same division of labor RedisSessionStateStore already
uses (`state.model_dump_json()` in, `Model(**json.loads(raw))` out).

One schema-init function, called once at app startup (app.main's
lifespan), rather than each store re-issuing `CREATE TABLE IF NOT
EXISTS` on every call - cheap either way in SQLite, but a single
up-front init is the more honest "this file owns the schema" story.
"""

from __future__ import annotations

import aiosqlite

_SCHEMA_STATEMENTS = [
    "CREATE TABLE IF NOT EXISTS session_state (session_id TEXT PRIMARY KEY, data TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS mastery (student_id TEXT NOT NULL, subtopic_id TEXT NOT NULL, data TEXT NOT NULL, PRIMARY KEY (student_id, subtopic_id))",
    "CREATE TABLE IF NOT EXISTS misconceptions (student_id TEXT NOT NULL, misconception_id TEXT NOT NULL, data TEXT NOT NULL, PRIMARY KEY (student_id, misconception_id))",
    "CREATE TABLE IF NOT EXISTS review_state (student_id TEXT NOT NULL, subtopic_id TEXT NOT NULL, data TEXT NOT NULL, PRIMARY KEY (student_id, subtopic_id))",
    "CREATE TABLE IF NOT EXISTS ia_project_state (student_id TEXT NOT NULL, project_id TEXT NOT NULL, data TEXT NOT NULL, PRIMARY KEY (student_id, project_id))",
    "CREATE TABLE IF NOT EXISTS disclosure_log (id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT NOT NULL, project_id TEXT NOT NULL, data TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS review_queue (entry_id TEXT PRIMARY KEY, student_id TEXT NOT NULL, status TEXT NOT NULL, data TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS response_log (id INTEGER PRIMARY KEY AUTOINCREMENT, template_id TEXT NOT NULL, student_id TEXT NOT NULL, data TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS guardrail_signals (id INTEGER PRIMARY KEY AUTOINCREMENT, turn_id TEXT NOT NULL, data TEXT NOT NULL)",
    "CREATE INDEX IF NOT EXISTS idx_disclosure_student ON disclosure_log (student_id, project_id)",
    "CREATE INDEX IF NOT EXISTS idx_review_queue_student ON review_queue (student_id, status)",
    "CREATE INDEX IF NOT EXISTS idx_response_log_template ON response_log (template_id)",
    "CREATE INDEX IF NOT EXISTS idx_response_log_student ON response_log (student_id)",
]


async def init_sqlite_schema(db_path: str) -> None:
    async with aiosqlite.connect(db_path) as db:
        # WAL mode lets concurrent readers proceed while a write is in
        # flight - the right default for a single-process FastAPI app
        # issuing overlapping async requests against one file, not just
        # a tuning knob.
        await db.execute("PRAGMA journal_mode=WAL")
        for statement in _SCHEMA_STATEMENTS:
            await db.execute(statement)
        await db.commit()


class SqliteBackedStore:
    """Shared base for every Sqlite*Store below: holds the db path and
    ensures the schema exists exactly once per instance before the
    first real operation, so a store works correctly on its own in a
    test (against a fresh tmp_path db) without the caller having to
    remember to call init_sqlite_schema first - app.main's lifespan
    still calls it explicitly too, for an honest startup-time log line
    rather than a silent first-request cost."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._schema_ready = False

    async def _ensure_schema(self) -> None:
        if not self._schema_ready:
            await init_sqlite_schema(self._db_path)
            self._schema_ready = True
