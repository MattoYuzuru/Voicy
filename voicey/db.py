"""One process owns this SQLite file. No external database service."""

import json
import sqlite3
import time
from pathlib import Path


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA busy_timeout=5000;
            CREATE TABLE IF NOT EXISTS settings (
                user_id INTEGER, key TEXT, value TEXT, PRIMARY KEY(user_id,key));
            CREATE TABLE IF NOT EXISTS terms (
                user_id INTEGER, canonical TEXT, aliases TEXT,
                PRIMARY KEY(user_id,canonical));
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL,
                chat_id INTEGER NOT NULL, message_id INTEGER NOT NULL,
                kind TEXT NOT NULL, payload TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued', result TEXT,
                created REAL NOT NULL, updated REAL NOT NULL,
                UNIQUE(chat_id,message_id,kind));
        """)
        self.db.execute("UPDATE jobs SET status='queued' WHERE status='running'")
        self.db.commit()

    def get(self, uid: int, key: str, default: str = "") -> str:
        r = self.db.execute("SELECT value FROM settings WHERE user_id=? AND key=?", (uid, key)).fetchone()
        return r[0] if r else default

    def set(self, uid: int, key: str, value: str):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO settings VALUES (?,?,?)", (uid, key, value))

    def terms(self, uid: int) -> list[dict]:
        return [
            dict(r)
            for r in self.db.execute("SELECT canonical,aliases FROM terms WHERE user_id=? ORDER BY rowid", (uid,))
        ]

    def add_terms(self, uid: int, entries: list[dict]):
        # Validate the complete import before writing anything.
        if len(entries) > 200:
            raise ValueError("Не больше 200 терминов за импорт.")
        normalized = []
        for e in entries:
            term = e["canonical"].strip()
            aliases = e.get("aliases", "")
            if isinstance(aliases, list):
                aliases = ", ".join(aliases)
            if not term or len(term) > 80 or len(aliases) > 200 or "\n" in term:
                raise ValueError("Термин: 1–80 символов; варианты: до 200.")
            normalized.append((uid, term, aliases))
        existing = {t["canonical"] for t in self.terms(uid)}
        if len(existing | {t[1] for t in normalized}) > 200:
            raise ValueError("В словаре максимум 200 терминов.")
        with self.db:
            self.db.executemany("INSERT OR REPLACE INTO terms VALUES (?,?,?)", normalized)

    def remove_term(self, uid: int, term: str):
        with self.db:
            self.db.execute("DELETE FROM terms WHERE user_id=? AND canonical=?", (uid, term))

    def hint(self, uid: int) -> str:
        # Newest entries first, bounded to fit Whisper's short prompt context.
        names = [t["canonical"] for t in reversed(self.terms(uid))][:25]
        return ", ".join(names)[:400]

    def enqueue(self, uid: int, chat: int, message: int, kind: str, payload: dict, limit: int = 12) -> int | None:
        existing = self.db.execute(
            "SELECT id,status FROM jobs WHERE chat_id=? AND message_id=? AND kind=?", (chat, message, kind)
        ).fetchone()
        if existing and existing["status"] != "error":
            return None
        if self.db.execute("SELECT count(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0] >= limit:
            raise ValueError("Очередь заполнена. Попробуй чуть позже.")
        now = time.time()
        with self.db:
            if existing:
                self.db.execute(
                    "UPDATE jobs SET status='queued',payload=?,updated=? WHERE id=?",
                    (json.dumps(payload), now, existing["id"]),
                )
                return existing["id"]
            cur = self.db.execute(
                """INSERT OR IGNORE INTO jobs
                (user_id,chat_id,message_id,kind,payload,created,updated) VALUES (?,?,?,?,?,?,?)""",
                (uid, chat, message, kind, json.dumps(payload), now, now),
            )
            return cur.lastrowid if cur.rowcount else None

    def claim(self) -> dict | None:
        # Synchronous transaction, no await between select and update.
        with self.db:
            r = self.db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY id LIMIT 1").fetchone()
            if not r:
                return None
            self.db.execute("UPDATE jobs SET status='running',updated=? WHERE id=?", (time.time(), r["id"]))
        return dict(r)

    def finish(self, jid: int, result: str | None, status: str = "done"):
        with self.db:
            self.db.execute(
                "UPDATE jobs SET status=?,result=?,updated=? WHERE id=?", (status, result, time.time(), jid)
            )

    def job(self, jid: int, uid: int) -> dict | None:
        r = self.db.execute("SELECT * FROM jobs WHERE id=? AND user_id=?", (jid, uid)).fetchone()
        return dict(r) if r else None

    def cleanup(self, days: int):
        with self.db:
            self.db.execute(
                "DELETE FROM jobs WHERE status IN ('done','error') AND updated<?", (time.time() - days * 86400,)
            )

    def close(self):
        self.db.close()
