"""SQLite persistence: history, embedding cache, clusters, resolutions.

One file, committed by the daily Action to a `data` branch. That gives versioned
history, zero infrastructure, and a database you can copy locally and query with
any SQLite client.

Vectors are stored L2-normalised as float32 blobs, so cosine similarity is a
plain dot product and similarity search is one `numpy` matmul over the centroid
matrix. At this corpus size — a few thousand distinct templates across all of
history — exact search takes well under a millisecond and is *more* correct than
an approximate index. A `sqlite-vec` virtual table becomes worth adding somewhere
north of ~10^5 vectors; the search is isolated behind `search_clusters` so that
swap is a one-function change. See docs/ARCHITECTURE.md for the reasoning.
"""

from __future__ import annotations

import contextlib
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .models import Cluster

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS clusters (
    cluster_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    label            TEXT NOT NULL,
    category         TEXT NOT NULL DEFAULT 'unknown',
    severity         TEXT NOT NULL DEFAULT 'unknown',
    root_cause       TEXT,
    suggested_action TEXT,
    centroid         BLOB,
    dim              INTEGER,
    template_count   INTEGER NOT NULL DEFAULT 0,
    total_events     INTEGER NOT NULL DEFAULT 0,
    first_seen       TEXT,
    last_seen        TEXT,
    labeled_by       TEXT NOT NULL DEFAULT 'rule'
);

CREATE TABLE IF NOT EXISTS templates (
    template_hash   TEXT PRIMARY KEY,
    text            TEXT NOT NULL,
    example_message TEXT,
    cluster_id      INTEGER REFERENCES clusters(cluster_id),
    event_count     INTEGER NOT NULL DEFAULT 0,
    first_seen      TEXT,
    last_seen       TEXT
);

CREATE TABLE IF NOT EXISTS embeddings (
    template_hash TEXT PRIMARY KEY,
    model         TEXT NOT NULL,
    dim           INTEGER NOT NULL,
    vector        BLOB NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    slack_ts             TEXT PRIMARY KEY,
    occurred_at          TEXT NOT NULL,
    message              TEXT NOT NULL,
    template_hash        TEXT NOT NULL,
    cluster_id           INTEGER REFERENCES clusters(cluster_id),
    quote_id             TEXT,
    quote_id_confidence  REAL NOT NULL DEFAULT 0,
    correlation_id       TEXT,
    http_status          INTEGER,
    service              TEXT,
    raw_excerpt          TEXT,
    ingested_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS resolutions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id INTEGER REFERENCES clusters(cluster_id),
    text       TEXT NOT NULL,
    author     TEXT,
    slack_ts   TEXT UNIQUE,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_date         TEXT PRIMARY KEY,
    started_at       TEXT NOT NULL,
    finished_at      TEXT,
    messages_scanned INTEGER NOT NULL DEFAULT 0,
    events_extracted INTEGER NOT NULL DEFAULT 0,
    new_clusters     INTEGER NOT NULL DEFAULT 0,
    embed_calls      INTEGER NOT NULL DEFAULT 0,
    llm_calls        INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_events_occurred ON events(occurred_at);
CREATE INDEX IF NOT EXISTS idx_events_cluster  ON events(cluster_id);
CREATE INDEX IF NOT EXISTS idx_events_template ON events(template_hash);
CREATE INDEX IF NOT EXISTS idx_templates_cluster ON templates(cluster_id);
"""


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class Store:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(_SCHEMA)
        self._migrate()
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self.conn.commit()

    def _migrate(self) -> None:
        """Additive migrations for databases created by an earlier version.

        The schema above is only applied to a fresh file; an existing corpus
        represents hours of backfill and must not have to be rebuilt to pick up a
        new column.
        """
        existing = {row["name"] for row in self.conn.execute("PRAGMA table_info(events)")}
        if "correlation_id" not in existing:
            self.conn.execute("ALTER TABLE events ADD COLUMN correlation_id TEXT")
        # Indexed here rather than in the schema script: on an older database the
        # script runs before the column can be added, and indexing a column that
        # does not exist yet fails the whole open.
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_correlation ON events(correlation_id)"
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        # Fold the write-ahead log back into the main file so the committed .db
        # is self-contained; otherwise CI would need to carry -wal and -shm too.
        # Best effort: a checkpoint failure must not stop the file closing.
        with contextlib.suppress(sqlite3.Error):
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- embedding cache ----------------------------------------------------

    def get_cached_embeddings(self, hashes: list[str], model: str) -> dict[str, np.ndarray]:
        if not hashes:
            return {}
        out: dict[str, np.ndarray] = {}
        chunk = 400
        for i in range(0, len(hashes), chunk):
            part = hashes[i : i + chunk]
            placeholders = ",".join("?" * len(part))
            rows = self.conn.execute(
                f"SELECT template_hash, vector FROM embeddings "
                f"WHERE model = ? AND template_hash IN ({placeholders})",
                (model, *part),
            ).fetchall()
            for row in rows:
                out[row["template_hash"]] = from_blob(row["vector"])
        return out

    def put_embeddings(self, vectors: dict[str, np.ndarray], model: str) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO embeddings(template_hash, model, dim, vector, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            [(h, model, int(v.shape[0]), to_blob(v), _now()) for h, v in vectors.items()],
        )
        self.conn.commit()

    # -- templates ----------------------------------------------------------

    def get_template_cluster(self, template_hash: str) -> int | None:
        row = self.conn.execute(
            "SELECT cluster_id FROM templates WHERE template_hash = ?", (template_hash,)
        ).fetchone()
        return row["cluster_id"] if row else None

    def upsert_template(
        self,
        template_hash: str,
        text: str,
        example_message: str,
        cluster_id: int | None,
        seen_at: str,
        added_events: int,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO templates(template_hash, text, example_message, cluster_id,
                                  event_count, first_seen, last_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(template_hash) DO UPDATE SET
                cluster_id  = COALESCE(excluded.cluster_id, templates.cluster_id),
                event_count = templates.event_count + excluded.event_count,
                last_seen   = MAX(templates.last_seen, excluded.last_seen)
            """,
            (template_hash, text, example_message, cluster_id, added_events, seen_at, seen_at),
        )

    # -- clusters -----------------------------------------------------------

    def load_centroids(self) -> tuple[list[int], np.ndarray]:
        """Return (cluster_ids, matrix) with one L2-normalised row per cluster."""
        rows = self.conn.execute(
            "SELECT cluster_id, centroid, dim FROM clusters "
            "WHERE centroid IS NOT NULL ORDER BY cluster_id"
        ).fetchall()
        if not rows:
            return [], np.zeros((0, 0), dtype=np.float32)
        ids = [row["cluster_id"] for row in rows]
        matrix = np.vstack([from_blob(row["centroid"]) for row in rows])
        return ids, matrix

    def search_clusters(self, vector: np.ndarray, top_k: int = 5) -> list[tuple[int, float]]:
        """Exact cosine search over cluster centroids. Vectors are pre-normalised,
        so the dot product is the cosine similarity."""
        ids, matrix = self.load_centroids()
        if not ids or matrix.size == 0 or matrix.shape[1] != vector.shape[0]:
            return []
        sims = matrix @ vector.astype(np.float32)
        order = np.argsort(-sims)[:top_k]
        return [(ids[i], float(sims[i])) for i in order]

    def create_cluster(
        self, label: str, centroid: np.ndarray, seen_at: str, labeled_by: str = "rule"
    ) -> int:
        cursor = self.conn.execute(
            "INSERT INTO clusters(label, centroid, dim, template_count, total_events, "
            "first_seen, last_seen, labeled_by) VALUES (?, ?, ?, 0, 0, ?, ?, ?)",
            (label, to_blob(centroid), int(centroid.shape[0]), seen_at, seen_at, labeled_by),
        )
        self.conn.commit()
        return int(cursor.lastrowid or 0)

    def update_centroid(self, cluster_id: int, vector: np.ndarray, weight: int = 1) -> None:
        """Fold a new member into the centroid as a count-weighted running mean,
        then re-normalise so the dot-product-is-cosine invariant holds."""
        row = self.conn.execute(
            "SELECT centroid, template_count FROM clusters WHERE cluster_id = ?", (cluster_id,)
        ).fetchone()
        if row is None:
            return
        current = from_blob(row["centroid"]) if row["centroid"] else np.zeros_like(vector)
        n = max(int(row["template_count"]), 0)
        blended = (current * n + vector.astype(np.float32) * weight) / max(n + weight, 1)
        norm = float(np.linalg.norm(blended))
        if norm > 0:
            blended = blended / norm
        self.conn.execute(
            "UPDATE clusters SET centroid = ?, template_count = template_count + ? "
            "WHERE cluster_id = ?",
            (to_blob(blended), weight, cluster_id),
        )

    def bump_cluster(self, cluster_id: int, events: int, seen_at: str) -> None:
        self.conn.execute(
            "UPDATE clusters SET total_events = total_events + ?, "
            "last_seen = MAX(COALESCE(last_seen, ''), ?) WHERE cluster_id = ?",
            (events, seen_at, cluster_id),
        )

    def set_cluster_label(
        self,
        cluster_id: int,
        label: str,
        category: str,
        severity: str,
        root_cause: str | None,
        suggested_action: str | None,
        labeled_by: str,
    ) -> None:
        # Two report lines carrying the same label are indistinguishable to a
        # reader, and nothing upstream can fully guarantee uniqueness — the rule
        # fallback is coarse and a model can repeat itself. Enforce it here, at
        # the last point before the label is persisted.
        taken = self.conn.execute(
            "SELECT 1 FROM clusters WHERE label = ? AND cluster_id != ?", (label, cluster_id)
        ).fetchone()
        if taken:
            label = f"{label} (#{cluster_id})"

        self.conn.execute(
            "UPDATE clusters SET label = ?, category = ?, severity = ?, root_cause = ?, "
            "suggested_action = ?, labeled_by = ? WHERE cluster_id = ?",
            (label, category, severity, root_cause, suggested_action, labeled_by, cluster_id),
        )
        self.conn.commit()

    def get_cluster(self, cluster_id: int) -> Cluster | None:
        row = self.conn.execute(
            "SELECT * FROM clusters WHERE cluster_id = ?", (cluster_id,)
        ).fetchone()
        return _row_to_cluster(row) if row else None

    def cluster_exemplars(self, cluster_id: int, limit: int = 5) -> list[str]:
        rows = self.conn.execute(
            "SELECT example_message FROM templates WHERE cluster_id = ? "
            "ORDER BY event_count DESC LIMIT ?",
            (cluster_id, limit),
        ).fetchall()
        return [row["example_message"] for row in rows if row["example_message"]]

    def cluster_templates(self, cluster_id: int, limit: int = 10) -> list[str]:
        rows = self.conn.execute(
            "SELECT text FROM templates WHERE cluster_id = ? ORDER BY event_count DESC LIMIT ?",
            (cluster_id, limit),
        ).fetchall()
        return [row["text"] for row in rows]

    # -- events -------------------------------------------------------------

    def event_exists(self, slack_ts: str) -> bool:
        return (
            self.conn.execute("SELECT 1 FROM events WHERE slack_ts = ?", (slack_ts,)).fetchone()
            is not None
        )

    def existing_event_ids(self, slack_ts: list[str]) -> set[str]:
        """Which of these events have we already ingested?

        Cumulative counters (`clusters.total_events`, `templates.event_count`)
        must only ever be incremented by genuinely new events. Windows overlap in
        practice — a manual `--days 3` re-run, a retried workflow — and without
        this check those counters inflate every time, quietly corrupting the
        historical baselines the trend comparison is built on.
        """
        if not slack_ts:
            return set()
        found: set[str] = set()
        chunk = 400
        for i in range(0, len(slack_ts), chunk):
            part = slack_ts[i : i + chunk]
            placeholders = ",".join("?" * len(part))
            rows = self.conn.execute(
                f"SELECT slack_ts FROM events WHERE slack_ts IN ({placeholders})", part
            ).fetchall()
            found.update(row["slack_ts"] for row in rows)
        return found

    def insert_event(self, **kwargs: Any) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO events(slack_ts, occurred_at, message, template_hash,
                cluster_id, quote_id, quote_id_confidence, correlation_id, http_status,
                service, raw_excerpt, ingested_at)
            VALUES (:slack_ts, :occurred_at, :message, :template_hash, :cluster_id,
                    :quote_id, :quote_id_confidence, :correlation_id, :http_status,
                    :service, :raw_excerpt, :ingested_at)
            """,
            {**kwargs, "ingested_at": _now()},
        )

    def prior_daily_mean(self, cluster_id: int, before: str, days: int = 7) -> float | None:
        """Mean events/day for this cluster over the N days before `before`,
        counting only days the pipeline actually ran."""
        row = self.conn.execute(
            """
            SELECT COUNT(*) AS total, COUNT(DISTINCT date(occurred_at)) AS days
            FROM events
            WHERE cluster_id = ?
              AND occurred_at < ?
              AND occurred_at >= datetime(?, ?)
            """,
            (cluster_id, before, before, f"-{days} days"),
        ).fetchone()
        if not row or not row["days"]:
            return None
        return float(row["total"]) / float(row["days"])

    def cluster_first_seen(self, cluster_id: int) -> str | None:
        row = self.conn.execute(
            "SELECT first_seen FROM clusters WHERE cluster_id = ?", (cluster_id,)
        ).fetchone()
        return row["first_seen"] if row else None

    # -- resolutions (RAG corpus) -------------------------------------------

    def add_resolution(self, cluster_id: int, text: str, author: str | None, slack_ts: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO resolutions(cluster_id, text, author, slack_ts, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (cluster_id, text, author, slack_ts, _now()),
        )
        self.conn.commit()

    def get_resolutions(self, cluster_id: int, limit: int = 3) -> list[str]:
        rows = self.conn.execute(
            "SELECT text FROM resolutions WHERE cluster_id = ? ORDER BY created_at DESC LIMIT ?",
            (cluster_id, limit),
        ).fetchall()
        return [row["text"] for row in rows]

    # -- runs ---------------------------------------------------------------

    def start_run(self, run_date: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO runs(run_date, started_at) VALUES (?, ?)",
            (run_date, _now()),
        )
        self.conn.commit()

    def finish_run(self, run_date: str, **stats: int) -> None:
        self.conn.execute(
            "UPDATE runs SET finished_at = ?, messages_scanned = ?, events_extracted = ?, "
            "new_clusters = ?, embed_calls = ?, llm_calls = ? WHERE run_date = ?",
            (
                _now(),
                stats.get("messages_scanned", 0),
                stats.get("events_extracted", 0),
                stats.get("new_clusters", 0),
                stats.get("embed_calls", 0),
                stats.get("llm_calls", 0),
                run_date,
            ),
        )
        self.conn.commit()

    def commit(self) -> None:
        self.conn.commit()


def _row_to_cluster(row: sqlite3.Row) -> Cluster:
    return Cluster(
        cluster_id=row["cluster_id"],
        label=row["label"],
        category=row["category"],
        severity=row["severity"],
        root_cause=row["root_cause"],
        suggested_action=row["suggested_action"],
        first_seen=_parse_dt(row["first_seen"]),
        last_seen=_parse_dt(row["last_seen"]),
        template_count=row["template_count"],
        total_events=row["total_events"],
    )


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
