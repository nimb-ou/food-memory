"""SQLite metadata store for Food Memory artifacts."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ImageRecord:
    row_id: int
    source_id: str
    split: str
    label_id: int
    label_name: str
    image_hash: str
    embedding_hash: str
    width: int
    height: int


class MetadataStore:
    """Small SQLite wrapper for labels, rows, and build metadata."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self, clear: bool = False) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            if clear:
                conn.executescript(
                    """
                    DROP TABLE IF EXISTS images;
                    DROP TABLE IF EXISTS labels;
                    DROP TABLE IF EXISTS metadata;
                    """
                )
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS labels (
                    label_id INTEGER PRIMARY KEY,
                    label_name TEXT NOT NULL UNIQUE
                );
                CREATE TABLE IF NOT EXISTS images (
                    row_id INTEGER PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    split TEXT NOT NULL,
                    label_id INTEGER NOT NULL,
                    label_name TEXT NOT NULL,
                    image_hash TEXT NOT NULL,
                    embedding_hash TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    FOREIGN KEY(label_id) REFERENCES labels(label_id)
                );
                CREATE INDEX IF NOT EXISTS idx_images_hash ON images(image_hash);
                CREATE INDEX IF NOT EXISTS idx_images_split_label ON images(split, label_id);
                """
            )

    def set_metadata(self, key: str, value: Any) -> None:
        encoded = json.dumps(value, sort_keys=True)
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO metadata(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, encoded),
            )

    def get_metadata(self, key: str, default: Any = None) -> Any:
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def insert_label(self, label_id: int, label_name: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO labels(label_id, label_name) VALUES (?, ?)",
                (int(label_id), label_name),
            )

    def insert_record(self, record: ImageRecord) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO images(
                    row_id, source_id, split, label_id, label_name,
                    image_hash, embedding_hash, width, height
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(record.row_id),
                    record.source_id,
                    record.split,
                    int(record.label_id),
                    record.label_name,
                    record.image_hash,
                    record.embedding_hash,
                    int(record.width),
                    int(record.height),
                ),
            )

    def insert_records(self, records: Iterable[ImageRecord]) -> None:
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO images(
                    row_id, source_id, split, label_id, label_name,
                    image_hash, embedding_hash, width, height
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        int(r.row_id),
                        r.source_id,
                        r.split,
                        int(r.label_id),
                        r.label_name,
                        r.image_hash,
                        r.embedding_hash,
                        int(r.width),
                        int(r.height),
                    )
                    for r in records
                ],
            )

    def labels(self) -> dict[int, str]:
        with self.connect() as conn:
            rows = conn.execute("SELECT label_id, label_name FROM labels ORDER BY label_id").fetchall()
        return {int(row["label_id"]): str(row["label_name"]) for row in rows}

    def rows(self, split: str | None = None) -> list[ImageRecord]:
        sql = "SELECT * FROM images"
        params: tuple[Any, ...] = ()
        if split is not None:
            sql += " WHERE split = ?"
            params = (split,)
        sql += " ORDER BY row_id"
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_record(row) for row in rows]

    def get_by_image_hash(self, image_hash: str) -> ImageRecord | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM images WHERE image_hash = ? ORDER BY row_id LIMIT 1",
                (image_hash,),
            ).fetchone()
        return _row_to_record(row) if row else None

    def count(self, split: str | None = None) -> int:
        sql = "SELECT COUNT(*) AS n FROM images"
        params: tuple[Any, ...] = ()
        if split is not None:
            sql += " WHERE split = ?"
            params = (split,)
        with self.connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return int(row["n"])


def _row_to_record(row: sqlite3.Row) -> ImageRecord:
    return ImageRecord(
        row_id=int(row["row_id"]),
        source_id=str(row["source_id"]),
        split=str(row["split"]),
        label_id=int(row["label_id"]),
        label_name=str(row["label_name"]),
        image_hash=str(row["image_hash"]),
        embedding_hash=str(row["embedding_hash"]),
        width=int(row["width"]),
        height=int(row["height"]),
    )

