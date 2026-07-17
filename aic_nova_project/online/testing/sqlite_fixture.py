"""Ephemeral SQLite fixture built from the current Online read schema."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

SQLITE_FIXTURE_SCHEMA_VERSION = "metadata-objects-v1"


def create_sqlite_fixture(
    path: str | Path,
    *,
    metadata_rows: Iterable[tuple[str, str, int, float]] = (),
    object_rows: Iterable[
        tuple[str, str, float, float, float, float, float, str | None]
    ] = (),
) -> Path:
    """Create a test-only database; never used by Online production code."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(target)
    try:
        connection.executescript(
            """
            PRAGMA foreign_keys=ON;
            CREATE TABLE metadata (
                frame_id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL,
                shot_id INTEGER NOT NULL,
                timestamp REAL NOT NULL
            );
            CREATE INDEX idx_metadata_video_id ON metadata(video_id);
            CREATE TABLE objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id TEXT NOT NULL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                x_min REAL NOT NULL,
                y_min REAL NOT NULL,
                x_max REAL NOT NULL,
                y_max REAL NOT NULL,
                model_source TEXT,
                FOREIGN KEY (frame_id) REFERENCES metadata(frame_id)
                    ON DELETE CASCADE
            );
            CREATE INDEX idx_objects_frame_id ON objects(frame_id);
            CREATE INDEX idx_objects_label ON objects(label);
            """
        )
        connection.executemany(
            "INSERT INTO metadata(frame_id, video_id, shot_id, timestamp) VALUES (?, ?, ?, ?)",
            tuple(metadata_rows),
        )
        connection.executemany(
            """
            INSERT INTO objects(
                frame_id, label, confidence, x_min, y_min, x_max, y_max, model_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            tuple(object_rows),
        )
        connection.commit()
    finally:
        connection.close()
    return target
