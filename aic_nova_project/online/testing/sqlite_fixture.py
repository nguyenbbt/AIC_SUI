"""Ephemeral SQLite fixture built from the current Online read schema."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

SQLITE_FIXTURE_SCHEMA_VERSION = "organizer-v1"


def create_sqlite_fixture(
    path: str | Path,
    *,
    video_rows: Iterable[tuple[str, str | None]] = (),
    metadata_rows: Iterable[
        tuple[str, str, int, int, float, float, int, str]
    ] = (),
    object_rows: Iterable[
        tuple[
            str,
            str,
            str,
            str | None,
            str | None,
            float,
            float,
            float,
            float,
            float,
            str,
        ]
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
            CREATE TABLE videos (
                video_id TEXT PRIMARY KEY,
                media_title TEXT,
                media_author TEXT,
                media_description TEXT,
                media_keywords_json TEXT,
                media_length_sec REAL,
                publish_date TEXT,
                watch_url TEXT,
                video_rel_path TEXT
            );
            CREATE TABLE metadata (
                frame_id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL,
                keyframe_no INTEGER NOT NULL,
                local_index INTEGER NOT NULL,
                pts_time_sec REAL NOT NULL,
                fps REAL NOT NULL,
                source_frame_idx INTEGER NOT NULL,
                image_rel_path TEXT NOT NULL,
                FOREIGN KEY (video_id) REFERENCES videos(video_id),
                UNIQUE(video_id, keyframe_no),
                UNIQUE(video_id, local_index)
            );
            CREATE INDEX idx_metadata_video_local ON metadata(video_id, local_index);
            CREATE TABLE objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id TEXT NOT NULL,
                label_display TEXT NOT NULL,
                label_normalized TEXT NOT NULL,
                class_mid TEXT,
                class_label_id TEXT,
                confidence REAL NOT NULL,
                x_min REAL NOT NULL,
                y_min REAL NOT NULL,
                x_max REAL NOT NULL,
                y_max REAL NOT NULL,
                model_source TEXT NOT NULL,
                FOREIGN KEY (frame_id) REFERENCES metadata(frame_id)
                    ON DELETE CASCADE
            );
            CREATE INDEX idx_objects_frame_id ON objects(frame_id);
            CREATE INDEX idx_objects_label ON objects(label_normalized);
            """
        )
        connection.executemany(
            "INSERT INTO videos(video_id, video_rel_path) VALUES (?, ?)",
            tuple(video_rows),
        )
        connection.executemany(
            """
            INSERT INTO metadata(
                frame_id, video_id, keyframe_no, local_index, pts_time_sec,
                fps, source_frame_idx, image_rel_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            tuple(metadata_rows),
        )
        connection.executemany(
            """
            INSERT INTO objects(
                frame_id, label_display, label_normalized, class_mid,
                class_label_id, confidence, x_min, y_min, x_max, y_max,
                model_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            tuple(object_rows),
        )
        connection.commit()
    finally:
        connection.close()
    return target
