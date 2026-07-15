"""
SQLite (Tabular) client for relational metadata and object detection storage.

Manages 2 tables:
- metadata: frame-level metadata (frame_id PK, video_id, shot_id, timestamp)
- objects: detected objects per frame (FK to metadata.frame_id)

Uses standard sqlite3 module from Python's standard library.
"""

import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

CREATE_METADATA_TABLE = """
CREATE TABLE IF NOT EXISTS metadata (
    frame_id   TEXT PRIMARY KEY,
    video_id   TEXT NOT NULL,
    shot_id    INTEGER NOT NULL,
    timestamp  REAL NOT NULL
);
"""

CREATE_METADATA_INDEX = """
CREATE INDEX IF NOT EXISTS idx_metadata_video_id ON metadata(video_id);
"""

CREATE_OBJECTS_TABLE = """
CREATE TABLE IF NOT EXISTS objects (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id     TEXT NOT NULL,
    label        TEXT NOT NULL,
    confidence   REAL NOT NULL,
    x_min        REAL NOT NULL,
    y_min        REAL NOT NULL,
    x_max        REAL NOT NULL,
    y_max        REAL NOT NULL,
    model_source TEXT,
    FOREIGN KEY (frame_id) REFERENCES metadata(frame_id) ON DELETE CASCADE
);
"""

CREATE_OBJECTS_INDEX_FRAME = """
CREATE INDEX IF NOT EXISTS idx_objects_frame_id ON objects(frame_id);
"""

CREATE_OBJECTS_INDEX_LABEL = """
CREATE INDEX IF NOT EXISTS idx_objects_label ON objects(label);
"""


class TabularClient:
    """Client for managing SQLite tables for metadata and object detection data."""

    def __init__(self, db_uri: str = "data/metadata.db"):
        """
        Args:
            db_uri: Path to the SQLite database file.
                    Can be prefixed with 'sqlite:///' which will be stripped.
        """
        # Strip SQLAlchemy-style prefix if present
        if db_uri.startswith("sqlite:///"):
            db_uri = db_uri[len("sqlite:///"):]
        self.db_path = Path(db_uri)
        self.conn: sqlite3.Connection | None = None

    def connect(self):
        """Open SQLite connection and enable WAL mode + foreign keys."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Connecting to SQLite at {self.db_path}...")
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")
        logger.info("SQLite connected.")

    def disconnect(self):
        """Close SQLite connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def create_tables(self):
        """Create metadata and objects tables with indices."""
        cursor = self.conn.cursor()
        cursor.execute(CREATE_METADATA_TABLE)
        cursor.execute(CREATE_METADATA_INDEX)
        cursor.execute(CREATE_OBJECTS_TABLE)
        cursor.execute(CREATE_OBJECTS_INDEX_FRAME)
        cursor.execute(CREATE_OBJECTS_INDEX_LABEL)
        self.conn.commit()
        logger.info("SQLite tables created.")

    def insert_metadata_batch(self, records: List[Dict[str, Any]]):
        """
        Insert or replace metadata records.
        Uses INSERT OR REPLACE to achieve upsert semantics.
        """
        if not records:
            return

        cursor = self.conn.cursor()
        cursor.executemany(
            "INSERT OR REPLACE INTO metadata (frame_id, video_id, shot_id, timestamp) "
            "VALUES (:frame_id, :video_id, :shot_id, :timestamp)",
            records,
        )
        self.conn.commit()
        logger.info(f"Inserted {len(records)} metadata records.")

    def insert_objects_batch(self, records: List[Dict[str, Any]]):
        """Insert object detection records."""
        if not records:
            return

        cursor = self.conn.cursor()
        cursor.executemany(
            "INSERT INTO objects (frame_id, label, confidence, x_min, y_min, x_max, y_max, model_source) "
            "VALUES (:frame_id, :label, :confidence, :x_min, :y_min, :x_max, :y_max, :model_source)",
            records,
        )
        self.conn.commit()
        logger.info(f"Inserted {len(records)} object records.")

    def delete_by_video_id(self, video_id: str):
        """
        Delete all metadata and object records for a given video_id.
        Foreign key CASCADE ensures objects are deleted when metadata rows are deleted.
        """
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM objects WHERE frame_id IN (SELECT frame_id FROM metadata WHERE video_id = ?)", (video_id,))
        cursor.execute("DELETE FROM metadata WHERE video_id = ?", (video_id,))
        self.conn.commit()
        logger.info(f"Deleted all records for video_id='{video_id}' from SQLite.")

    def reset(self):
        """Drop and recreate all tables."""
        cursor = self.conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS objects")
        cursor.execute("DROP TABLE IF EXISTS metadata")
        self.conn.commit()
        self.create_tables()
        logger.info("SQLite tables reset.")
