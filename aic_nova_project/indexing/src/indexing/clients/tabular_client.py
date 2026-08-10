"""
SQLite (Tabular) client for relational metadata and object detection storage.

Manages 3 tables:
- videos: source-video identity and dimensions
- metadata: frame-level metadata and the decoded source-frame identity
- objects: detected objects per frame (FK to metadata.frame_id)

Uses standard sqlite3 module from Python's standard library.
"""

import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

CREATE_VIDEOS_TABLE = """
CREATE TABLE IF NOT EXISTS videos (
    video_id              TEXT PRIMARY KEY,
    source_video_rel_path TEXT NOT NULL,
    fps                   REAL NOT NULL,
    duration_sec          REAL NOT NULL,
    frame_count           INTEGER NOT NULL,
    width                 INTEGER NOT NULL,
    height                INTEGER NOT NULL
);
"""

CREATE_METADATA_TABLE = """
CREATE TABLE IF NOT EXISTS metadata (
    frame_id         TEXT PRIMARY KEY,
    video_id         TEXT NOT NULL,
    shot_id          INTEGER NOT NULL,
    source_frame_idx INTEGER NOT NULL,
    timestamp        REAL NOT NULL,
    image_rel_path   TEXT NOT NULL,
    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
);
"""

CREATE_METADATA_INDEX = """
CREATE INDEX IF NOT EXISTS idx_metadata_video_id ON metadata(video_id);
"""

CREATE_METADATA_TIMELINE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_metadata_video_timeline
ON metadata(video_id, timestamp, frame_id);
"""

CREATE_METADATA_SOURCE_FRAME_INDEX = """
CREATE INDEX IF NOT EXISTS idx_metadata_video_source_frame
ON metadata(video_id, source_frame_idx);
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
        """Create self-indexed-v2 videos, metadata, and objects tables."""
        cursor = self.conn.cursor()
        try:
            cursor.execute(CREATE_VIDEOS_TABLE)
            cursor.execute(CREATE_METADATA_TABLE)
            cursor.execute(CREATE_METADATA_INDEX)
            cursor.execute(CREATE_METADATA_TIMELINE_INDEX)
            cursor.execute(CREATE_METADATA_SOURCE_FRAME_INDEX)
            cursor.execute(CREATE_OBJECTS_TABLE)
            cursor.execute(CREATE_OBJECTS_INDEX_FRAME)
            cursor.execute(CREATE_OBJECTS_INDEX_LABEL)
        except sqlite3.OperationalError as exc:
            self.conn.rollback()
            raise ValueError(
                "SQLite schema contract mismatch while creating indexes."
            ) from exc
        self.conn.commit()
        self.audit_schema()
        logger.info("SQLite tables created.")

    def audit_schema(self) -> None:
        """Fail if existing SQLite tables, indexes, or FK differ."""
        expected_columns = {
            "videos": {
                "video_id": ("TEXT", 0, 1),
                "source_video_rel_path": ("TEXT", 1, 0),
                "fps": ("REAL", 1, 0),
                "duration_sec": ("REAL", 1, 0),
                "frame_count": ("INTEGER", 1, 0),
                "width": ("INTEGER", 1, 0),
                "height": ("INTEGER", 1, 0),
            },
            "metadata": {
                "frame_id": ("TEXT", 0, 1),
                "video_id": ("TEXT", 1, 0),
                "shot_id": ("INTEGER", 1, 0),
                "source_frame_idx": ("INTEGER", 1, 0),
                "timestamp": ("REAL", 1, 0),
                "image_rel_path": ("TEXT", 1, 0),
            },
            "objects": {
                "id": ("INTEGER", 0, 1),
                "frame_id": ("TEXT", 1, 0),
                "label": ("TEXT", 1, 0),
                "confidence": ("REAL", 1, 0),
                "x_min": ("REAL", 1, 0),
                "y_min": ("REAL", 1, 0),
                "x_max": ("REAL", 1, 0),
                "y_max": ("REAL", 1, 0),
                "model_source": ("TEXT", 0, 0),
            },
        }
        for table_name, expected in expected_columns.items():
            rows = self.conn.execute(
                f"PRAGMA table_info({table_name})"
            ).fetchall()
            actual = {
                row[1]: (str(row[2]).upper(), int(row[3]), int(row[5]))
                for row in rows
            }
            if actual != expected:
                raise ValueError(
                    f"SQLite schema contract mismatch for "
                    f"table '{table_name}'."
                )

        required_indexes = {
            "metadata": {
                "idx_metadata_video_id",
                "idx_metadata_video_timeline",
                "idx_metadata_video_source_frame",
            },
            "objects": {
                "idx_objects_frame_id",
                "idx_objects_label",
            },
        }
        for table_name, required in required_indexes.items():
            actual_indexes = {
                row[1]
                for row in self.conn.execute(
                    f"PRAGMA index_list({table_name})"
                ).fetchall()
            }
            if not required.issubset(actual_indexes):
                raise ValueError(
                    f"SQLite schema contract mismatch: missing index "
                    f"for table '{table_name}'."
                )

        foreign_keys = self.conn.execute(
            "PRAGMA foreign_key_list(objects)"
        ).fetchall()
        has_required_fk = any(
            row[2] == "metadata"
            and row[3] == "frame_id"
            and row[4] == "frame_id"
            and str(row[6]).upper() == "CASCADE"
            for row in foreign_keys
        )
        if not has_required_fk:
            raise ValueError(
                "SQLite schema contract mismatch: objects.frame_id "
                "must reference metadata.frame_id ON DELETE CASCADE."
            )

        metadata_foreign_keys = self.conn.execute(
            "PRAGMA foreign_key_list(metadata)"
        ).fetchall()
        has_video_fk = any(
            row[2] == "videos"
            and row[3] == "video_id"
            and row[4] == "video_id"
            and str(row[6]).upper() == "CASCADE"
            for row in metadata_foreign_keys
        )
        if not has_video_fk:
            raise ValueError(
                "SQLite schema contract mismatch: metadata.video_id "
                "must reference videos.video_id ON DELETE CASCADE."
            )

    def insert_video_batch(self, records: List[Dict[str, Any]]):
        """Upsert source-video rows without REPLACE cascade side effects."""
        if not records:
            return
        self.conn.executemany(
            "INSERT INTO videos (video_id, source_video_rel_path, fps, "
            "duration_sec, frame_count, width, height) VALUES "
            "(:video_id, :source_video_rel_path, :fps, :duration_sec, "
            ":frame_count, :width, :height) "
            "ON CONFLICT(video_id) DO UPDATE SET "
            "source_video_rel_path=excluded.source_video_rel_path, "
            "fps=excluded.fps, duration_sec=excluded.duration_sec, "
            "frame_count=excluded.frame_count, width=excluded.width, "
            "height=excluded.height",
            records,
        )
        self.conn.commit()
        logger.info("Inserted %s video records.", len(records))

    def insert_metadata_batch(self, records: List[Dict[str, Any]]):
        """
        Insert or replace metadata records.
        Uses INSERT OR REPLACE to achieve upsert semantics.
        """
        if not records:
            return

        cursor = self.conn.cursor()
        cursor.executemany(
            "INSERT OR REPLACE INTO metadata "
            "(frame_id, video_id, shot_id, source_frame_idx, timestamp, "
            "image_rel_path) VALUES (:frame_id, :video_id, :shot_id, "
            ":source_frame_idx, :timestamp, :image_rel_path)",
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
        cursor.execute("DELETE FROM videos WHERE video_id = ?", (video_id,))
        self.conn.commit()
        logger.info(f"Deleted all records for video_id='{video_id}' from SQLite.")

    def snapshot_by_video_id(
        self,
        video_id: str,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Read metadata and child objects for a restorable video snapshot."""
        cursor = self.conn.cursor()
        metadata_rows = cursor.execute(
            "SELECT frame_id, video_id, shot_id, source_frame_idx, "
            "timestamp, image_rel_path "
            "FROM metadata WHERE video_id = ? ORDER BY frame_id",
            (video_id,),
        ).fetchall()
        object_rows = cursor.execute(
            "SELECT o.frame_id, o.label, o.confidence, "
            "o.x_min, o.y_min, o.x_max, o.y_max, o.model_source "
            "FROM objects AS o "
            "JOIN metadata AS m ON m.frame_id = o.frame_id "
            "WHERE m.video_id = ? ORDER BY o.id",
            (video_id,),
        ).fetchall()
        metadata_keys = (
            "frame_id",
            "video_id",
            "shot_id",
            "source_frame_idx",
            "timestamp",
            "image_rel_path",
        )
        object_keys = (
            "frame_id",
            "label",
            "confidence",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
            "model_source",
        )
        return (
            [dict(zip(metadata_keys, row)) for row in metadata_rows],
            [dict(zip(object_keys, row)) for row in object_rows],
        )

    def snapshot_video_by_id(self, video_id: str) -> Dict[str, Any] | None:
        """Read the source-video row for rollback and resume comparison."""
        row = self.conn.execute(
            "SELECT video_id, source_video_rel_path, fps, duration_sec, "
            "frame_count, width, height FROM videos WHERE video_id = ?",
            (video_id,),
        ).fetchone()
        if row is None:
            return None
        keys = (
            "video_id",
            "source_video_rel_path",
            "fps",
            "duration_sec",
            "frame_count",
            "width",
            "height",
        )
        return dict(zip(keys, row))

    def restore_video_snapshot(self, record: Dict[str, Any] | None) -> None:
        """Restore one source-video row before its child metadata."""
        if record is not None:
            self.insert_video_batch([record])

    def restore_snapshot(
        self,
        metadata_records: List[Dict[str, Any]],
        object_records: List[Dict[str, Any]],
    ) -> None:
        """Restore one SQLite snapshot in a single local transaction."""
        cursor = self.conn.cursor()
        try:
            if metadata_records:
                cursor.executemany(
                    "INSERT OR REPLACE INTO metadata "
                    "(frame_id, video_id, shot_id, source_frame_idx, "
                    "timestamp, image_rel_path) VALUES "
                    "(:frame_id, :video_id, :shot_id, :source_frame_idx, "
                    ":timestamp, :image_rel_path)",
                    metadata_records,
                )
            if object_records:
                cursor.executemany(
                    "INSERT INTO objects "
                    "(frame_id, label, confidence, x_min, y_min, "
                    "x_max, y_max, model_source) "
                    "VALUES (:frame_id, :label, :confidence, :x_min, "
                    ":y_min, :x_max, :y_max, :model_source)",
                    object_records,
                )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def reset(self):
        """Drop and recreate all tables."""
        cursor = self.conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS objects")
        cursor.execute("DROP TABLE IF EXISTS metadata")
        cursor.execute("DROP TABLE IF EXISTS videos")
        self.conn.commit()
        self.create_tables()
        logger.info("SQLite tables reset.")
