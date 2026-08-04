from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path
from uuid import uuid4

from online.adapters.sqlite import SQLiteReadAdapter
from online.config import SQLiteResourceConfig
from online.domain.errors import ContractMismatchError, InvalidQueryError
from online.testing import create_sqlite_fixture


SCHEMA = """
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
    image_rel_path TEXT NOT NULL
);
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
    model_source TEXT NOT NULL
);
"""


class SQLiteAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        connection = sqlite3.connect(":memory:")
        connection.executescript(SCHEMA)
        connection.executemany(
            "INSERT INTO videos(video_id, video_rel_path) VALUES (?, ?)",
            [
                ("L21_V001", "video/L21_V001.mp4"),
                ("L21_V002", "video/L21_V002.mp4"),
            ],
        )
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("L21_V001_003", "L21_V001", 3, 2, 10.0, 30.0, 300, "keyframes/L21_V001/003.jpg"),
                ("L21_V001_001", "L21_V001", 1, 0, 0.0, 30.0, 0, "keyframes/L21_V001/001.jpg"),
                ("L21_V001_002", "L21_V001", 2, 1, 1 / 30, 30.0, 0, "keyframes/L21_V001/002.jpg"),
                ("L21_V002_001", "L21_V002", 1, 0, 0.0, 29.97, 0, "keyframes/L21_V002/ảnh-001.jpg"),
            ],
        )
        connection.executemany(
            """
            INSERT INTO objects(
                frame_id, label_display, label_normalized, class_mid,
                class_label_id, confidence, x_min, y_min, x_max, y_max,
                model_source
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            [
                ("L21_V001_002", "Person", "person", "/m/01g317", "0", 0.95, 0.1, 0.1, 0.3, 0.8, "organizer_open_images"),
                ("L21_V001_002", "Person", "person", "/m/01g317", "0", 0.35, 0.4, 0.1, 0.6, 0.8, "organizer_open_images"),
                ("L21_V001_002", "Car", "car", "/m/0k4j", "1", 0.80, 0.5, 0.2, 0.9, 0.7, "organizer_open_images"),
            ],
        )
        connection.commit()
        self.adapter = SQLiteReadAdapter(
            SQLiteResourceConfig(batch_size=2), connection=connection
        )

    def tearDown(self) -> None:
        self.adapter._conn().close()

    def test_batch_hydration_missing_and_deterministic_order(self) -> None:
        result = self.adapter.get_frames_by_ids(
            ["L21_V001_001", "missing", "L21_V001_003"]
        )
        self.assertEqual(set(result), {"L21_V001_001", "L21_V001_003"})
        ordered = self.adapter.get_ordered_frames_by_video("L21_V001")
        self.assertEqual([frame.local_index for frame in ordered], [0, 1, 2])
        self.assertEqual(ordered[1].source_frame_idx, ordered[0].source_frame_idx)
        self.assertEqual(ordered[1].timestamp_sec, 1 / 30)
        self.assertEqual(self.adapter.get_ordered_frames_by_video("L21_V999"), ())
        self.assertEqual(self.adapter.list_video_ids(), ("L21_V001", "L21_V002"))
        self.assertEqual(
            self.adapter.get_ordered_frames_by_video("L21_V002")[0].image_rel_path,
            "keyframes/L21_V002/ảnh-001.jpg",
        )

    def test_object_filter_by_label_and_confidence(self) -> None:
        result = self.adapter.get_objects_by_frame_ids(
            ["L21_V001_002", "L21_V001_001"],
            label="person",
            min_confidence=0.5,
        )
        self.assertEqual(len(result["L21_V001_002"]), 1)
        self.assertEqual(result["L21_V001_002"][0].class_mid, "/m/01g317")
        self.assertEqual(result["L21_V001_001"], ())
        by_mid = self.adapter.get_objects_by_frame_ids(
            ["L21_V001_002"], label="/m/0k4j", min_confidence=0.5
        )
        self.assertEqual(by_mid["L21_V001_002"][0].label_normalized, "car")

    def test_empty_input_does_not_query_full_table(self) -> None:
        self.assertEqual(self.adapter.get_frames_by_ids([]), {})
        self.assertEqual(self.adapter.get_objects_by_frame_ids([]), {})

    def test_invalid_caller_inputs_are_domain_errors(self) -> None:
        for value in ("L21_V001_001", b"id"):
            with self.assertRaises(InvalidQueryError):
                self.adapter.get_frames_by_ids(value)  # type: ignore[arg-type]
        with self.assertRaises(InvalidQueryError):
            self.adapter.get_objects_by_frame_ids(["L21_V001_001"], label=123)  # type: ignore[arg-type]
        for value in ("bad", True):
            with self.assertRaises(InvalidQueryError):
                self.adapter.get_objects_by_frame_ids(
                    ["L21_V001_001"], min_confidence=value  # type: ignore[arg-type]
                )
        with self.assertRaises(InvalidQueryError):
            self.adapter.sample_records("metadata", "frame_id", 1)  # type: ignore[arg-type]
        with self.assertRaises(InvalidQueryError):
            self.adapter.sample_records("metadata", ("frame_id",), True)

    def test_ids_are_parameter_bound(self) -> None:
        malicious = "x') OR 1=1 --"
        self.assertEqual(self.adapter.get_frames_by_ids([malicious]), {})
        self.assertEqual(len(self.adapter.get_ordered_frames_by_video("L21_V001")), 3)

    def test_connection_is_read_only(self) -> None:
        with self.assertRaises(sqlite3.OperationalError):
            self.adapter._conn().execute(
                "INSERT INTO videos(video_id, video_rel_path) VALUES ('x','x')"
            )

    def test_table_columns(self) -> None:
        self.assertEqual(
            set(self.adapter.table_columns("metadata")),
            {
                "frame_id", "video_id", "keyframe_no", "local_index",
                "pts_time_sec", "fps", "source_frame_idx", "image_rel_path",
            },
        )

    def test_invalid_object_and_metadata_rows_translate_to_contract_error(self) -> None:
        connection = self.adapter._conn()
        connection.execute("PRAGMA query_only=OFF")
        connection.execute(
            """
            INSERT INTO objects(
                frame_id, label_display, label_normalized, confidence,
                x_min, y_min, x_max, y_max, model_source
            ) VALUES (?,?,?,?,?,?,?,?,?)
            """,
            ("L21_V001_001", "Bad", "bad", 1.5, 0, 0, 1, 1, "fixture"),
        )
        with self.assertRaises(ContractMismatchError):
            self.adapter.get_objects_by_frame_ids(["L21_V001_001"])
        connection.execute(
            "INSERT INTO metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("L21_V001_004", "L21_V001", 4, 3, "not-a-number", 30.0, 90, "keyframes/L21_V001/004.jpg"),
        )
        connection.execute("PRAGMA query_only=ON")
        with self.assertRaises(ContractMismatchError):
            self.adapter.get_frames_by_ids(["L21_V001_004"])

    def test_reproducible_file_fixture_opens_read_only(self) -> None:
        path = Path(f".online-test-{uuid4().hex}.db")
        try:
            create_sqlite_fixture(
                path,
                video_rows=(("L21_V001", "video/L21_V001.mp4"),),
                metadata_rows=(("L21_V001_001", "L21_V001", 1, 0, 0.0, 30.0, 0, "keyframes/L21_V001/001.jpg"),),
                object_rows=(
                    ("L21_V001_001", "Person", "person", "/m/01g317", "0", 0.9, 0.1, 0.1, 0.9, 0.9, "fixture"),
                ),
            )
            adapter = SQLiteReadAdapter(SQLiteResourceConfig(path=path))
            adapter.connect()
            self.assertEqual(
                adapter.get_ordered_frames_by_video("L21_V001")[0].frame_id,
                "L21_V001_001",
            )
            adapter.close()
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
