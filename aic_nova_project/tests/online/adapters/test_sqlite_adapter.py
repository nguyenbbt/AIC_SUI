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
    source_video_rel_path TEXT NOT NULL,
    fps REAL NOT NULL,
    duration_sec REAL NOT NULL,
    frame_count INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL
);
CREATE TABLE metadata (
    frame_id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    shot_id INTEGER NOT NULL,
    source_frame_idx INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    image_rel_path TEXT NOT NULL
);
CREATE TABLE objects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id TEXT NOT NULL,
    label TEXT NOT NULL,
    confidence REAL NOT NULL,
    x_min REAL NOT NULL,
    y_min REAL NOT NULL,
    x_max REAL NOT NULL,
    y_max REAL NOT NULL,
    model_source TEXT
);
"""


class SQLiteAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        connection = sqlite3.connect(":memory:")
        connection.executescript(SCHEMA)
        connection.execute(
            "INSERT INTO videos VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("V001", "videos/V001.mp4", 30.0, 20.0, 600, 100, 50),
        )
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("V001_00001_050", "V001", 1, 300, 10.0, "keyframes/V001/1.webp"),
                ("V001_00000_015", "V001", 0, 45, 1.5, "keyframes/V001/0a.webp"),
                ("V001_00000_050", "V001", 0, 150, 5.0, "keyframes/V001/0b.webp"),
            ],
        )
        connection.executemany(
            "INSERT INTO objects (frame_id,label,confidence,x_min,y_min,x_max,y_max,model_source) VALUES (?,?,?,?,?,?,?,?)",
            [
                ("V001_00000_050", "person", 0.95, 0, 0, 10, 20, "yolo"),
                ("V001_00000_050", "person", 0.35, 20, 0, 30, 20, "detr"),
                ("V001_00000_050", "car", 0.80, 40, 0, 80, 20, "detr"),
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
            ["V001_00000_015", "missing", "V001_00001_050"]
        )
        self.assertEqual(set(result), {"V001_00000_015", "V001_00001_050"})
        ordered = self.adapter.get_ordered_frames_by_video("V001")
        self.assertEqual([frame.timestamp_sec for frame in ordered], [1.5, 5.0, 10.0])
        self.assertEqual(self.adapter.get_ordered_frames_by_video("V999"), ())

    def test_object_filter_by_label_and_confidence(self) -> None:
        result = self.adapter.get_objects_by_frame_ids(
            ["V001_00000_050", "V001_00000_015"],
            label="person",
            min_confidence=0.5,
        )
        self.assertEqual(len(result["V001_00000_050"]), 1)
        self.assertEqual(result["V001_00000_015"], ())
        detection = result["V001_00000_050"][0]
        self.assertEqual((detection.x_min, detection.x_max), (0.0, 0.1))

    def test_object_catalog_is_canonical_counted_and_sorted(self) -> None:
        labels = self.adapter.list_object_labels()
        self.assertEqual(
            [(item.label, item.detection_count) for item in labels],
            [("car", 1), ("person", 2)],
        )

    def test_empty_input_does_not_query_full_table(self) -> None:
        self.assertEqual(self.adapter.get_frames_by_ids([]), {})
        self.assertEqual(self.adapter.get_objects_by_frame_ids([]), {})

    def test_invalid_caller_inputs_are_domain_errors(self) -> None:
        for value in ("V001_00000_015", b"id"):
            with self.assertRaises(InvalidQueryError):
                self.adapter.get_frames_by_ids(value)  # type: ignore[arg-type]
        with self.assertRaises(InvalidQueryError):
            self.adapter.get_objects_by_frame_ids(["V001_00000_015"], label=123)  # type: ignore[arg-type]
        for value in ("bad", True):
            with self.assertRaises(InvalidQueryError):
                self.adapter.get_objects_by_frame_ids(
                    ["V001_00000_015"], min_confidence=value  # type: ignore[arg-type]
                )
        with self.assertRaises(InvalidQueryError):
            self.adapter.sample_records("metadata", "frame_id", 1)  # type: ignore[arg-type]
        with self.assertRaises(InvalidQueryError):
            self.adapter.sample_records("metadata", ("frame_id",), True)

    def test_ids_are_parameter_bound(self) -> None:
        malicious = "x') OR 1=1 --"
        self.assertEqual(self.adapter.get_frames_by_ids([malicious]), {})
        self.assertEqual(len(self.adapter.get_ordered_frames_by_video("V001")), 3)

    def test_connection_is_read_only(self) -> None:
        with self.assertRaises(sqlite3.OperationalError):
            self.adapter._conn().execute(
                "INSERT INTO metadata VALUES ('x_00000_050','x',0,0,0,'x.webp')"
            )

    def test_table_columns(self) -> None:
        self.assertEqual(
            set(self.adapter.table_columns("metadata")),
            {"frame_id", "video_id", "shot_id", "source_frame_idx", "timestamp", "image_rel_path"},
        )

    def test_full_record_iterator_is_bounded_and_complete(self) -> None:
        batches = tuple(
            self.adapter.iter_records(
                "metadata", ("frame_id", "source_frame_idx"), batch_size=2
            )
        )
        self.assertEqual([len(batch) for batch in batches], [2, 1])
        self.assertEqual(
            {record["frame_id"] for batch in batches for record in batch},
            {"V001_00000_015", "V001_00000_050", "V001_00001_050"},
        )
        with self.assertRaises(InvalidQueryError):
            tuple(
                self.adapter.iter_records(
                    "metadata", ("frame_id",), batch_size=0
                )
            )

    def test_invalid_object_and_metadata_rows_translate_to_contract_error(self) -> None:
        connection = self.adapter._conn()
        connection.execute("PRAGMA query_only=OFF")
        connection.execute(
            "INSERT INTO objects (frame_id,label,confidence,x_min,y_min,x_max,y_max) VALUES (?,?,?,?,?,?,?)",
            ("V001_00000_015", "bad", 1.5, 0, 0, 1, 1),
        )
        with self.assertRaises(ContractMismatchError):
            self.adapter.get_objects_by_frame_ids(["V001_00000_015"])
        connection.execute(
            "INSERT INTO metadata VALUES (?, ?, ?, ?, ?, ?)",
            ("V001_00001_099", "V001", 1, 31, "not-a-number", "keyframes/V001/bad.webp"),
        )
        connection.execute("PRAGMA query_only=ON")
        with self.assertRaises(ContractMismatchError):
            self.adapter.get_frames_by_ids(["V001_00001_099"])

    def test_reproducible_file_fixture_opens_read_only(self) -> None:
        path = Path(f".online-test-{uuid4().hex}.db")
        try:
            create_sqlite_fixture(
                path,
                video_rows=(("V001", "videos/V001.mp4", 30.0, 10.0, 300, 100, 50),),
                metadata_rows=(("V001_00000_015", "V001", 0, 45, 1.5, "keyframes/V001/0.webp"),),
                object_rows=(
                    ("V001_00000_015", "person", 0.9, 0, 0, 10, 10, "fixture"),
                ),
            )
            adapter = SQLiteReadAdapter(SQLiteResourceConfig(path=path))
            adapter.connect()
            self.assertEqual(
                adapter.get_ordered_frames_by_video("V001")[0].frame_id,
                "V001_00000_015",
            )
            adapter.close()
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
