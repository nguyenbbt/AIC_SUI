from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path
from uuid import uuid4

from online.adapters.sqlite import SQLiteReadAdapter
from online.config import SQLiteResourceConfig
from online.domain.errors import ContractMismatchError
from online.testing import create_sqlite_fixture


SCHEMA = """
CREATE TABLE metadata (
    frame_id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    shot_id INTEGER NOT NULL,
    timestamp REAL NOT NULL
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
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?, ?, ?)",
            [
                ("V001_00001_050", "V001", 1, 10.0),
                ("V001_00000_015", "V001", 0, 1.5),
                ("V001_00000_050", "V001", 0, 5.0),
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

    def test_empty_input_does_not_query_full_table(self) -> None:
        self.assertEqual(self.adapter.get_frames_by_ids([]), {})
        self.assertEqual(self.adapter.get_objects_by_frame_ids([]), {})

    def test_ids_are_parameter_bound(self) -> None:
        malicious = "x') OR 1=1 --"
        self.assertEqual(self.adapter.get_frames_by_ids([malicious]), {})
        self.assertEqual(len(self.adapter.get_ordered_frames_by_video("V001")), 3)

    def test_connection_is_read_only(self) -> None:
        with self.assertRaises(sqlite3.OperationalError):
            self.adapter._conn().execute(
                "INSERT INTO metadata VALUES ('x','x',0,0)"
            )

    def test_table_columns(self) -> None:
        self.assertEqual(
            set(self.adapter.table_columns("metadata")),
            {"frame_id", "video_id", "shot_id", "timestamp"},
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
            "INSERT INTO metadata VALUES (?, ?, ?, ?)",
            ("V001_00001_099", "V001", 1, "not-a-number"),
        )
        connection.execute("PRAGMA query_only=ON")
        with self.assertRaises(ContractMismatchError):
            self.adapter.get_frames_by_ids(["V001_00001_099"])

    def test_reproducible_file_fixture_opens_read_only(self) -> None:
        path = Path(f".online-test-{uuid4().hex}.db")
        try:
            create_sqlite_fixture(
                path,
                metadata_rows=(("V001_00000_015", "V001", 0, 1.5),),
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
