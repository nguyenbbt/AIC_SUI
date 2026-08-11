"""Read-only SQLite metadata and object adapter."""

from __future__ import annotations

import math
import sqlite3
import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path

from online.config import SQLiteResourceConfig
from online.domain.candidates import ObjectDetection
from online.domain.errors import ContractMismatchError, InvalidQueryError, ResourceUnavailableError
from online.ports.records import FrameMetadata, ObjectLabelStat, VideoMetadata

from ._errors import call_backend


class SQLiteReadAdapter:
    def __init__(
        self,
        config: SQLiteResourceConfig,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        self.config = config
        self._connection = connection
        self._owns_connection = connection is None
        self._lock = threading.RLock()
        if connection is not None:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only=ON")

    @property
    def connected(self) -> bool:
        return self._connection is not None

    @staticmethod
    def _quote_identifier(identifier: str) -> str:
        return f'"{identifier}"'

    def connect(self) -> None:
        with self._lock:
            if self._connection is not None:
                return
            path = Path(self.config.path).expanduser().resolve()
            if not path.is_file():
                raise ResourceUnavailableError(
                    "SQLite metadata database does not exist",
                    details={"resource": "sqlite"},
                )
            uri = f"file:{path.as_posix()}?mode=ro"
            try:
                connection = sqlite3.connect(
                    uri,
                    uri=True,
                    timeout=self.config.timeout_sec,
                    check_same_thread=False,
                )
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA query_only=ON")
            except sqlite3.Error as exc:
                raise ResourceUnavailableError(
                    "Unable to open SQLite metadata database read-only",
                    details={"resource": "sqlite"},
                ) from exc
            self._connection = connection

    def close(self) -> None:
        with self._lock:
            if self._connection is not None and self._owns_connection:
                self._connection.close()
            self._connection = None

    def _conn(self) -> sqlite3.Connection:
        if self._connection is None:
            raise ResourceUnavailableError("SQLite adapter is not connected")
        return self._connection

    def health_check(self) -> None:
        with self._lock:
            call_backend(
                "health_check",
                "sqlite",
                lambda: self._conn().execute("SELECT 1").fetchone(),
            )

    def _chunks(self, values: Sequence[str]) -> Iterator[tuple[str, ...]]:
        for start in range(0, len(values), self.config.batch_size):
            yield tuple(values[start : start + self.config.batch_size])

    @staticmethod
    def _validate_ids(values: Sequence[str], name: str) -> tuple[str, ...]:
        if isinstance(values, (str, bytes)):
            raise InvalidQueryError(f"{name} must be a sequence of strings")
        try:
            result = tuple(dict.fromkeys(values))
        except (TypeError, ValueError) as exc:
            raise InvalidQueryError(f"{name} must be a sequence of strings") from exc
        if any(
            not isinstance(value, str)
            or not value.strip()
            or value != value.strip()
            for value in result
        ):
            raise InvalidQueryError(f"{name} must contain non-empty strings")
        return result

    def get_frames_by_ids(self, frame_ids: Sequence[str]) -> Mapping[str, FrameMetadata]:
        ids = self._validate_ids(frame_ids, "frame_ids")
        if not ids:
            return {}
        table = self._quote_identifier(self.config.metadata_table)
        output: dict[str, FrameMetadata] = {}
        with self._lock:
            for chunk in self._chunks(ids):
                placeholders = ",".join("?" for _ in chunk)
                sql = (
                    f"SELECT frame_id, video_id, shot_id, source_frame_idx, "
                    f"timestamp, image_rel_path FROM {table} "
                    f"WHERE frame_id IN ({placeholders})"
                )
                rows = call_backend(
                    "get_frames_by_ids", "sqlite", lambda sql=sql, chunk=chunk: self._conn().execute(sql, chunk).fetchall()
                )
                for row in rows:
                    metadata = self._frame_from_row(row)
                    output[metadata.frame_id] = metadata
        return output

    def get_ordered_frames_by_video(self, video_id: str) -> Sequence[FrameMetadata]:
        if (
            not isinstance(video_id, str)
            or not video_id.strip()
            or video_id != video_id.strip()
        ):
            raise InvalidQueryError("video_id must not be empty")
        table = self._quote_identifier(self.config.metadata_table)
        sql = (
            f"SELECT frame_id, video_id, shot_id, source_frame_idx, "
            f"timestamp, image_rel_path FROM {table} "
            "WHERE video_id = ? ORDER BY timestamp ASC, frame_id ASC"
        )
        with self._lock:
            rows = call_backend(
                "get_ordered_frames_by_video",
                "sqlite",
                lambda: self._conn().execute(sql, (video_id,)).fetchall(),
            )
        return tuple(self._frame_from_row(row) for row in rows)

    def get_videos_by_ids(self, video_ids: Sequence[str]) -> Mapping[str, VideoMetadata]:
        ids = self._validate_ids(video_ids, "video_ids")
        if not ids:
            return {}
        table = self._quote_identifier(self.config.videos_table)
        output: dict[str, VideoMetadata] = {}
        with self._lock:
            for chunk in self._chunks(ids):
                placeholders = ",".join("?" for _ in chunk)
                sql = (
                    "SELECT video_id, source_video_rel_path, fps, duration_sec, "
                    f"frame_count, width, height FROM {table} "
                    f"WHERE video_id IN ({placeholders})"
                )
                rows = call_backend(
                    "get_videos_by_ids",
                    "sqlite",
                    lambda sql=sql, chunk=chunk: self._conn().execute(sql, chunk).fetchall(),
                )
                for row in rows:
                    video = self._video_from_row(row)
                    output[video.video_id] = video
        return output

    def get_objects_by_frame_ids(
        self,
        frame_ids: Sequence[str],
        *,
        label: str | None = None,
        min_confidence: float = 0.0,
    ) -> Mapping[str, Sequence[ObjectDetection]]:
        ids = self._validate_ids(frame_ids, "frame_ids")
        if not ids:
            return {}
        if label is not None and (
            not isinstance(label, str)
            or not label.strip()
            or label != label.strip()
        ):
            raise InvalidQueryError("label must not be empty")
        if (
            isinstance(min_confidence, bool)
            or not isinstance(min_confidence, (int, float))
            or not math.isfinite(min_confidence)
            or not 0.0 <= min_confidence <= 1.0
        ):
            raise InvalidQueryError("min_confidence must be within [0, 1]")

        objects_table = self._quote_identifier(self.config.objects_table)
        metadata_table = self._quote_identifier(self.config.metadata_table)
        videos_table = self._quote_identifier(self.config.videos_table)
        output: dict[str, list[ObjectDetection]] = {frame_id: [] for frame_id in ids}
        with self._lock:
            for chunk in self._chunks(ids):
                placeholders = ",".join("?" for _ in chunk)
                predicates = [f"o.frame_id IN ({placeholders})", "o.confidence >= ?"]
                parameters: list[object] = [*chunk, min_confidence]
                if label is not None:
                    predicates.append("o.label = ?")
                    parameters.append(label.casefold())
                sql = (
                    "SELECT o.frame_id, o.label, o.confidence, o.x_min, o.y_min, "
                    "o.x_max, o.y_max, o.model_source, v.width, v.height "
                    f"FROM {objects_table} AS o "
                    f"JOIN {metadata_table} AS m ON m.frame_id = o.frame_id "
                    f"JOIN {videos_table} AS v ON v.video_id = m.video_id "
                    f"WHERE {' AND '.join(predicates)} "
                    "ORDER BY o.frame_id ASC, o.confidence DESC, o.label ASC, o.id ASC"
                )
                rows = call_backend(
                    "get_objects_by_frame_ids",
                    "sqlite",
                    lambda sql=sql, parameters=parameters: self._conn().execute(sql, parameters).fetchall(),
                )
                for row in rows:
                    try:
                        frame_id = row["frame_id"]
                        label_value = row["label"]
                        if (
                            not isinstance(frame_id, str)
                            or not frame_id.strip()
                            or not isinstance(label_value, str)
                            or not label_value.strip()
                        ):
                            raise ValueError("invalid object identifier/label")
                        output[frame_id].append(
                            ObjectDetection(
                                label=label_value.casefold(),
                                confidence=self._row_float(row, "confidence"),
                                x_min=self._normalized_coordinate(row, "x_min", "width"),
                                y_min=self._normalized_coordinate(row, "y_min", "height"),
                                x_max=self._normalized_coordinate(row, "x_max", "width"),
                                y_max=self._normalized_coordinate(row, "y_max", "height"),
                                model_source=(
                                    None
                                    if row["model_source"] is None
                                    else self._row_text(row, "model_source")
                                ),
                            )
                        )
                    except Exception as exc:
                        if isinstance(exc, ContractMismatchError):
                            raise
                        raise ContractMismatchError(
                            "Invalid object row returned by SQLite"
                        ) from exc
        return {frame_id: tuple(objects) for frame_id, objects in output.items()}

    def list_object_labels(self) -> Sequence[ObjectLabelStat]:
        """Return the exact object vocabulary present in the active SQLite index."""

        table = self._quote_identifier(self.config.objects_table)
        sql = (
            f"SELECT LOWER(TRIM(label)) AS label, COUNT(*) AS detection_count "
            f"FROM {table} WHERE TRIM(label) <> '' "
            "GROUP BY LOWER(TRIM(label)) ORDER BY label ASC"
        )
        with self._lock:
            rows = call_backend(
                "list_object_labels",
                "sqlite",
                lambda: self._conn().execute(sql).fetchall(),
            )
        try:
            return tuple(
                ObjectLabelStat(
                    label=self._row_text(row, "label"),
                    detection_count=self._row_int(row, "detection_count"),
                )
                for row in rows
            )
        except Exception as exc:
            raise ContractMismatchError(
                "Invalid object catalog row returned by SQLite"
            ) from exc

    @staticmethod
    def _frame_from_row(row: sqlite3.Row) -> FrameMetadata:
        try:
            frame_id = row["frame_id"]
            video_id = row["video_id"]
            if (
                not isinstance(frame_id, str)
                or not frame_id.strip()
                or not isinstance(video_id, str)
                or not video_id.strip()
                or row["shot_id"] is None
                or row["source_frame_idx"] is None
                or row["timestamp"] is None
                or row["image_rel_path"] is None
            ):
                raise ValueError("missing metadata field")
            return FrameMetadata(
                frame_id=frame_id,
                video_id=video_id,
                shot_id=SQLiteReadAdapter._row_int(row, "shot_id"),
                source_frame_idx=SQLiteReadAdapter._row_int(row, "source_frame_idx"),
                timestamp_sec=SQLiteReadAdapter._row_float(row, "timestamp"),
                image_rel_path=SQLiteReadAdapter._row_text(row, "image_rel_path"),
            )
        except Exception as exc:
            raise ContractMismatchError("Invalid metadata row returned by SQLite") from exc

    @staticmethod
    def _video_from_row(row: sqlite3.Row) -> VideoMetadata:
        try:
            return VideoMetadata(
                video_id=SQLiteReadAdapter._row_text(row, "video_id"),
                source_video_rel_path=SQLiteReadAdapter._row_text(
                    row, "source_video_rel_path"
                ),
                fps=SQLiteReadAdapter._row_float(row, "fps"),
                duration_sec=SQLiteReadAdapter._row_float(row, "duration_sec"),
                frame_count=SQLiteReadAdapter._row_int(row, "frame_count"),
                width=SQLiteReadAdapter._row_int(row, "width"),
                height=SQLiteReadAdapter._row_int(row, "height"),
            )
        except Exception as exc:
            raise ContractMismatchError("Invalid video row returned by SQLite") from exc

    @staticmethod
    def _normalized_coordinate(
        row: sqlite3.Row,
        coordinate_field: str,
        size_field: str,
    ) -> float:
        coordinate = SQLiteReadAdapter._row_float(row, coordinate_field)
        size = SQLiteReadAdapter._row_int(row, size_field)
        if size <= 0 or coordinate < 0.0 or coordinate > size:
            raise ValueError("object bounding box is outside video dimensions")
        return coordinate / size

    @staticmethod
    def _row_text(row: sqlite3.Row, field: str) -> str:
        value = row[field]
        if (
            not isinstance(value, str)
            or not value.strip()
            or value != value.strip()
        ):
            raise ValueError(f"{field} must be canonical text")
        return value

    @staticmethod
    def _row_float(row: sqlite3.Row, field: str) -> float:
        value = row[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field} must be numeric")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{field} must be finite")
        return result

    @staticmethod
    def _row_int(row: sqlite3.Row, field: str) -> int:
        value = row[field]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field} must be an integer")
        return value

    def table_columns(self, table_name: str) -> Mapping[str, str]:
        if table_name not in {
            self.config.videos_table,
            self.config.metadata_table,
            self.config.objects_table,
        }:
            raise InvalidQueryError("table is not managed by the Online adapter")
        table = self._quote_identifier(table_name)
        with self._lock:
            rows = call_backend(
                "table_columns",
                "sqlite",
                lambda: self._conn().execute(f"PRAGMA table_info({table})").fetchall(),
            )
        return {str(row["name"]): str(row["type"]).upper() for row in rows}

    def sample_records(
        self,
        table_name: str,
        fields: Sequence[str],
        limit: int,
    ) -> Sequence[Mapping[str, object]]:
        if table_name not in {
            self.config.videos_table,
            self.config.metadata_table,
            self.config.objects_table,
        }:
            raise InvalidQueryError("table is not managed by the Online adapter")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise InvalidQueryError("limit must be >= 1")
        if isinstance(fields, (str, bytes)) or not fields or any(
            not isinstance(field, str)
            or not field.strip()
            or field != field.strip()
            for field in fields
        ):
            raise InvalidQueryError("fields must contain non-empty field names")
        columns = self.table_columns(table_name)
        if any(field not in columns for field in fields):
            raise ContractMismatchError("Requested sample field is missing from SQLite table")
        selected = ", ".join(self._quote_identifier(field) for field in fields)
        table = self._quote_identifier(table_name)
        with self._lock:
            rows = call_backend(
                "sample_records",
                "sqlite",
                lambda: self._conn().execute(
                    f"SELECT {selected} FROM {table} ORDER BY rowid ASC LIMIT ?", (limit,)
                ).fetchall(),
            )
        return tuple({field: row[field] for field in fields} for row in rows)

    def iter_records(
        self,
        table_name: str,
        fields: Sequence[str],
        *,
        batch_size: int,
    ) -> Iterable[tuple[Mapping[str, object], ...]]:
        """Stream a managed table in stable rowid order without unbounded memory."""

        if table_name not in {
            self.config.videos_table,
            self.config.metadata_table,
            self.config.objects_table,
        }:
            raise InvalidQueryError("table is not managed by the Online adapter")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise InvalidQueryError("batch_size must be a positive integer")
        if isinstance(fields, (str, bytes)) or not fields or any(
            not isinstance(field, str)
            or not field.strip()
            or field != field.strip()
            for field in fields
        ):
            raise InvalidQueryError("fields must contain non-empty field names")
        columns = self.table_columns(table_name)
        if any(field not in columns for field in fields):
            raise ContractMismatchError("Requested audit field is missing from SQLite table")
        selected = ", ".join(self._quote_identifier(field) for field in fields)
        table = self._quote_identifier(table_name)

        def generate() -> Iterable[tuple[Mapping[str, object], ...]]:
            last_rowid = 0
            while True:
                with self._lock:
                    rows = call_backend(
                        "iter_records",
                        "sqlite",
                        lambda last_rowid=last_rowid: self._conn().execute(
                            f"SELECT rowid AS _audit_rowid, {selected} FROM {table} "
                            "WHERE rowid > ? ORDER BY rowid ASC LIMIT ?",
                            (last_rowid, batch_size),
                        ).fetchall(),
                    )
                if not rows:
                    break
                batch = tuple(
                    {field: row[field] for field in fields}
                    for row in rows
                )
                if not batch:
                    raise ContractMismatchError("SQLite audit iterator returned an empty batch")
                next_rowid = rows[-1]["_audit_rowid"]
                if (
                    isinstance(next_rowid, bool)
                    or not isinstance(next_rowid, int)
                    or next_rowid <= last_rowid
                ):
                    raise ContractMismatchError("SQLite audit rowid did not advance")
                last_rowid = next_rowid
                yield batch

        return generate()
