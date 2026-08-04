"""Read-only SQLite metadata and object adapter."""

from __future__ import annotations

import math
import sqlite3
import threading
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

from online.config import SQLiteResourceConfig
from online.domain.candidates import ObjectDetection
from online.domain.errors import ContractMismatchError, InvalidQueryError, ResourceUnavailableError
from online.ports.records import FrameMetadata

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
                    "SELECT frame_id, video_id, keyframe_no, local_index, "
                    "pts_time_sec, fps, source_frame_idx, image_rel_path "
                    f"FROM {table} "
                    f"WHERE frame_id IN ({placeholders})"
                )
                rows = call_backend(
                    "get_frames_by_ids", "sqlite", lambda sql=sql, chunk=chunk: self._conn().execute(sql, chunk).fetchall()
                )
                for row in rows:
                    metadata = self._frame_from_row(row)
                    output[metadata.frame_id] = metadata
        return output

    def list_video_ids(self) -> Sequence[str]:
        table = self._quote_identifier(self.config.videos_table)
        with self._lock:
            rows = call_backend(
                "list_video_ids",
                "sqlite",
                lambda: self._conn().execute(
                    f"SELECT video_id FROM {table} ORDER BY video_id ASC"
                ).fetchall(),
            )
        try:
            video_ids = tuple(self._row_text(row, "video_id") for row in rows)
        except Exception as exc:
            raise ContractMismatchError(
                "Invalid video row returned by SQLite"
            ) from exc
        if len(video_ids) != len(set(video_ids)):
            raise ContractMismatchError("SQLite videos table contains duplicate video_id")
        return video_ids

    def get_ordered_frames_by_video(self, video_id: str) -> Sequence[FrameMetadata]:
        if (
            not isinstance(video_id, str)
            or not video_id.strip()
            or video_id != video_id.strip()
        ):
            raise InvalidQueryError("video_id must not be empty")
        table = self._quote_identifier(self.config.metadata_table)
        sql = (
            "SELECT frame_id, video_id, keyframe_no, local_index, "
            "pts_time_sec, fps, source_frame_idx, image_rel_path "
            f"FROM {table} WHERE video_id = ? "
            "ORDER BY local_index ASC, frame_id ASC"
        )
        with self._lock:
            rows = call_backend(
                "get_ordered_frames_by_video",
                "sqlite",
                lambda: self._conn().execute(sql, (video_id,)).fetchall(),
            )
        return tuple(self._frame_from_row(row) for row in rows)

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

        table = self._quote_identifier(self.config.objects_table)
        output: dict[str, list[ObjectDetection]] = {frame_id: [] for frame_id in ids}
        with self._lock:
            for chunk in self._chunks(ids):
                placeholders = ",".join("?" for _ in chunk)
                predicates = [f"frame_id IN ({placeholders})", "confidence >= ?"]
                parameters: list[object] = [*chunk, min_confidence]
                if label is not None:
                    predicates.append("(label_normalized = ? OR class_mid = ?)")
                    parameters.extend((label, label))
                sql = (
                    "SELECT frame_id, label_display, label_normalized, class_mid, "
                    "class_label_id, confidence, x_min, y_min, x_max, y_max, model_source "
                    f"FROM {table} WHERE {' AND '.join(predicates)} "
                    "ORDER BY frame_id ASC, confidence DESC, label_normalized ASC, id ASC"
                )
                rows = call_backend(
                    "get_objects_by_frame_ids",
                    "sqlite",
                    lambda sql=sql, parameters=parameters: self._conn().execute(sql, parameters).fetchall(),
                )
                for row in rows:
                    try:
                        frame_id = row["frame_id"]
                        if (
                            not isinstance(frame_id, str)
                            or not frame_id.strip()
                        ):
                            raise ValueError("invalid object frame identifier")
                        output[frame_id].append(
                            ObjectDetection(
                                label_display=self._row_text(row, "label_display"),
                                label_normalized=self._row_text(row, "label_normalized"),
                                class_mid=self._row_optional_text(row, "class_mid"),
                                class_label_id=self._row_optional_text(
                                    row, "class_label_id"
                                ),
                                confidence=self._row_float(row, "confidence"),
                                x_min=self._row_float(row, "x_min"),
                                y_min=self._row_float(row, "y_min"),
                                x_max=self._row_float(row, "x_max"),
                                y_max=self._row_float(row, "y_max"),
                                model_source=self._row_text(row, "model_source"),
                            )
                        )
                    except Exception as exc:
                        if isinstance(exc, ContractMismatchError):
                            raise
                        raise ContractMismatchError(
                            "Invalid object row returned by SQLite"
                        ) from exc
        return {frame_id: tuple(objects) for frame_id, objects in output.items()}

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
            ):
                raise ValueError("missing metadata field")
            return FrameMetadata(
                frame_id=frame_id,
                video_id=video_id,
                keyframe_no=SQLiteReadAdapter._row_int(row, "keyframe_no"),
                local_index=SQLiteReadAdapter._row_int(row, "local_index"),
                timestamp_sec=SQLiteReadAdapter._row_float(row, "pts_time_sec"),
                fps=SQLiteReadAdapter._row_float(row, "fps"),
                source_frame_idx=SQLiteReadAdapter._row_int(row, "source_frame_idx"),
                image_rel_path=SQLiteReadAdapter._row_text(row, "image_rel_path"),
            )
        except Exception as exc:
            raise ContractMismatchError("Invalid metadata row returned by SQLite") from exc

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
    def _row_optional_text(row: sqlite3.Row, field: str) -> str | None:
        return None if row[field] is None else SQLiteReadAdapter._row_text(row, field)

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
