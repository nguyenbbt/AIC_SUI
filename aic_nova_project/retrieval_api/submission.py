"""BTC-compliant AIC 2026 preliminary submission serialization.

The organizer requires one UTF-8, headerless comma-delimited CSV per query and
one ZIP whose root contains a ``submission/`` directory. Internal canonical
``frame_id`` values are never submitted: the organizer-facing integer is the
Offline-produced ``source_frame_idx``.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Sequence
from io import BytesIO, StringIO
from typing import Annotated, Literal
from zipfile import ZIP_DEFLATED, ZipFile

from pydantic import Field, field_validator, model_validator

from online.domain.base import NonEmptyStr, StrictFrozenModel, StrictIntValue
from online.domain.candidates import FusedFrameCandidate
from online.domain.errors import ContractMismatchError, InvalidQueryError
from online.domain.trake import TRAKEVideoResult
from online.domain.vqa import ImageEvidence, VLMResponse, VLMResponseStatus


MAX_ANSWERS_PER_QUERY = 100
MAX_VQA_ANSWER_CHARS = 100
OrganizerFrameId = Annotated[StrictIntValue, Field(strict=True, ge=0)]

_QUERY_FILENAME = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_-]*-(kis|qa|trake)(?:\.(txt|csv))?$",
    re.IGNORECASE,
)
_ARCHIVE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class _OrganizerRow(StrictFrozenModel):
    video_id: NonEmptyStr

    @field_validator("video_id")
    @classmethod
    def validate_video_id(cls, value: str) -> str:
        if value.lower().endswith(".mp4"):
            raise ValueError("video_id must not include the .mp4 extension")
        if _VIDEO_ID.fullmatch(value) is None:
            raise ValueError("video_id must be a plain video filename stem")
        return value


class KISSubmissionRow(_OrganizerRow):
    frame_id: OrganizerFrameId


class VQASubmissionRow(KISSubmissionRow):
    answer: str = Field(min_length=1, max_length=MAX_VQA_ANSWER_CHARS)

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("answer must contain at least one non-whitespace character")
        # BTC preserves surrounding whitespace, so validate but return it unchanged.
        return value


class TRAKESubmissionRow(_OrganizerRow):
    frame_ids: tuple[OrganizerFrameId, ...] = Field(min_length=2)


class KISSubmissionFile(StrictFrozenModel):
    mode: Literal["kis"]
    query_filename: NonEmptyStr
    rows: tuple[KISSubmissionRow, ...] = Field(
        min_length=1, max_length=MAX_ANSWERS_PER_QUERY
    )

    @model_validator(mode="after")
    def validate_filename(self) -> "KISSubmissionFile":
        _validate_query_filename(self.query_filename, self.mode)
        return self


class VQASubmissionFile(StrictFrozenModel):
    mode: Literal["qa"]
    query_filename: NonEmptyStr
    rows: tuple[VQASubmissionRow, ...] = Field(
        min_length=1, max_length=MAX_ANSWERS_PER_QUERY
    )

    @model_validator(mode="after")
    def validate_filename(self) -> "VQASubmissionFile":
        _validate_query_filename(self.query_filename, self.mode)
        return self


class TRAKESubmissionFile(StrictFrozenModel):
    mode: Literal["trake"]
    query_filename: NonEmptyStr
    event_count: StrictIntValue = Field(ge=2)
    rows: tuple[TRAKESubmissionRow, ...] = Field(
        min_length=1, max_length=MAX_ANSWERS_PER_QUERY
    )

    @model_validator(mode="after")
    def validate_file_contract(self) -> "TRAKESubmissionFile":
        _validate_query_filename(self.query_filename, self.mode)
        if any(len(row.frame_ids) != self.event_count for row in self.rows):
            raise ValueError(
                "every TRAKE row must contain exactly one frame_id per requested event"
            )
        return self


SubmissionFile = Annotated[
    KISSubmissionFile | VQASubmissionFile | TRAKESubmissionFile,
    Field(discriminator="mode"),
]


class SubmissionPackageRequest(StrictFrozenModel):
    archive_name: str = Field(min_length=1)
    files: tuple[SubmissionFile, ...] = Field(min_length=1)

    @field_validator("archive_name")
    @classmethod
    def validate_archive_name(cls, value: str) -> str:
        stem = value[:-4] if value.lower().endswith(".zip") else value
        if not _ARCHIVE_NAME.fullmatch(stem):
            raise ValueError(
                "archive_name may contain ASCII letters, digits, underscore and hyphen only"
            )
        return stem

    @model_validator(mode="after")
    def validate_unique_query_files(self) -> "SubmissionPackageRequest":
        names = tuple(submission_csv_filename(item) for item in self.files)
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError("submission package contains duplicate query filenames")
        return self

    @property
    def download_filename(self) -> str:
        return f"{self.archive_name}.zip"


def serialize_kis_submissions(
    candidates: Sequence[FusedFrameCandidate],
    *,
    limit: int = MAX_ANSWERS_PER_QUERY,
) -> tuple[KISSubmissionRow, ...]:
    """Map internal source-frame identity to BTC's external ``frame_id`` name."""

    values = _validated_sequence(candidates, FusedFrameCandidate, "candidates")
    bounded_limit = _validated_limit(limit)
    return tuple(
        KISSubmissionRow(video_id=item.video_id, frame_id=item.source_frame_idx)
        for item in values[:bounded_limit]
    )


def serialize_vqa_submission(
    *,
    image: ImageEvidence,
    response: VLMResponse,
) -> VQASubmissionRow:
    """Build one Q&A row from explicitly selected source-frame evidence."""

    if not isinstance(image, ImageEvidence) or not isinstance(response, VLMResponse):
        raise ContractMismatchError("VQA submission requires validated image and response")
    if response.status is not VLMResponseStatus.ANSWERED or response.answer is None:
        raise ContractMismatchError("VQA submission requires an answered VLM response")
    if len(response.answer) > MAX_VQA_ANSWER_CHARS:
        raise InvalidQueryError("VQA answer must not exceed 100 characters")
    return VQASubmissionRow(
        video_id=image.video_id,
        frame_id=image.source_frame_idx,
        answer=response.answer,
    )


def serialize_trake_submissions(
    results: Sequence[TRAKEVideoResult],
    *,
    limit: int = MAX_ANSWERS_PER_QUERY,
) -> tuple[TRAKESubmissionRow, ...]:
    """Preserve event order while mapping every match to its original frame index."""

    values = _validated_sequence(results, TRAKEVideoResult, "results")
    bounded_limit = _validated_limit(limit)
    return tuple(
        TRAKESubmissionRow(
            video_id=item.video_id,
            frame_ids=tuple(match.source_frame_idx for match in item.sequence),
        )
        for item in values[:bounded_limit]
    )


def submission_csv_filename(item: SubmissionFile) -> str:
    """Map ``query-X-kind[.txt|.csv]`` to the required CSV filename."""

    match = _validate_query_filename(item.query_filename, item.mode)
    base = item.query_filename
    if match.group(2) is not None:
        base = base.rsplit(".", 1)[0]
    return f"{base}.csv"


def render_submission_csv(item: SubmissionFile) -> bytes:
    """Render one organizer CSV as UTF-8 without BOM and without a header."""

    output = StringIO(newline="")
    writer = csv.writer(
        output,
        delimiter=",",
        quotechar='"',
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )
    if isinstance(item, KISSubmissionFile):
        writer.writerows((row.video_id, row.frame_id) for row in item.rows)
    elif isinstance(item, VQASubmissionFile):
        writer.writerows((row.video_id, row.frame_id, row.answer) for row in item.rows)
    elif isinstance(item, TRAKESubmissionFile):
        writer.writerows((row.video_id, *row.frame_ids) for row in item.rows)
    else:  # pragma: no cover - guarded by the discriminated Pydantic union
        raise ContractMismatchError("unsupported submission file type")
    return output.getvalue().encode("utf-8")


def build_submission_zip(request: SubmissionPackageRequest) -> bytes:
    """Build and then structurally verify the exact organizer ZIP hierarchy."""

    if not isinstance(request, SubmissionPackageRequest):
        raise InvalidQueryError("request must be a validated SubmissionPackageRequest")
    output = BytesIO()
    with ZipFile(output, mode="w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("submission/", b"")
        for item in request.files:
            archive.writestr(
                f"submission/{submission_csv_filename(item)}",
                render_submission_csv(item),
            )
    payload = output.getvalue()
    _validate_built_zip(payload, request)
    return payload


def _validate_built_zip(payload: bytes, request: SubmissionPackageRequest) -> None:
    expected = {
        "submission/",
        *(f"submission/{submission_csv_filename(item)}" for item in request.files),
    }
    with ZipFile(BytesIO(payload), mode="r") as archive:
        if set(archive.namelist()) != expected:
            raise ContractMismatchError("submission ZIP hierarchy is invalid")
        for item in request.files:
            path = f"submission/{submission_csv_filename(item)}"
            raw = archive.read(path)
            if raw.startswith(b"\xef\xbb\xbf"):
                raise ContractMismatchError("submission CSV must be UTF-8 without BOM")
            decoded = raw.decode("utf-8")
            parsed = list(csv.reader(StringIO(decoded, newline=""), delimiter=","))
            if len(parsed) != len(item.rows):
                raise ContractMismatchError("submission CSV row count changed during export")


def _validate_query_filename(query_filename: str, mode: str) -> re.Match[str]:
    match = _QUERY_FILENAME.fullmatch(query_filename)
    if match is None:
        raise ValueError(
            "query_filename must look like query-1-kis.txt, query-2-qa.txt, "
            "or query-3-trake.txt"
        )
    if match.group(1).lower() != mode:
        raise ValueError("query filename suffix does not match submission mode")
    return match


def _validated_limit(limit: int) -> int:
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= MAX_ANSWERS_PER_QUERY
    ):
        raise InvalidQueryError("submission limit must be within [1, 100]")
    return limit


def _validated_sequence(values: object, item_type: type, name: str) -> tuple:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise InvalidQueryError(f"{name} must be a sequence")
    result = tuple(values)
    if any(not isinstance(item, item_type) for item in result):
        raise ContractMismatchError(f"{name} contains an invalid submission item")
    return result


__all__ = [
    "KISSubmissionFile",
    "KISSubmissionRow",
    "MAX_ANSWERS_PER_QUERY",
    "MAX_VQA_ANSWER_CHARS",
    "SubmissionPackageRequest",
    "TRAKESubmissionFile",
    "TRAKESubmissionRow",
    "VQASubmissionFile",
    "VQASubmissionRow",
    "build_submission_zip",
    "render_submission_csv",
    "serialize_kis_submissions",
    "serialize_trake_submissions",
    "serialize_vqa_submission",
    "submission_csv_filename",
]
