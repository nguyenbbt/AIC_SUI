from __future__ import annotations

import csv
from io import BytesIO, StringIO
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from online.domain.errors import ContractMismatchError, InvalidQueryError
from online.domain.vqa import VLMConfidence, VLMResponse, VLMResponseStatus
from online.testing import build_advanced_modes_fixture
from retrieval_api.submission import (
    KISSubmissionFile,
    KISSubmissionRow,
    SubmissionPackageRequest,
    TRAKESubmissionFile,
    TRAKESubmissionRow,
    VQASubmissionFile,
    VQASubmissionRow,
    build_submission_zip,
    render_submission_csv,
    serialize_kis_submissions,
    serialize_trake_submissions,
    serialize_vqa_submission,
)
from retrieval_api.search_engine import create_app


def test_aic2026_logical_submission_rows_use_original_video_frame_indices() -> None:
    fixture = build_advanced_modes_fixture()
    kis = serialize_kis_submissions(fixture.ranked_vqa_candidates)
    assert kis[0].model_dump(mode="json") == {
        "video_id": fixture.ranked_vqa_candidates[0].video_id,
        "frame_id": fixture.ranked_vqa_candidates[0].source_frame_idx,
    }

    image = next(iter(fixture.images_by_frame_id.values()))
    response = VLMResponse(
        status=VLMResponseStatus.ANSWERED,
        answer="5",
        answer_type=fixture.vqa_question.answer_type,
        confidence=VLMConfidence.HIGH,
        evidence_ids=(image.evidence_id,),
    )
    vqa = serialize_vqa_submission(image=image, response=response)
    assert vqa.model_dump(mode="json") == {
        "video_id": image.video_id,
        "frame_id": image.source_frame_idx,
        "answer": "5",
    }


def test_trake_submission_preserves_event_order() -> None:
    import asyncio

    from online.trake import TRAKEService

    fixture = build_advanced_modes_fixture()
    service = TRAKEService(corpus=fixture.visual_corpus(), encoder=fixture.text_encoder())
    execution = asyncio.run(service.execute(fixture.trake_query))
    service.close()

    rows = serialize_trake_submissions(execution.results)

    assert rows[0].video_id == execution.results[0].video_id
    assert rows[0].frame_ids == tuple(
        match.source_frame_idx for match in execution.results[0].sequence
    )


def test_submission_contract_enforces_cap_and_answered_vqa() -> None:
    fixture = build_advanced_modes_fixture()
    with pytest.raises(InvalidQueryError):
        serialize_kis_submissions(fixture.ranked_vqa_candidates, limit=101)
    image = next(iter(fixture.images_by_frame_id.values()))
    response = VLMResponse(
        status=VLMResponseStatus.INSUFFICIENT_EVIDENCE,
        answer_type=fixture.vqa_question.answer_type,
        confidence=VLMConfidence.LOW,
    )
    with pytest.raises(ContractMismatchError):
        serialize_vqa_submission(image=image, response=response)


def test_csv_is_utf8_headerless_and_uses_standard_qa_escaping() -> None:
    item = VQASubmissionFile(
        mode="qa",
        query_filename="query-3-qa.txt",
        rows=(
            VQASubmissionRow(
                video_id="L02_V011",
                frame_id=1200,
                answer='Năm người, anh ấy nói "Xin chào"',
            ),
        ),
    )

    payload = render_submission_csv(item)

    assert not payload.startswith(b"\xef\xbb\xbf")
    decoded = payload.decode("utf-8")
    assert not decoded.startswith("video_id")
    assert list(csv.reader(StringIO(decoded, newline=""))) == [
        ["L02_V011", "1200", 'Năm người, anh ấy nói "Xin chào"']
    ]
    assert '"Năm người, anh ấy nói ""Xin chào"""' in decoded


def test_answer_keeps_surrounding_whitespace_but_is_limited_to_100_characters() -> None:
    row = VQASubmissionRow(video_id="L01_V001", frame_id=5, answer=" 5 ")
    assert row.answer == " 5 "

    with pytest.raises(ValidationError):
        VQASubmissionRow(video_id="L01_V001", frame_id=5, answer="x" * 101)
    with pytest.raises(ValidationError):
        VQASubmissionRow(video_id="L01_V001", frame_id=5, answer="   ")


def test_video_name_is_stem_and_frame_id_is_strict_non_negative_integer() -> None:
    with pytest.raises(ValidationError):
        KISSubmissionRow(video_id="L01_V001.mp4", frame_id=10)
    with pytest.raises(ValidationError):
        KISSubmissionRow(video_id="L01 V001", frame_id=10)
    with pytest.raises(ValidationError):
        KISSubmissionRow(video_id="../L01_V001", frame_id=10)
    with pytest.raises(ValidationError):
        KISSubmissionRow(video_id="L01_V001", frame_id="10")
    with pytest.raises(ValidationError):
        KISSubmissionRow(video_id="L01_V001", frame_id=-1)


def test_query_filename_suffix_and_100_row_cap_are_enforced() -> None:
    row = KISSubmissionRow(video_id="L01_V001", frame_id=10)
    with pytest.raises(ValidationError):
        KISSubmissionFile(mode="kis", query_filename="query-1-qa.txt", rows=(row,))
    with pytest.raises(ValidationError):
        KISSubmissionFile(
            mode="kis",
            query_filename="query-1-kis.txt",
            rows=tuple(row for _ in range(101)),
        )


def test_trake_requires_exactly_one_frame_per_event() -> None:
    row = TRAKESubmissionRow(
        video_id="L10_V001", frame_ids=(1200, 1850, 2100, 2450)
    )
    valid = TRAKESubmissionFile(
        mode="trake",
        query_filename="query-4-trake.txt",
        event_count=4,
        rows=(row,),
    )
    assert render_submission_csv(valid).decode("utf-8") == (
        "L10_V001,1200,1850,2100,2450\n"
    )

    with pytest.raises(ValidationError):
        TRAKESubmissionFile(
            mode="trake",
            query_filename="query-4-trake.txt",
            event_count=3,
            rows=(row,),
        )


def test_zip_contains_submission_directory_and_exact_csv_names() -> None:
    request = SubmissionPackageRequest(
        archive_name="TeamABCRound1",
        files=(
            KISSubmissionFile(
                mode="kis",
                query_filename="query-1-kis.txt",
                rows=(KISSubmissionRow(video_id="L00_V000", frame_id=1234),),
            ),
            VQASubmissionFile(
                mode="qa",
                query_filename="query-3-qa.txt",
                rows=(
                    VQASubmissionRow(
                        video_id="L01_V028", frame_id=3450, answer="Năm người"
                    ),
                ),
            ),
        ),
    )

    payload = build_submission_zip(request)

    with ZipFile(BytesIO(payload)) as archive:
        assert set(archive.namelist()) == {
            "submission/",
            "submission/query-1-kis.csv",
            "submission/query-3-qa.csv",
        }
        assert archive.read("submission/query-1-kis.csv") == b"L00_V000,1234\n"
        assert archive.read("submission/query-3-qa.csv").decode("utf-8") == (
            "L01_V028,3450,Năm người\n"
        )


def test_package_rejects_duplicate_query_names_and_unsafe_archive_name() -> None:
    file = KISSubmissionFile(
        mode="kis",
        query_filename="query-1-kis.txt",
        rows=(KISSubmissionRow(video_id="L00_V000", frame_id=1234),),
    )
    with pytest.raises(ValidationError):
        SubmissionPackageRequest(archive_name="../team", files=(file,))
    with pytest.raises(ValidationError):
        SubmissionPackageRequest(archive_name="TeamABC", files=(file, file))


def test_submission_package_endpoint_returns_a_downloadable_zip() -> None:
    response = TestClient(create_app()).post(
        "/submission/package",
        json={
            "archive_name": "TeamABCRound1",
            "files": [
                {
                    "mode": "kis",
                    "query_filename": "query-1-kis.txt",
                    "rows": [{"video_id": "L00_V000", "frame_id": 1234}],
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert response.headers["content-disposition"] == (
        'attachment; filename="TeamABCRound1.zip"'
    )
    with ZipFile(BytesIO(response.content)) as archive:
        assert archive.read("submission/query-1-kis.csv") == b"L00_V000,1234\n"
