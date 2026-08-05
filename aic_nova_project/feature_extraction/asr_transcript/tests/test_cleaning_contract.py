import asyncio
from types import SimpleNamespace

import pytest

from feature_extraction.asr_transcript.llm.cleaning_prompt import (
    CleaningContractError,
    build_cleaning_prompt,
    validate_cleaned_text,
)
from feature_extraction.asr_transcript.pipeline import ASRTranscriptPipeline


def test_cleaning_prompt_requires_vietnamese_edit_only_output() -> None:
    prompt = build_cleaning_prompt(
        "hom nay bao so 3 vao Da Nang",
        "ban tin thoi tiet",
    )

    assert "tiếng Việt" in prompt
    assert "không dịch" in prompt.casefold()
    assert "không bổ sung" in prompt
    assert "Da Nang" in prompt


@pytest.mark.parametrize(
    ("cleaned_text", "error_code"),
    [
        ('{"cleaned_text": "Nội dung"}', "format"),
        ("Nội dung mới " * 30, "expansion"),
        ("The report 3 is about a major storm and the local response.", "language"),
        ("Bão số 4 đi vào Đà Nẵng.", "numbers"),
    ],
)
def test_cleaning_validator_rejects_unsafe_output(
    cleaned_text: str,
    error_code: str,
) -> None:
    raw_text = "hom nay bao so 3 vao Da Nang"

    with pytest.raises(CleaningContractError) as error:
        validate_cleaned_text(cleaned_text, raw_text)

    assert error.value.code == error_code


def test_cleaning_validator_accepts_vietnamese_correction() -> None:
    assert validate_cleaned_text(
        "Hôm nay, bão số 3 vào Da Nang.",
        "hom nay bao so 3 vao Da Nang",
    ) == "Hôm nay, bão số 3 vào Da Nang."


def test_cleaning_validator_rejects_inserted_vietnamese_number_words() -> None:
    with pytest.raises(CleaningContractError) as error:
        validate_cleaned_text(
            "Gió mạnh cấp bảy.",
            "gio manh cap",
        )

    assert error.value.code == "numbers"


def test_pipeline_falls_back_when_cleaning_contract_is_invalid() -> None:
    pipeline = ASRTranscriptPipeline.__new__(ASRTranscriptPipeline)
    pipeline.llm = SimpleNamespace(
        clean=lambda raw_text, context: "The report invents unrelated content."
    )
    interval = {
        "interval_id": "0",
        "start_time_sec": 0.0,
        "end_time_sec": 25.0,
        "raw_text": "hom nay bao so 3 vao Da Nang",
        "segment_ids": [0],
    }

    cleaned = asyncio.run(
        pipeline._clean_interval_async(interval, "", asyncio.Semaphore(1))
    )

    assert cleaned["cleaned_text"] == interval["raw_text"]
    assert cleaned["cleaning_failed"] is True
