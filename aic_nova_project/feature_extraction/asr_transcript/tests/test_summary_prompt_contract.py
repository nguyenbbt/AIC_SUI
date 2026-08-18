import sys
from types import SimpleNamespace
from types import ModuleType
from unittest.mock import MagicMock

import pytest


openai_module = ModuleType("openai")
openai_module.AzureOpenAI = MagicMock
sys.modules.setdefault("openai", openai_module)

from feature_extraction.asr_transcript.llm.azure_llm import (
    AzureTranscriptLLM,
)
from feature_extraction.asr_transcript.llm.gemini_llm import (
    GeminiTranscriptLLM,
)
from feature_extraction.asr_transcript.llm.local_llm import (
    LocalTranscriptLLM,
)
from feature_extraction.asr_transcript.llm.summary_prompt import (
    validate_summary_contract,
    validate_vietnamese_summary,
)


TRANSCRIPT = (
    "Prime Minister yêu cầu xử lý sự cố tại Hạ Long ngày 19/07, "
    "liên quan tàu Wonder Sea và 25 hành khách."
)
LONG_TRANSCRIPT = " ".join(
    f"dữ liệu sự kiện số {index} rõ ràng"
    for index in range(100)
)
REPETITIVE_TRANSCRIPT = ("tôi nghĩ " * 500).strip()


def _vietnamese_words(count: int) -> str:
    words = ("nội dung này là rõ ràng và chính xác ".split() * 30)
    return " ".join(words[:count])


def test_summary_language_guard_rejects_english_output():
    with pytest.raises(ValueError, match="written in Vietnamese"):
        validate_vietnamese_summary(
            "Prime Minister issues a directive after the boat accident."
        )


def test_summary_language_guard_accepts_vietnamese_output():
    summary = (
        "Thủ tướng đã ban hành chỉ đạo xử lý vụ tai nạn và yêu cầu "
        "các cơ quan điều tra nguyên nhân."
    )

    assert validate_vietnamese_summary(summary) == summary


def test_summary_contract_requires_100_words_for_substantial_source():
    with pytest.raises(ValueError, match="at least 100 words"):
        validate_summary_contract(_vietnamese_words(61), LONG_TRANSCRIPT)


def test_summary_contract_allows_short_output_for_short_source():
    summary = _vietnamese_words(61)

    assert validate_summary_contract(summary, TRANSCRIPT) == summary


def test_summary_contract_allows_short_output_for_repetitive_source():
    summary = _vietnamese_words(40)

    assert (
        validate_summary_contract(summary, REPETITIVE_TRANSCRIPT)
        == summary
    )


def test_summary_contract_trims_more_than_180_words_without_adding_facts():
    first_sentence = _vietnamese_words(120) + "."
    second_sentence = _vietnamese_words(80) + "."

    summary = validate_summary_contract(
        f"{first_sentence} {second_sentence}",
        LONG_TRANSCRIPT,
    )

    assert summary == first_sentence
    assert len(summary.split()) == 120


def test_summary_contract_rejects_list_output():
    summary = "- " + _vietnamese_words(100)

    with pytest.raises(ValueError, match="one paragraph"):
        validate_summary_contract(summary, LONG_TRANSCRIPT)


def _assert_vietnamese_summary_contract(prompt: str) -> None:
    normalized = prompt.casefold()

    assert "luôn bằng tiếng việt" in normalized
    assert "một đoạn văn" in normalized
    assert "100-180 từ" in normalized
    assert "không dùng danh sách" in normalized
    assert "giữ nguyên chính xác" in normalized
    assert "tên riêng" in normalized
    assert "địa danh" in normalized
    assert "ký hiệu" in normalized
    assert "số liệu" in normalized
    assert "không suy đoán" in normalized
    assert "không ghép số liệu" in normalized
    assert "mốc thời gian" in normalized
    assert "ngắn hơn 100 từ" in normalized
    assert "không thêm kiến thức bên ngoài" in normalized
    assert f"<transcript>\n{TRANSCRIPT}\n</transcript>" in prompt


def test_azure_summary_prompt_enforces_vietnamese_contract():
    llm = AzureTranscriptLLM.__new__(AzureTranscriptLLM)
    llm.deployment_name = "gpt-4o"
    llm.total_tokens_used = 0
    llm.client = MagicMock()
    llm.client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"summary":"Bản tóm tắt tiếng Việt."}'
                )
            )
        ],
        usage=None,
    )

    assert llm.summarize(TRANSCRIPT) == "Bản tóm tắt tiếng Việt."

    messages = llm.client.chat.completions.create.call_args.kwargs[
        "messages"
    ]
    _assert_vietnamese_summary_contract(
        "\n".join(message["content"] for message in messages)
    )


def test_azure_clean_keeps_cleaning_system_contract():
    llm = AzureTranscriptLLM.__new__(AzureTranscriptLLM)
    llm.deployment_name = "gpt-4o"
    llm.total_tokens_used = 0
    llm.client = MagicMock()
    llm.client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"cleaned_text":"Nội dung đã làm sạch."}'
                )
            )
        ],
        usage=None,
    )

    assert llm.clean("noi dung") == "Nội dung đã làm sạch."

    system_prompt = (
        llm.client.chat.completions.create.call_args.kwargs["messages"][0][
            "content"
        ]
    )
    assert "cleaned_text" in system_prompt
    assert "khóa summary" not in system_prompt


def test_gemini_summary_prompt_enforces_vietnamese_contract():
    llm = GeminiTranscriptLLM.__new__(GeminiTranscriptLLM)
    llm.model_name = "gemini-2.5-flash"
    llm.total_tokens_used = 0
    llm.client = MagicMock()
    llm.client.models.generate_content.return_value = SimpleNamespace(
        parsed=SimpleNamespace(summary="Bản tóm tắt tiếng Việt."),
        usage_metadata=None,
    )

    assert llm.summarize(TRANSCRIPT) == "Bản tóm tắt tiếng Việt."

    prompt = llm.client.models.generate_content.call_args.kwargs[
        "contents"
    ]
    _assert_vietnamese_summary_contract(prompt)


def test_local_summary_prompt_enforces_vietnamese_contract():
    llm = LocalTranscriptLLM.__new__(LocalTranscriptLLM)
    llm.generator = MagicMock(
        return_value=[
            {
                "generated_text": [
                    {
                        "content": (
                            '{"summary":"Bản tóm tắt tiếng Việt."}'
                        )
                    }
                ]
            }
        ]
    )

    assert llm.summarize(TRANSCRIPT) == "Bản tóm tắt tiếng Việt."

    messages = llm.generator.call_args.args[0]
    assert llm.generator.call_args.kwargs["do_sample"] is False
    _assert_vietnamese_summary_contract(
        "\n".join(message["content"] for message in messages)
    )


def test_local_summary_rewrites_an_english_response_in_vietnamese():
    llm = LocalTranscriptLLM.__new__(LocalTranscriptLLM)
    llm.generator = MagicMock(
        side_effect=[
            [
                {
                    "generated_text": [
                        {
                            "content": (
                                '{"summary":"Prime Minister issues a '
                                'directive after the boat accident."}'
                            )
                        }
                    ]
                }
            ],
            [
                {
                    "generated_text": [
                        {
                            "content": (
                                '{"summary":"Thủ tướng đã ban hành chỉ '
                                'đạo sau vụ tai nạn tàu và yêu cầu các cơ '
                                'quan điều tra nguyên nhân."}'
                            )
                        }
                    ]
                }
            ],
        ]
    )

    summary = llm.summarize(TRANSCRIPT)

    assert summary.startswith("Thủ tướng đã ban hành chỉ đạo")
    assert llm.generator.call_count == 2

    repair_messages = llm.generator.call_args_list[1].args[0]
    repair_prompt = "\n".join(
        message["content"] for message in repair_messages
    )
    assert TRANSCRIPT not in repair_prompt
    assert "Prime Minister issues a directive" in repair_prompt
    assert "viết lại" in repair_prompt.casefold()
    assert "tiếng việt" in repair_prompt.casefold()
    assert llm.generator.call_args_list[1].kwargs["do_sample"] is False


def test_local_summary_logs_bounded_preview_when_rewrite_fails(caplog):
    english_summary = (
        "Prime Minister issues a directive after the boat accident. "
        * 10
    )
    llm = LocalTranscriptLLM.__new__(LocalTranscriptLLM)
    llm.generator = MagicMock(
        return_value=[
            {
                "generated_text": [
                    {
                        "content": (
                            '{"summary":"' + english_summary + '"}'
                        )
                    }
                ]
            }
        ]
    )

    with pytest.raises(ValueError, match="written in Vietnamese"):
        llm.summarize(TRANSCRIPT)

    failure_logs = [
        record
        for record in caplog.records
        if "summary_contract_validation_failed" in record.message
    ]
    assert len(failure_logs) == 3
    assert failure_logs[-1].attempt == 3
    assert len(failure_logs[-1].summary_preview) <= 240
    assert failure_logs[-1].summary_chars == len(english_summary.strip())


def test_local_summary_repairs_too_short_output_from_substantial_source():
    short_summary = _vietnamese_words(61)
    valid_summary = _vietnamese_words(110)
    llm = LocalTranscriptLLM.__new__(LocalTranscriptLLM)
    llm.generator = MagicMock(
        side_effect=[
            [
                {
                    "generated_text": [
                        {
                            "content": (
                                '{"summary":"' + short_summary + '"}'
                            )
                        }
                    ]
                }
            ],
            [
                {
                    "generated_text": [
                        {
                            "content": (
                                '{"summary":"' + valid_summary + '"}'
                            )
                        }
                    ]
                }
            ],
        ]
    )

    assert llm.summarize(LONG_TRANSCRIPT) == valid_summary

    repair_prompt = "\n".join(
        message["content"]
        for message in llm.generator.call_args_list[1].args[0]
    )
    assert LONG_TRANSCRIPT in repair_prompt
    assert short_summary in repair_prompt
    assert "100-180 từ" in repair_prompt


def test_local_summary_allows_three_contract_attempts():
    first_short_summary = _vietnamese_words(61)
    second_short_summary = _vietnamese_words(75)
    valid_summary = _vietnamese_words(110)
    llm = LocalTranscriptLLM.__new__(LocalTranscriptLLM)
    llm.generator = MagicMock(
        side_effect=[
            [
                {
                    "generated_text": [
                        {
                            "content": (
                                '{"summary":"'
                                + first_short_summary
                                + '"}'
                            )
                        }
                    ]
                }
            ],
            [
                {
                    "generated_text": [
                        {
                            "content": (
                                '{"summary":"'
                                + second_short_summary
                                + '"}'
                            )
                        }
                    ]
                }
            ],
            [
                {
                    "generated_text": [
                        {
                            "content": (
                                '{"summary":"' + valid_summary + '"}'
                            )
                        }
                    ]
                }
            ],
        ]
    )

    assert llm.summarize(LONG_TRANSCRIPT) == valid_summary
    assert llm.generator.call_count == 3


def test_local_summary_recovers_when_contract_rewrite_is_not_json():
    short_summary = _vietnamese_words(61)
    valid_summary = _vietnamese_words(110)
    malformed_rewrite = "Đây là bản viết lại nhưng không được bọc trong JSON."
    llm = LocalTranscriptLLM.__new__(LocalTranscriptLLM)
    llm.generator = MagicMock(
        side_effect=[
            [
                {
                    "generated_text": [
                        {
                            "content": (
                                '{"summary":"' + short_summary + '"}'
                            )
                        }
                    ]
                }
            ],
            [
                {
                    "generated_text": [
                        {"content": malformed_rewrite}
                    ]
                }
            ],
            [
                {
                    "generated_text": [
                        {
                            "content": (
                                '{"summary":"' + valid_summary + '"}'
                            )
                        }
                    ]
                }
            ],
        ]
    )

    assert llm.summarize(LONG_TRANSCRIPT) == valid_summary
    assert llm.generator.call_count == 3

    format_repair_prompt = "\n".join(
        message["content"]
        for message in llm.generator.call_args_list[2].args[0]
    )
    assert malformed_rewrite in format_repair_prompt
    assert LONG_TRANSCRIPT in format_repair_prompt
    assert "JSON" in format_repair_prompt
