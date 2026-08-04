from unittest.mock import MagicMock

from feature_extraction.asr_transcript.summarizer import VideoSummarizer


def test_summary_falls_back_to_raw_text_for_leaked_json_wrapper():
    llm = MagicMock()
    llm.summarize.return_value = "Bản tóm tắt bằng tiếng Việt."
    summarizer = VideoSummarizer(llm)
    raw_text = "Bão Ba Vì đang tiến gần Philippines."

    summarizer.summarize_video(
        [
            {
                "raw_text": raw_text,
                "cleaned_text": '{"cleaned_text":"Bão Ba Vì',
            }
        ]
    )

    llm.summarize.assert_called_once_with(raw_text)


def test_summary_falls_back_to_raw_text_for_expanded_cleaning_output():
    llm = MagicMock()
    llm.summarize.return_value = "Bản tóm tắt bằng tiếng Việt."
    summarizer = VideoSummarizer(llm)
    raw_text = "Thông tin chính về cơn bão."

    summarizer.summarize_video(
        [
            {
                "raw_text": raw_text,
                "cleaned_text": "tôi nghĩ " * 100,
            }
        ]
    )

    llm.summarize.assert_called_once_with(raw_text)


def test_summary_prefers_valid_cleaned_text():
    llm = MagicMock()
    llm.summarize.return_value = "Bản tóm tắt bằng tiếng Việt."
    summarizer = VideoSummarizer(llm)
    cleaned_text = "Cơn bão Ba Vì đang tiến gần Philippines."

    summarizer.summarize_video(
        [
            {
                "raw_text": "con bao dang den gan philippines",
                "cleaned_text": cleaned_text,
            }
        ]
    )

    llm.summarize.assert_called_once_with(cleaned_text)


def test_summary_prefers_short_cleaned_text_despite_high_ratio():
    llm = MagicMock()
    llm.summarize.return_value = "Bản tóm tắt bằng tiếng Việt."
    summarizer = VideoSummarizer(llm)

    summarizer.summarize_video(
        [{"raw_text": "raw", "cleaned_text": "cleaned"}]
    )

    llm.summarize.assert_called_once_with("cleaned")


def test_summary_skips_repetitive_chunk_when_informative_chunk_exists():
    informative_chunk = " ".join(
        f"sự kiện {index} có địa điểm và số liệu rõ ràng"
        for index in range(100)
    )
    repetitive_chunk = ("tôi nghĩ " * 500).strip()

    selected = VideoSummarizer._select_informative_chunks(
        [informative_chunk, repetitive_chunk]
    )

    assert selected == [informative_chunk]


def test_summary_keeps_short_meaningful_chunk():
    informative_chunk = " ".join(
        f"sự kiện {index} có địa điểm và số liệu rõ ràng"
        for index in range(100)
    )
    meaningful_short_chunk = (
        "Bão đổ bộ tại Philippines ngày 8/7 và gây mưa lớn."
    )

    selected = VideoSummarizer._select_informative_chunks(
        [informative_chunk, meaningful_short_chunk]
    )

    assert selected == [informative_chunk, meaningful_short_chunk]


def test_summary_keeps_input_when_all_chunks_lack_information():
    repetitive_chunks = [
        ("tôi nghĩ " * 200).strip(),
        ("không rõ " * 200).strip(),
    ]

    selected = VideoSummarizer._select_informative_chunks(
        repetitive_chunks
    )

    assert selected == repetitive_chunks
