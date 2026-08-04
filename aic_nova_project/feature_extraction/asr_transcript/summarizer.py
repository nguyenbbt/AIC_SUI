import logging
import re
from typing import List, Dict, Any
from .llm.base import TranscriptLLM
from .llm.summary_prompt import source_is_repetitive_noise

logger = logging.getLogger(__name__)

_LEAKED_CLEANED_JSON = re.compile(
    r'^\s*(?:```json\s*)?\{\s*"cleaned_text"\s*:',
    flags=re.IGNORECASE,
)
_MAX_CLEANED_EXPANSION_RATIO = 2.0
_MIN_CLEANED_EXPANSION_CHARS = 256

class VideoSummarizer:
    """
    Summarizes an entire video based on its cleaned transcript intervals.
    """
    def __init__(
        self,
        llm: TranscriptLLM,
        max_chunk_chars: int = 12_000,
        max_reduction_rounds: int = 8,
    ):
        if max_chunk_chars <= 0:
            raise ValueError("max_chunk_chars must be greater than zero.")
        if max_reduction_rounds <= 0:
            raise ValueError("max_reduction_rounds must be greater than zero.")

        self.llm = llm
        self.max_chunk_chars = max_chunk_chars
        self.max_reduction_rounds = max_reduction_rounds

    def _split_text(self, text: str) -> List[str]:
        """Split text into non-empty chunks that respect the request budget."""
        chunks: List[str] = []
        current = ""

        for word in text.split():
            while len(word) > self.max_chunk_chars:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(word[:self.max_chunk_chars])
                word = word[self.max_chunk_chars:]

            if not word:
                continue

            candidate = f"{current} {word}".strip()
            if len(candidate) <= self.max_chunk_chars:
                current = candidate
            else:
                chunks.append(current)
                current = word

        if current:
            chunks.append(current)

        return chunks

    def _summarize_text(self, text: str) -> str:
        """Summarize one bounded chunk and reject unusable provider output."""
        summary = self.llm.summarize(text)
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("LLM returned an empty summary.")
        return summary

    @staticmethod
    def _select_summary_text(interval: Dict[str, Any]) -> str:
        """Prefer cleaned text unless it violates basic quality bounds."""
        raw_text = str(interval.get("raw_text", "")).strip()
        cleaned_text = str(interval.get("cleaned_text", "")).strip()
        if not cleaned_text:
            return raw_text
        if not raw_text:
            return cleaned_text

        leaked_json = bool(_LEAKED_CLEANED_JSON.match(cleaned_text))
        expanded = (
            len(cleaned_text)
            > len(raw_text) * _MAX_CLEANED_EXPANSION_RATIO
            and len(cleaned_text) - len(raw_text)
            > _MIN_CLEANED_EXPANSION_CHARS
        )
        if leaked_json or expanded:
            logger.warning(
                "Ignoring unusable cleaned transcript interval "
                "(raw_chars=%d, cleaned_chars=%d, leaked_json=%s).",
                len(raw_text),
                len(cleaned_text),
                leaked_json,
            )
            return raw_text
        return cleaned_text

    @staticmethod
    def _select_informative_chunks(chunks: List[str]) -> List[str]:
        """Drop repetitive chunks only when informative chunks remain."""
        informative_chunks = [
            chunk
            for chunk in chunks
            if not source_is_repetitive_noise(chunk)
        ]
        if not informative_chunks:
            return chunks
        skipped_count = len(chunks) - len(informative_chunks)
        if skipped_count:
            logger.warning(
                "Skipping low-information transcript chunks "
                "(skipped=%d, total=%d).",
                skipped_count,
                len(chunks),
            )
        return informative_chunks

    def summarize_video(self, intervals: List[Dict[str, Any]]) -> str:
        """
        Concatenates all cleaned texts and passes them to the LLM for a final summary.
        
        Args:
            intervals: The list of cleaned intervals.
            
        Returns:
            A string containing the summary.
        """
        if not intervals:
            raise ValueError("Cannot summarize an empty interval list.")
            
        # Extract and concatenate all cleaned texts
        # Fall back to raw_text if cleaning failed for some reason
        texts = [self._select_summary_text(interval) for interval in intervals]
        
        full_text = " ".join([t for t in texts if t.strip()])
        
        if not full_text:
            raise ValueError("Cannot summarize an empty transcript.")
            
        logger.info("Sending transcript to LLM for summarization...")
        try:
            chunks = self._select_informative_chunks(
                self._split_text(full_text)
            )
            if len(chunks) == 1:
                summary = self._summarize_text(chunks[0])
            else:
                summaries = [
                    self._summarize_text(chunk)
                    for chunk in chunks
                ]
                for _ in range(self.max_reduction_rounds):
                    reduction_input = "\n".join(summaries)
                    reduction_chunks = self._split_text(reduction_input)
                    if len(reduction_chunks) == 1:
                        summary = self._summarize_text(reduction_chunks[0])
                        break
                    summaries = [
                        self._summarize_text(chunk)
                        for chunk in reduction_chunks
                    ]
                else:
                    raise RuntimeError(
                        "Summary reduction did not converge within "
                        f"{self.max_reduction_rounds} rounds."
                    )
        except Exception:
            logger.exception("Summarization failed.")
            raise

        logger.info("Summarization successful.")
        return summary
