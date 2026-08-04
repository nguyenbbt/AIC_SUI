import logging
from typing import List, Dict, Any
from .llm.base import TranscriptLLM

logger = logging.getLogger(__name__)

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
        texts = [
            interval.get("cleaned_text", interval.get("raw_text", ""))
            for interval in intervals
        ]
        
        full_text = " ".join([t for t in texts if t.strip()])
        
        if not full_text:
            raise ValueError("Cannot summarize an empty transcript.")
            
        logger.info("Sending transcript to LLM for summarization...")
        try:
            chunks = self._split_text(full_text)
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
