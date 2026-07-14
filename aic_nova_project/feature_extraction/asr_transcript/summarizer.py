import logging
from typing import List, Dict, Any
from .llm.base import TranscriptLLM

logger = logging.getLogger(__name__)

class VideoSummarizer:
    """
    Summarizes an entire video based on its cleaned transcript intervals.
    """
    def __init__(self, llm: TranscriptLLM):
        self.llm = llm

    def summarize_video(self, intervals: List[Dict[str, Any]]) -> str:
        """
        Concatenates all cleaned texts and passes them to the LLM for a final summary.
        
        Args:
            intervals: The list of cleaned intervals.
            
        Returns:
            A string containing the summary.
        """
        if not intervals:
            return ""
            
        # Extract and concatenate all cleaned texts
        # Fall back to raw_text if cleaning failed for some reason
        texts = [
            interval.get("cleaned_text", interval.get("raw_text", ""))
            for interval in intervals
        ]
        
        full_text = " ".join([t for t in texts if t.strip()])
        
        if not full_text:
            return ""
            
        logger.info("Sending full transcript to LLM for summarization...")
        try:
            summary = self.llm.summarize(full_text)
            logger.info("Summarization successful.")
            return summary
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return ""
