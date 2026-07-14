from abc import ABC, abstractmethod

class TranscriptLLM(ABC):
    """
    Abstract base class for LLM implementations used for transcript cleaning and summarization.
    """
    
    @abstractmethod
    def clean(self, raw_text: str, context: str = "") -> str:
        """
        Cleans the raw ASR text (fixes typos, missing punctuation, unifies spelling).
        
        Args:
            raw_text: The noisy text from ASR.
            context: Optional previous sentences to provide context for ambiguous words.
            
        Returns:
            The cleaned text string.
        """
        pass
        
    @abstractmethod
    def summarize(self, full_cleaned_text: str) -> str:
        """
        Summarizes the entire video based on the full cleaned transcript.
        
        Args:
            full_cleaned_text: The concatenated cleaned text of all intervals.
            
        Returns:
            The video summary.
        """
        pass
