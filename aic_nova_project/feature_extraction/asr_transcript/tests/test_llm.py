import pytest
from unittest.mock import patch, MagicMock
from feature_extraction.asr_transcript.llm.base import TranscriptLLM

class DummyTranscriptLLM(TranscriptLLM):
    """A dummy implementation of TranscriptLLM for testing the interface."""
    def clean(self, raw_text: str, context: str = "") -> str:
        return f"CLEANED: {raw_text}"
        
    def summarize(self, full_cleaned_text: str) -> str:
        return "SUMMARY"

def test_llm_interface():
    # Test that the abstract base class forces implementation
    llm = DummyTranscriptLLM()
    
    # Test clean
    res = llm.clean("noisy text")
    assert res == "CLEANED: noisy text"
    
    # Test summarize
    res = llm.summarize("all clean text")
    assert res == "SUMMARY"

@patch("google.genai.Client")
def test_gemini_llm_mocked(mock_client_class, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test_key")
    
    from feature_extraction.asr_transcript.llm.gemini_llm import GeminiTranscriptLLM
    
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    
    mock_response = MagicMock()
    # Mocking parsed output
    mock_parsed = MagicMock()
    mock_parsed.cleaned_text = "clean result"
    mock_response.parsed = mock_parsed
    mock_response.usage_metadata.total_token_count = 10
    
    mock_client.models.generate_content.return_value = mock_response
    
    llm = GeminiTranscriptLLM()
    result = llm.clean("raw")
    
    assert result == "clean result"
    assert llm.total_tokens_used == 10
    mock_client.models.generate_content.assert_called_once()
