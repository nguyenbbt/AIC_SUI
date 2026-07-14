import os
import glob
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional

from .audio_extractor import AudioExtractor
from .caption_parser import CaptionParser
from .asr_engine import ASREngine
from .segment_grouper import SegmentGrouper
from .summarizer import VideoSummarizer

logger = logging.getLogger(__name__)

class ASRTranscriptPipeline:
    def __init__(self, 
                 video_dir: str, 
                 metadata_dir: str, 
                 caption_dir: str, 
                 output_dir: str,
                 whisper_size: str = "medium",
                 llm_provider: str = "gemini",
                 llm_model: str = "gemini-2.5-flash",
                 group_size: int = 5,
                 device: str = "auto",
                 concurrency: int = 10,
                 force: bool = False):
        self.video_dir = video_dir
        self.metadata_dir = metadata_dir
        self.caption_dir = caption_dir
        self.output_dir = output_dir
        self.group_size = group_size
        self.force = force
        self.concurrency = concurrency

        self.audio_dir = os.path.join(output_dir, "audio")
        self.transcripts_dir = os.path.join(output_dir, "transcripts")
        self.summaries_dir = os.path.join(output_dir, "summaries")

        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.transcripts_dir, exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)

        self.asr_engine = None
        self.whisper_size = whisper_size
        self.device = device

        if llm_provider == "gemini":
            from .llm.gemini_llm import GeminiTranscriptLLM
            self.llm = GeminiTranscriptLLM(model_name=llm_model)
        elif llm_provider == "azure":
            from .llm.azure_llm import AzureTranscriptLLM
            self.llm = AzureTranscriptLLM(model_name=llm_model)
        else:
            from .llm.local_llm import LocalTranscriptLLM
            self.llm = LocalTranscriptLLM(model_name=llm_model, device=device)

        self.summarizer = VideoSummarizer(self.llm)

    def _init_asr_engine(self):
        """Lazy initialization of ASR Engine to save VRAM if not needed."""
        if self.asr_engine is None:
            self.asr_engine = ASREngine(model_size=self.whisper_size, device=self.device)

    def _get_video_ids(self) -> List[str]:
        video_ids = []
        for metadata_file in glob.glob(os.path.join(self.metadata_dir, "*.json")):
            video_id = os.path.basename(metadata_file).replace(".json", "")
            video_ids.append(video_id)
        return sorted(video_ids)

    def _find_video_file(self, video_id: str) -> Optional[str]:
        # Tries to find the video file with any common extension
        for ext in ['.mp4', '.mkv', '.avi', '.webm']:
            path = os.path.join(self.video_dir, f"{video_id}{ext}")
            if os.path.exists(path):
                return path
        return None

    async def _clean_interval_async(self, interval: Dict[str, Any], context: str, sem: asyncio.Semaphore) -> Dict[str, Any]:
        """Async wrapper for the LLM clean function to allow parallel processing."""
        async with sem:
            # We must run the synchronous LLM call in a threadpool to not block the event loop
            try:
                loop = asyncio.get_event_loop()
                cleaned_text = await loop.run_in_executor(None, self.llm.clean, interval["raw_text"], context)
                interval["cleaned_text"] = cleaned_text
                interval["cleaning_failed"] = False
            except Exception as e:
                logger.error(f"Cleaning failed for interval {interval.get('interval_id')}: {e}")
                interval["cleaned_text"] = interval["raw_text"]
                interval["cleaning_failed"] = True
            return interval

    async def _clean_video_intervals(self, intervals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Processes all intervals of a video concurrently, respecting concurrency limit."""
        sem = asyncio.Semaphore(self.concurrency)
        tasks = []
        
        # Determine context for each interval (using the raw text of the previous interval)
        for i, interval in enumerate(intervals):
            context = ""
            if i > 0:
                context = intervals[i-1]["raw_text"]
            task = self._clean_interval_async(interval, context, sem)
            tasks.append(task)
            
        cleaned_intervals = await asyncio.gather(*tasks)
        return cleaned_intervals

    def process_video(self, video_id: str):
        logger.info(f"--- Processing video {video_id} ---")
        
        raw_transcript_path = os.path.join(self.transcripts_dir, f"{video_id}_raw.json")
        cleaned_transcript_path = os.path.join(self.transcripts_dir, f"{video_id}_cleaned.json")
        summary_path = os.path.join(self.summaries_dir, f"{video_id}.json")
        
        # 1. Get Raw Segments (Captions or ASR)
        segments = []
        source = ""
        
        if os.path.exists(raw_transcript_path) and not self.force:
            logger.info(f"Raw transcript exists for {video_id}, loading from file.")
            with open(raw_transcript_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                segments = data.get("segments", [])
                source = data.get("source", "unknown")
        else:
            # Check for captions
            segments = CaptionParser.get_captions(video_id, self.caption_dir)
            if segments:
                source = "caption"
                logger.info(f"Found captions for {video_id}.")
            else:
                # Need ASR
                video_path = self._find_video_file(video_id)
                if not video_path:
                    logger.error(f"Video file not found for {video_id}, skipping ASR.")
                    return
                    
                audio_path = os.path.join(self.audio_dir, f"{video_id}.wav")
                if not AudioExtractor.extract_audio(video_path, audio_path, force=self.force):
                    logger.error(f"Failed to extract audio for {video_id}, skipping.")
                    return
                
                self._init_asr_engine()
                segments = self.asr_engine.transcribe(audio_path)
                source = "asr"
                
            # Save raw segments
            with open(raw_transcript_path, 'w', encoding='utf-8') as f:
                json.dump({"video_id": video_id, "source": source, "segments": segments}, f, ensure_ascii=False, indent=2)

        if not segments:
            logger.warning(f"No segments generated for {video_id}, skipping cleaning.")
            return

        # 2. Group and Clean
        cleaned_intervals = []
        if os.path.exists(cleaned_transcript_path) and not self.force:
            logger.info(f"Cleaned transcript exists for {video_id}, loading from file.")
            with open(cleaned_transcript_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                cleaned_intervals = data.get("intervals", [])
        else:
            intervals = SegmentGrouper.group_segments(segments, self.group_size)
            logger.info(f"Grouped into {len(intervals)} intervals. Starting LLM cleaning...")
            
            # Run async cleaning
            try:
                loop = asyncio.get_running_loop()
                # If there's an existing loop, use it
                cleaned_intervals = loop.run_until_complete(self._clean_video_intervals(intervals))
            except RuntimeError:
                # If no loop exists, use asyncio.run
                cleaned_intervals = asyncio.run(self._clean_video_intervals(intervals))
            
            
            # Save cleaned transcript
            with open(cleaned_transcript_path, 'w', encoding='utf-8') as f:
                output_data = {
                    "video_id": video_id,
                    "source": source,
                    "llm_provider": self.llm.__class__.__name__,
                    "intervals": cleaned_intervals
                }
                json.dump(output_data, f, ensure_ascii=False, indent=2)

        # 3. Summarize
        if os.path.exists(summary_path) and not self.force:
            logger.info(f"Summary exists for {video_id}, skipping.")
        else:
            logger.info(f"Generating summary for {video_id}...")
            summary_text = self.summarizer.summarize_video(cleaned_intervals)
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "video_id": video_id,
                    "summary": summary_text,
                    "llm_provider": self.llm.__class__.__name__
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"Finished processing {video_id}.")

    def run(self):
        video_ids = self._get_video_ids()
        logger.info(f"Found {len(video_ids)} videos to process in metadata dir.")
        
        for vid in video_ids:
            try:
                self.process_video(vid)
            except Exception as e:
                logger.error(f"Pipeline crashed on video {vid}: {e}")
                
        if getattr(self.llm, "total_tokens_used", None) is not None:
            provider_name = self.llm.__class__.__name__.replace("TranscriptLLM", "")
            logger.info(f"Total {provider_name} tokens used in this run: {self.llm.total_tokens_used}")
