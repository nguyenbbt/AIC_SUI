import os
import glob
import json
import logging
import asyncio
import math
from numbers import Real
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Coroutine, Dict, List, Optional, TypeVar

from .audio_extractor import AudioExtractor
from .caption_parser import CaptionParser
from .asr_engine import ASREngine
from .segment_grouper import SegmentGrouper
from .summarizer import VideoSummarizer
from .artifact_writer import (
    write_cleaned_transcript,
    write_raw_transcript,
    write_video_summary,
)
from .llm.cleaning_prompt import validate_cleaned_text

logger = logging.getLogger(__name__)
T = TypeVar("T")


def _run_coroutine_sync(coroutine: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine from synchronous code, including inside a running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    # A running event loop cannot be nested in the same thread. Use a short-lived
    # worker so synchronous callers embedded in async services remain supported.
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coroutine).result()


def _is_valid_summary_artifact(path: str, video_id: str) -> bool:
    """Return whether a cached summary is complete and belongs to the video."""
    try:
        with open(path, "r", encoding="utf-8") as summary_file:
            data = json.load(summary_file)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False

    return (
        isinstance(data, dict)
        and data.get("video_id") == video_id
        and isinstance(data.get("summary"), str)
        and bool(data["summary"].strip())
    )


def _load_valid_raw_artifact(
    path: str,
    video_id: str,
) -> Optional[tuple[List[Dict[str, Any]], str]]:
    """Load a complete raw cache or return ``None`` for regeneration."""
    try:
        with open(path, "r", encoding="utf-8") as raw_file:
            data = json.load(raw_file)
        if not isinstance(data, dict) or data.get("video_id") != video_id:
            return None
        source = data.get("source")
        segments = data.get("segments")
        if not isinstance(source, str) or not source.strip():
            return None
        if not isinstance(segments, list) or not segments:
            return None
        return SegmentGrouper.validate_segments(segments), source
    except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError):
        return None


def _load_valid_cleaned_artifact(
    path: str,
    video_id: str,
) -> Optional[List[Dict[str, Any]]]:
    """Load a safe cleaned cache or return ``None`` for regeneration."""
    try:
        with open(path, "r", encoding="utf-8") as cleaned_file:
            data = json.load(cleaned_file)
        if not isinstance(data, dict) or data.get("video_id") != video_id:
            return None
        intervals = data.get("intervals")
        if not isinstance(intervals, list) or not intervals:
            return None

        for interval in intervals:
            if not _is_valid_cached_interval(interval):
                return None
        return intervals
    except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError):
        return None


def _is_valid_cached_interval(interval: Any) -> bool:
    if not isinstance(interval, dict):
        return False
    start = interval.get("start_time_sec")
    end = interval.get("end_time_sec")
    if not _is_finite_number(start) or not _is_finite_number(end):
        return False
    if float(start) < 0.0 or float(end) <= float(start):
        return False
    if not isinstance(interval.get("interval_id"), str):
        return False
    if not isinstance(interval.get("segment_ids"), list):
        return False

    raw_text = interval.get("raw_text")
    cleaned_text = interval.get("cleaned_text")
    cleaning_failed = interval.get("cleaning_failed")
    if not isinstance(raw_text, str) or not raw_text.strip():
        return False
    if not isinstance(cleaning_failed, bool):
        return False
    if cleaning_failed:
        return cleaned_text == raw_text

    validate_cleaned_text(cleaned_text, raw_text)
    return True


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


class ASRTranscriptPipeline:
    def __init__(self, 
                 video_dir: str, 
                 metadata_dir: str, 
                 caption_dir: str, 
                 output_dir: str,
                 whisper_size: str = "medium",
                 llm_provider: str = "gemini",
                 llm_model: str = "gemini-2.5-flash",
                 group_size: Optional[int] = None,
                 min_interval_sec: float = 20.0,
                 target_interval_sec: float = 40.0,
                 max_interval_sec: float = 60.0,
                 device: str = "auto",
                 concurrency: int = 10,
                 summary_chunk_chars: int = 12_000,
                 force: bool = False):
        self.video_dir = video_dir
        self.metadata_dir = metadata_dir
        self.caption_dir = caption_dir
        self.output_dir = output_dir
        self.group_size = group_size
        self.min_interval_sec = min_interval_sec
        self.target_interval_sec = target_interval_sec
        self.max_interval_sec = max_interval_sec
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

        self.summarizer = VideoSummarizer(
            self.llm,
            max_chunk_chars=summary_chunk_chars,
        )

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
                interval["cleaned_text"] = validate_cleaned_text(
                    cleaned_text,
                    interval["raw_text"],
                )
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

    def process_video(self, video_id: str) -> None:
        logger.info(f"--- Processing video {video_id} ---")
        
        raw_transcript_path = os.path.join(self.transcripts_dir, f"{video_id}_raw.json")
        cleaned_transcript_path = os.path.join(self.transcripts_dir, f"{video_id}_cleaned.json")
        summary_path = os.path.join(self.summaries_dir, f"{video_id}.json")
        
        # 1. Get Raw Segments (Captions or ASR)
        segments = []
        source = ""
        raw_was_regenerated = False
        
        cached_raw = None
        if os.path.exists(raw_transcript_path) and not self.force:
            cached_raw = _load_valid_raw_artifact(raw_transcript_path, video_id)

        if cached_raw is not None:
            logger.info(f"Raw transcript exists for {video_id}, loading from file.")
            segments, source = cached_raw
        else:
            if os.path.exists(raw_transcript_path) and not self.force:
                logger.warning(
                    "Cached raw transcript for %s is invalid; regenerating it.",
                    video_id,
                )
            # Check for captions
            segments = CaptionParser.get_captions(video_id, self.caption_dir)
            if segments:
                source = "caption"
                logger.info(f"Found captions for {video_id}.")
            else:
                # Need ASR
                video_path = self._find_video_file(video_id)
                if not video_path:
                    raise FileNotFoundError(
                        f"Video file not found for {video_id}; ASR cannot run."
                    )
                    
                audio_path = os.path.join(self.audio_dir, f"{video_id}.wav")
                if not AudioExtractor.extract_audio(video_path, audio_path, force=self.force):
                    raise RuntimeError(
                        f"Failed to extract audio for {video_id}."
                    )
                
                self._init_asr_engine()
                segments = self.asr_engine.transcribe(audio_path)
                source = "asr"

            if not segments:
                raise RuntimeError(
                    f"No transcript segments generated for {video_id}."
                )

            write_raw_transcript(
                Path(raw_transcript_path),
                video_id=video_id,
                source=source,
                segments=segments,
            )
            raw_was_regenerated = True

        if not segments:
            raise RuntimeError(
                f"Cached raw transcript contains no segments for {video_id}."
            )

        # 2. Group and Clean
        cleaned_intervals = []
        cleaned_was_regenerated = False
        cached_cleaned = None
        if (
            os.path.exists(cleaned_transcript_path)
            and not self.force
            and not raw_was_regenerated
        ):
            cached_cleaned = _load_valid_cleaned_artifact(
                cleaned_transcript_path,
                video_id,
            )

        if cached_cleaned is not None:
            logger.info(f"Cleaned transcript exists for {video_id}, loading from file.")
            cleaned_intervals = cached_cleaned
        else:
            if os.path.exists(cleaned_transcript_path) and not self.force:
                logger.warning(
                    "Cached cleaned transcript for %s is invalid; regenerating it.",
                    video_id,
                )
            intervals = SegmentGrouper.group_segments(
                segments,
                self.group_size,
                min_duration_sec=self.min_interval_sec,
                target_duration_sec=self.target_interval_sec,
                max_duration_sec=self.max_interval_sec,
            )
            logger.info(f"Grouped into {len(intervals)} intervals. Starting LLM cleaning...")
            
            cleaned_intervals = _run_coroutine_sync(
                self._clean_video_intervals(intervals)
            )
            
            
            # Save cleaned transcript through the shared producer serializer.
            write_cleaned_transcript(
                Path(cleaned_transcript_path),
                video_id=video_id,
                source=source,
                llm_provider=self.llm.__class__.__name__,
                intervals=cleaned_intervals,
            )
            cleaned_was_regenerated = True

        # 3. Summarize
        if (
            not cleaned_was_regenerated
            and os.path.exists(summary_path)
            and not self.force
            and _is_valid_summary_artifact(summary_path, video_id)
        ):
            logger.info(f"Summary exists for {video_id}, skipping.")
        else:
            if os.path.exists(summary_path) and not self.force:
                if cleaned_was_regenerated:
                    logger.info(
                        "Transcript changed for %s; regenerating its summary.",
                        video_id,
                    )
                else:
                    logger.warning(
                        "Cached summary for %s is invalid; regenerating it.",
                        video_id,
                    )
            logger.info(f"Generating summary for {video_id}...")
            summary_text = self.summarizer.summarize_video(cleaned_intervals)
            write_video_summary(
                Path(summary_path),
                video_id=video_id,
                summary=summary_text,
                llm_provider=self.llm.__class__.__name__,
            )
            logger.info(f"Finished processing {video_id}.")

    def run(self) -> None:
        video_ids = self._get_video_ids()
        logger.info(f"Found {len(video_ids)} videos to process in metadata dir.")
        failures: Dict[str, str] = {}

        for vid in video_ids:
            try:
                self.process_video(vid)
            except Exception as exc:
                logger.exception("Pipeline crashed on video %s", vid)
                failures[vid] = f"{type(exc).__name__}: {exc}"
                
        if getattr(self.llm, "total_tokens_used", None) is not None:
            provider_name = self.llm.__class__.__name__.replace("TranscriptLLM", "")
            logger.info(f"Total {provider_name} tokens used in this run: {self.llm.total_tokens_used}")

        if failures:
            details = "; ".join(
                f"{video_id}: {reason}"
                for video_id, reason in failures.items()
            )
            raise RuntimeError(
                "ASR transcript pipeline failed for "
                f"{len(failures)}/{len(video_ids)} video(s): {details}"
            )
