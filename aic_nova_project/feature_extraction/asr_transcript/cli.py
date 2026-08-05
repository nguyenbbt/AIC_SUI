import argparse
import logging
from .pipeline import ASRTranscriptPipeline

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="ASR and Transcript Cleaning/Summarization Pipeline (Module 3)")
    
    parser.add_argument("--video-dir", type=str, required=True, help="Directory containing raw videos.")
    parser.add_argument("--metadata-dir", type=str, required=True, help="Directory containing metadata JSONs from Module 1.")
    parser.add_argument("--caption-dir", type=str, required=True, help="Directory containing existing .srt or .vtt captions.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save output audio, transcripts, and summaries.")
    
    parser.add_argument(
        "--whisper-size",
        type=str,
        default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Published PhoWhisper model size.",
    )
    parser.add_argument("--llm-provider", type=str, default="gemini", choices=["gemini", "local", "azure"], help="LLM provider for cleaning/summarization.")
    parser.add_argument("--llm-model", type=str, default="gemini-2.5-flash", help="Model name for the LLM.")
    
    parser.add_argument(
        "--group-size",
        type=int,
        default=None,
        help="Legacy fixed segment count. Omit to group by real timestamps.",
    )
    parser.add_argument("--min-interval-sec", type=float, default=20.0)
    parser.add_argument("--target-interval-sec", type=float, default=40.0)
    parser.add_argument("--max-interval-sec", type=float, default=60.0)
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent LLM API calls for cleaning.")
    parser.add_argument(
        "--summary-chunk-chars",
        type=int,
        default=12_000,
        help="Maximum transcript characters sent in one summary request.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device for local models.")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing outputs.")
    
    args = parser.parse_args()
    
    pipeline = ASRTranscriptPipeline(
        video_dir=args.video_dir,
        metadata_dir=args.metadata_dir,
        caption_dir=args.caption_dir,
        output_dir=args.output_dir,
        whisper_size=args.whisper_size,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        group_size=args.group_size,
        min_interval_sec=args.min_interval_sec,
        target_interval_sec=args.target_interval_sec,
        max_interval_sec=args.max_interval_sec,
        device=args.device,
        concurrency=args.concurrency,
        summary_chunk_chars=args.summary_chunk_chars,
        force=args.force
    )
    
    try:
        pipeline.run()
    except Exception:
        logger.exception("ASR transcript pipeline failed.")
        return 1

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
