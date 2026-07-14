import argparse
import logging
from .pipeline import ASRTranscriptPipeline

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="ASR and Transcript Cleaning/Summarization Pipeline (Module 3)")
    
    parser.add_argument("--video-dir", type=str, required=True, help="Directory containing raw videos.")
    parser.add_argument("--metadata-dir", type=str, required=True, help="Directory containing metadata JSONs from Module 1.")
    parser.add_argument("--caption-dir", type=str, required=True, help="Directory containing existing .srt or .vtt captions.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save output audio, transcripts, and summaries.")
    
    parser.add_argument("--whisper-size", type=str, default="medium", choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"], help="PhoWhisper model size.")
    parser.add_argument("--llm-provider", type=str, default="gemini", choices=["gemini", "local", "azure"], help="LLM provider for cleaning/summarization.")
    parser.add_argument("--llm-model", type=str, default="gemini-2.5-flash", help="Model name for the LLM.")
    
    parser.add_argument("--group-size", type=int, default=5, help="Number of ASR segments to group into one cleaning interval.")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent LLM API calls for cleaning.")
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
        device=args.device,
        concurrency=args.concurrency,
        force=args.force
    )
    
    pipeline.run()

if __name__ == "__main__":
    main()
