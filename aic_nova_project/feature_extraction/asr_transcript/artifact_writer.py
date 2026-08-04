import json
from pathlib import Path
from typing import Any, Dict, Sequence


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    """Publish one JSON artifact only after serialization succeeds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as output_file:
            json.dump(
                payload,
                output_file,
                ensure_ascii=False,
                indent=2,
            )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_cleaned_transcript(
    path: Path,
    *,
    video_id: str,
    source: str,
    llm_provider: str,
    intervals: Sequence[Dict[str, Any]],
) -> None:
    """Write the canonical Module 3 cleaned-transcript envelope."""
    _write_json_atomic(
        path,
        {
            "video_id": video_id,
            "source": source,
            "llm_provider": llm_provider,
            "intervals": list(intervals),
        },
    )


def write_video_summary(
    path: Path,
    *,
    video_id: str,
    summary: str,
    llm_provider: str,
) -> None:
    """Write the canonical Module 3 video-summary envelope."""
    _write_json_atomic(
        path,
        {
            "video_id": video_id,
            "summary": summary,
            "llm_provider": llm_provider,
        },
    )
