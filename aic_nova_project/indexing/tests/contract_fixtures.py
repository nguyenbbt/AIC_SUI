"""Canonical self-indexed-v2 records shared by indexing tests."""


def canonical_video_record(video_id: str) -> dict:
    """Build the required SQLite source-video record."""
    return {
        "video_id": video_id,
        "source_video_rel_path": f"videos/{video_id}.mp4",
        "fps": 25.0,
        "duration_sec": 4.0,
        "frame_count": 100,
        "width": 1920,
        "height": 1080,
    }
