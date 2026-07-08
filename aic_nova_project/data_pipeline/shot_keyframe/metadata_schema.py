from typing import List
from pydantic import BaseModel, Field

class KeyframeMetadata(BaseModel):
    position: float = Field(..., ge=0.0, le=1.0, description="Normalized position within the shot (e.g., 0.15)")
    frame_index: int = Field(..., ge=0, description="The absolute frame index in the video")
    time_sec: float = Field(..., ge=0.0, description="The timestamp in seconds")
    file_path: str = Field(..., description="Relative path to the saved WebP image")

class ShotMetadata(BaseModel):
    shot_id: int = Field(..., ge=0, description="ID of the shot within the video (0-indexed)")
    start_frame: int = Field(..., ge=0, description="Start frame index of the shot")
    end_frame: int = Field(..., ge=0, description="End frame index of the shot")
    start_time_sec: float = Field(..., ge=0.0, description="Start time in seconds")
    end_time_sec: float = Field(..., ge=0.0, description="End time in seconds")
    keyframes: List[KeyframeMetadata] = Field(..., description="List of keyframes extracted for this shot")

class VideoMetadata(BaseModel):
    video_id: str = Field(..., description="Unique identifier for the video")
    source_path: str = Field(..., description="Path to the original video file")
    fps: float = Field(..., gt=0.0, description="Frames per second")
    duration_sec: float = Field(..., ge=0.0, description="Total duration in seconds")
    num_shots: int = Field(..., ge=0, description="Total number of detected shots")
    shots: List[ShotMetadata] = Field(..., description="List of shots and their keyframes")
