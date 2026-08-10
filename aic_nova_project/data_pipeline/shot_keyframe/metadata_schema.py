from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, List, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, model_validator


def _validate_relative_posix_path(value: str, field_name: str) -> str:
    """Return a safe POSIX path relative to the configured dataset root."""
    if not value or value != value.strip() or "\\" in value or "\x00" in value:
        raise ValueError(f"{field_name} must be a normalized POSIX relative path")

    parsed = urlsplit(value)
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        parsed.scheme
        or parsed.netloc
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or ".." in posix_path.parts
    ):
        raise ValueError(f"{field_name} must be a safe relative path")
    return value

class KeyframeMetadata(BaseModel):
    position: float = Field(..., ge=0.0, le=1.0, description="Normalized position within the shot (e.g., 0.15)")
    position_code: int = Field(..., ge=0, le=100)
    frame_index: int = Field(..., ge=0, description="The absolute frame index in the video")
    source_frame_idx: int = Field(..., ge=0, description="Zero-based frame that was actually decoded")
    time_sec: float = Field(..., ge=0.0, description="The timestamp in seconds")
    file_path: str = Field(..., description="Relative path to the saved WebP image")
    image_rel_path: str = Field(..., description="POSIX path relative to the dataset root")
    image_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 of the published WebP image",
    )

    @model_validator(mode="before")
    @classmethod
    def populate_additive_contract_fields(cls, value: Any) -> Any:
        """Keep legacy fields while exposing canonical self-indexed-v2 names."""
        if not isinstance(value, dict):
            return value
        values = dict(value)
        if "position" in values:
            values.setdefault("position_code", round(float(values["position"]) * 100))
        if "frame_index" in values:
            values.setdefault("source_frame_idx", values["frame_index"])
        if "file_path" in values:
            values.setdefault("image_rel_path", values["file_path"])
        return values

    @model_validator(mode="after")
    def validate_aliases(self) -> "KeyframeMetadata":
        expected_position_code = round(self.position * 100)
        if self.position_code != expected_position_code:
            raise ValueError("position_code does not match position")
        if self.frame_index != self.source_frame_idx:
            raise ValueError("frame_index must equal source_frame_idx")
        if self.file_path != self.image_rel_path:
            raise ValueError("file_path must equal image_rel_path")
        _validate_relative_posix_path(self.file_path, "file_path")
        return self

class ShotMetadata(BaseModel):
    shot_id: int = Field(..., ge=0, description="ID of the shot within the video (0-indexed)")
    start_frame: int = Field(..., ge=0, description="Start frame index of the shot")
    end_frame: int = Field(..., ge=0, description="End frame index of the shot")
    start_time_sec: float = Field(..., ge=0.0, description="Start time in seconds")
    end_time_sec: float = Field(..., ge=0.0, description="End time in seconds")
    keyframes: List[KeyframeMetadata] = Field(..., description="List of keyframes extracted for this shot")

class VideoMetadata(BaseModel):
    contract_version: Literal["self-indexed-v2"] = "self-indexed-v2"
    video_id: str = Field(..., description="Unique identifier for the video")
    source_path: str = Field(..., description="Path to the original video file")
    source_video_rel_path: str = Field(..., description="POSIX path relative to the dataset root")
    source_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 of the raw source video",
    )
    producer_config_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 of the Module 1 producer configuration",
    )
    fps: float = Field(..., gt=0.0, description="Frames per second")
    duration_sec: float = Field(..., ge=0.0, description="Total duration in seconds")
    frame_count: int = Field(..., gt=0)
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    num_shots: int = Field(..., ge=0, description="Total number of detected shots")
    shots: List[ShotMetadata] = Field(..., description="List of shots and their keyframes")

    @model_validator(mode="after")
    def validate_video_contract(self) -> "VideoMetadata":
        _validate_relative_posix_path(
            self.source_video_rel_path,
            "source_video_rel_path",
        )
        for shot in self.shots:
            for keyframe in shot.keyframes:
                if keyframe.source_frame_idx >= self.frame_count:
                    raise ValueError("source_frame_idx must be below frame_count")
        return self
