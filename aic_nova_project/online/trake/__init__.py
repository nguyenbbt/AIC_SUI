"""TRAKE/DANTE algorithm package."""

from .config import (
    DANTEConfig,
    DANTE_POLICY_NAME,
    DEFAULT_DANTE_LAMBDA,
    MAX_DANTE_LAMBDA,
    MIN_DANTE_LAMBDA,
)
from .dante import DANTEPath, SimilarityMatrix, solve_dante
from .service import TRAKEExecution, TRAKEService, TRAKEServiceConfig
from .similarity import (
    DEFAULT_NORM_TOLERANCE,
    EncodedTRAKEEvents,
    VideoSimilarityMatrix,
    compute_video_similarity,
    encode_trake_events,
    load_video_similarity,
)

__all__ = [
    "DANTEConfig",
    "DANTEPath",
    "DANTE_POLICY_NAME",
    "DEFAULT_NORM_TOLERANCE",
    "DEFAULT_DANTE_LAMBDA",
    "EncodedTRAKEEvents",
    "MAX_DANTE_LAMBDA",
    "MIN_DANTE_LAMBDA",
    "SimilarityMatrix",
    "TRAKEExecution",
    "TRAKEService",
    "TRAKEServiceConfig",
    "VideoSimilarityMatrix",
    "compute_video_similarity",
    "encode_trake_events",
    "load_video_similarity",
    "solve_dante",
]
