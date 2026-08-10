"""Canonical enums shared by all Online components."""

from enum import Enum


class QueryMode(str, Enum):
    KIS_TEXT = "kis_text"
    KIS_VIDEO = "kis_video"
    TRAKE = "trake"
    VQA = "vqa"


class RetrievalBranch(str, Enum):
    VISUAL_DENSE = "visual_dense"
    OCR_DENSE = "ocr_dense"
    OCR_BM25 = "ocr_bm25"
    ASR_DENSE = "asr_dense"
    ASR_BM25 = "asr_bm25"
    SUMMARY_DENSE = "summary_dense"
    SUMMARY_BM25 = "summary_bm25"


class CandidateLevel(str, Enum):
    FRAME = "frame"
    ASR_INTERVAL = "asr_interval"
    VIDEO = "video"


class BranchStatus(str, Enum):
    SUCCESS = "success"
    DEGRADED = "degraded"
    FAILED = "failed"
    DISABLED = "disabled"


class CountOperator(str, Enum):
    EQ = "eq"
    GTE = "gte"
    LTE = "lte"


class FilterMode(str, Enum):
    HARD = "hard"
    SOFT = "soft"
