"""Backward-compatible import for the former PE-Core encoder name."""

from .open_clip_encoder import OpenCLIPEncoder

PECoreEncoder = OpenCLIPEncoder

__all__ = ["PECoreEncoder"]
