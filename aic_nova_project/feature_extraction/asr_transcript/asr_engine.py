"""Vietnamese ASR with timestamps derived from real audio windows."""

from __future__ import annotations

import logging
import wave
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import pipeline


logger = logging.getLogger(__name__)


class ASREngine:
    """Transcribe fixed WAV windows with PhoWhisper.

    Window boundaries are calculated from PCM sample offsets. This avoids the
    unreliable timestamp-token output seen when PhoWhisper processes an entire
    long recording while keeping every persisted timestamp tied to audio that
    was actually passed to the model.
    """

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "auto",
        window_duration_sec: float = 30.0,
    ) -> None:
        if window_duration_sec <= 0.0 or window_duration_sec > 30.0:
            raise ValueError("ASR window duration must be in the range (0, 30]")

        self.model_name = f"vinai/PhoWhisper-{model_size}"
        self.window_duration_sec = window_duration_sec
        if device == "auto":
            self.device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device_name = device

        if self.device_name == "cpu":
            logger.warning("Running ASR on CPU. This will be very slow.")
        else:
            logger.info("Running ASR on %s.", self.device_name)

        logger.info("Loading ASR model %s...", self.model_name)
        self.transcriber = pipeline(
            "automatic-speech-recognition",
            model=self.model_name,
            device=self.device_name,
            dtype=(
                torch.float16
                if "cuda" in self.device_name
                else torch.float32
            ),
        )
        logger.info("ASR model loaded successfully.")

    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        """Transcribe WAV slices and return the existing raw segment schema."""
        logger.info(
            "Transcribing audio in %.1f-second windows: %s",
            self.window_duration_sec,
            audio_path,
        )
        samples, sample_rate = self._read_wav(audio_path)
        window_samples = max(1, int(round(self.window_duration_sec * sample_rate)))
        segments: List[Dict[str, Any]] = []

        try:
            for start_sample in range(0, len(samples), window_samples):
                end_sample = min(start_sample + window_samples, len(samples))
                audio_window = samples[start_sample:end_sample]
                result = self.transcriber(
                    {
                        "array": audio_window,
                        "sampling_rate": sample_rate,
                    },
                    return_timestamps=False,
                    generate_kwargs={"task": "transcribe", "language": "vi"},
                )
                text = self._extract_text(result)
                if not text:
                    logger.info(
                        "ASR returned no speech for audio window %.3f-%.3f.",
                        start_sample / sample_rate,
                        end_sample / sample_rate,
                    )
                    continue

                segments.append(
                    {
                        "timestamp": (
                            start_sample / sample_rate,
                            end_sample / sample_rate,
                        ),
                        "text": text,
                    }
                )
        except Exception:
            logger.exception("Error during transcription of %s", audio_path)
            raise

        if not segments:
            raise ValueError("ASR result contains no usable audio windows")
        return segments

    @staticmethod
    def _read_wav(audio_path: str) -> Tuple[np.ndarray, int]:
        """Read the mono 16-bit PCM contract produced by ``AudioExtractor``."""
        try:
            with wave.open(audio_path, "rb") as audio_file:
                channels = audio_file.getnchannels()
                sample_width = audio_file.getsampwidth()
                sample_rate = audio_file.getframerate()
                frame_count = audio_file.getnframes()
                pcm_data = audio_file.readframes(frame_count)
        except (OSError, wave.Error) as exc:
            raise ValueError(f"cannot read WAV audio {audio_path}") from exc

        if channels != 1 or sample_width != 2 or sample_rate <= 0:
            raise ValueError(
                "ASR input must be mono 16-bit PCM WAV with a valid sample rate"
            )

        samples = np.frombuffer(pcm_data, dtype="<i2").astype(np.float32)
        samples /= 32768.0
        if samples.size == 0:
            raise ValueError("ASR input WAV contains no audio samples")
        return samples, sample_rate

    @staticmethod
    def _extract_text(result: Any) -> str:
        if not isinstance(result, dict):
            raise ValueError("ASR provider returned an invalid response")
        text = result.get("text")
        if not isinstance(text, str):
            raise ValueError("ASR provider response does not contain text")
        return text.strip()
