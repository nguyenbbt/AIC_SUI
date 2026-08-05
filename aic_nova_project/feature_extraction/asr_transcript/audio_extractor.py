import os
import subprocess
import logging

logger = logging.getLogger(__name__)

class AudioExtractor:
    """
    Extracts audio from video files using ffmpeg.
    Converts to mono, 16kHz WAV format for ASR.
    """

    @staticmethod
    def extract_audio(video_path: str, output_path: str, force: bool = False) -> bool:
        """
        Extracts 16kHz mono audio from the given video file.
        
        Args:
            video_path: Path to the input video file.
            output_path: Path to save the extracted .wav file.
            force: If True, overwrite the output file if it exists.
            
        Returns:
            True if extraction was successful or file already exists (and not forced).
            False if extraction failed (e.g., no audio stream or ffmpeg error).
        """
        if os.path.exists(output_path) and not force:
            logger.debug(f"Audio file already exists at {output_path}, skipping extraction.")
            return True

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # ffmpeg command to extract audio: 
        # -y (overwrite), -i (input), -vn (no video), -acodec pcm_s16le (wav), 
        # -ar 16000 (16kHz), -ac 1 (mono)
        command = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            output_path
        ]

        try:
            # Run ffmpeg, capturing output to avoid cluttering stdout, unless it errors
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            logger.info(f"Successfully extracted audio to {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            # This might happen if the video has no audio track, or ffmpeg is not installed
            logger.error(f"Failed to extract audio from {video_path}")
            logger.error(f"ffmpeg stderr: {e.stderr}")
            # If a partial file was created, remove it
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except Exception:
                    pass
            return False
        except FileNotFoundError:
            logger.error("ffmpeg command not found. Please ensure ffmpeg is installed and in your PATH.")
            return False
