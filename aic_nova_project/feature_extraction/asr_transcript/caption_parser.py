import os
import re
from typing import List, Dict, Any, Optional

class CaptionParser:
    """
    Parses subtitle files (.srt, .vtt) into segment dictionaries.
    """

    @staticmethod
    def parse_time_to_seconds(time_str: str) -> float:
        """
        Converts SRT/VTT timestamp string to seconds.
        Example SRT: 00:01:23,456
        Example VTT: 00:01:23.456
        """
        # Replace comma with dot for uniform parsing
        time_str = time_str.replace(',', '.')
        parts = time_str.split(':')
        
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h = 0
            m, s = parts
        else:
            return 0.0

        return float(h) * 3600 + float(m) * 60 + float(s)

    @staticmethod
    def parse_srt(file_path: str) -> List[Dict[str, Any]]:
        segments = []
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                content = f.read()
        except Exception:
            return []

        # Split blocks by double newline
        blocks = re.split(r'\n\s*\n', content.strip())
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                # Line 0: Index, Line 1: Timestamp, Line 2+: Text
                time_line = lines[1]
                if '-->' in time_line:
                    start_str, end_str = time_line.split('-->')
                    start_time = CaptionParser.parse_time_to_seconds(start_str.strip())
                    end_time = CaptionParser.parse_time_to_seconds(end_str.strip())
                    text = ' '.join([line.strip() for line in lines[2:]])
                    
                    segments.append({
                        "timestamp": (start_time, end_time),
                        "text": text
                    })
        return segments

    @staticmethod
    def parse_vtt(file_path: str) -> List[Dict[str, Any]]:
        segments = []
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                content = f.read()
        except Exception:
            return []

        blocks = re.split(r'\n\s*\n', content.strip())
        for block in blocks:
            if block.startswith('WEBVTT'):
                continue
            
            lines = block.strip().split('\n')
            # VTT might or might not have an index line
            time_line_idx = -1
            for i, line in enumerate(lines):
                if '-->' in line:
                    time_line_idx = i
                    break
            
            if time_line_idx != -1 and len(lines) > time_line_idx + 1:
                time_line = lines[time_line_idx]
                start_str, end_str = time_line.split('-->')
                start_time = CaptionParser.parse_time_to_seconds(start_str.strip())
                end_time = CaptionParser.parse_time_to_seconds(end_str.strip().split(' ')[0]) # Strip extra VTT styling settings
                text = ' '.join([line.strip() for line in lines[time_line_idx+1:]])
                
                segments.append({
                    "timestamp": (start_time, end_time),
                    "text": text
                })
        return segments

    @staticmethod
    def get_captions(video_id: str, captions_dir: str) -> Optional[List[Dict[str, Any]]]:
        """
        Looks for a caption file (.srt or .vtt) for the given video_id and parses it.
        Returns a list of segments or None if no caption file is found.
        """
        if not os.path.isdir(captions_dir):
            return None

        srt_path = os.path.join(captions_dir, f"{video_id}.srt")
        if os.path.exists(srt_path):
            return CaptionParser.parse_srt(srt_path)

        vtt_path = os.path.join(captions_dir, f"{video_id}.vtt")
        if os.path.exists(vtt_path):
            return CaptionParser.parse_vtt(vtt_path)

        return None
