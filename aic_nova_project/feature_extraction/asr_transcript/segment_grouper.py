from typing import List, Dict, Any

class SegmentGrouper:
    """
    Groups raw transcript segments into larger intervals.
    """
    @staticmethod
    def group_segments(segments: List[Dict[str, Any]], group_size: int = 5) -> List[Dict[str, Any]]:
        """
        Groups `group_size` consecutive segments into one interval.
        
        Args:
            segments: List of dicts with 'timestamp' and 'text'.
            group_size: Number of segments per interval.
            
        Returns:
            List of intervals.
        """
        intervals = []
        interval_id = 0
        
        for i in range(0, len(segments), group_size):
            chunk = segments[i:i + group_size]
            
            # Determine start and end times
            # Timestamp is a tuple (start, end)
            start_time = chunk[0]['timestamp'][0]
            
            # End time is the end time of the last segment in the chunk.
            # If it's None (sometimes happens at the very end of whisper output), use a fallback
            end_time = chunk[-1]['timestamp'][1]
            if end_time is None:
                # Fallback: start time of last segment + a few seconds, or just None
                last_start = chunk[-1]['timestamp'][0]
                if last_start is not None:
                    end_time = last_start + 2.0
                elif start_time is not None:
                    end_time = start_time + 2.0
                else:
                    end_time = 0.0
                    
            if start_time is None:
                start_time = 0.0
                
            raw_text = " ".join([seg['text'] for seg in chunk if seg['text']])
            segment_ids = list(range(i, i + len(chunk)))
            
            intervals.append({
                "interval_id": str(interval_id),
                "start_time_sec": start_time,
                "end_time_sec": end_time,
                "raw_text": raw_text,
                "segment_ids": segment_ids
            })
            interval_id += 1
            
        return intervals
