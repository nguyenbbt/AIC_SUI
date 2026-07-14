import json
from pathlib import Path
import os
from ocr_module.pipeline import OCRPipeline

def test_pipeline_resume(tmp_path):
    # Mock directories
    k_dir = tmp_path / "keyframes"
    m_dir = tmp_path / "metadata"
    o_dir = tmp_path / "ocr"
    
    k_dir.mkdir()
    m_dir.mkdir()
    o_dir.mkdir()
    
    # Create mock existing output
    video_id = "V001"
    out_file = o_dir / f"{video_id}.json"
    
    with open(out_file, 'w') as f:
        json.dump({"existing": "data"}, f)
        
    pipeline = OCRPipeline(use_gpu=False)
    
    # Run pipeline with force=False
    pipeline.process_video(video_id, k_dir, m_dir, o_dir, force=False)
    
    # Ensure it wasn't overwritten
    with open(out_file, 'r') as f:
        data = json.load(f)
        assert data == {"existing": "data"}
