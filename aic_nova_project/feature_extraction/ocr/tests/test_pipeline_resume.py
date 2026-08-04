import json
from pathlib import Path
import os
from ocr_module.pipeline import OCRPipeline
from ocr_module.resume_validation import (
    OCR_SCHEMA_VERSION,
    build_ocr_provenance,
)

def test_pipeline_resume(tmp_path):
    # Mock directories
    k_dir = tmp_path / "keyframes"
    m_dir = tmp_path / "metadata"
    o_dir = tmp_path / "ocr"
    
    k_dir.mkdir()
    m_dir.mkdir()
    o_dir.mkdir()
    
    video_id = "V001"
    out_file = o_dir / f"{video_id}.json"
    frame_id = "V001_00000_015"
    with open(m_dir / f"{video_id}.json", "w") as f:
        json.dump({"frames": [{"frame_id": frame_id}]}, f)

    provenance = build_ocr_provenance(
        backbone="vgg_transformer",
        confidence_threshold=0.4,
        width_ths=0.7,
        mag_ratio=1.5,
    )
    expected = {
        "schema_version": OCR_SCHEMA_VERSION,
        "video_id": video_id,
        "provenance": provenance,
        "frames": [{"frame_id": frame_id}],
    }
    with open(out_file, 'w') as f:
        json.dump(expected, f)
        
    pipeline = OCRPipeline(use_gpu=False)
    
    # Run pipeline with force=False
    pipeline.process_video(video_id, k_dir, m_dir, o_dir, force=False)
    
    # Ensure it wasn't overwritten
    with open(out_file, 'r') as f:
        data = json.load(f)
        assert data == expected
