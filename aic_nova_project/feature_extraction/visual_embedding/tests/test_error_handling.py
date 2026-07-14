import pytest
import os
import shutil
import json
from pathlib import Path
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from feature_extraction.visual_embedding.pipeline import run_pipeline
import pyarrow.parquet as pq

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')
METADATA_DIR = os.path.join(FIXTURE_DIR, 'metadata')
KEYFRAME_DIR = os.path.join(FIXTURE_DIR, 'keyframes')
OUTPUT_DIR = os.path.join(FIXTURE_DIR, 'output_err')

@pytest.fixture(autouse=True)
def setup_teardown():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    yield
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

def test_error_handling_missing_image():
    # Create metadata with one missing image and one valid image
    err_meta_dir = os.path.join(FIXTURE_DIR, 'metadata_err')
    os.makedirs(err_meta_dir, exist_ok=True)
    
    meta = {
        'video_id': 'V002', 
        'shots': [
            {'shot_id': 0, 'keyframes': [{'position': 0.1, 'frame_index': 10, 'time_sec': 0.5, 'file_path': 'keyframes/V001/shot_00000_pos_015.webp'}]}, # valid (exists from previous fixture)
            {'shot_id': 1, 'keyframes': [{'position': 0.2, 'frame_index': 20, 'time_sec': 1.0, 'file_path': 'keyframes/V002/missing.webp'}]} # invalid
        ]
    }
    
    with open(os.path.join(err_meta_dir, 'V002.json'), 'w') as f:
        json.dump(meta, f)
        
    try:
        run_pipeline(
            metadata_dir=err_meta_dir,
            keyframe_dir=KEYFRAME_DIR,
            output_dir=OUTPUT_DIR,
            model_id="hf-hub:timm/PE-Core-bigG-14-448",
            device="cpu",
            precision="fp32",
            batch_size=2,
            num_workers=0,
            force=True
        )
        
        # Pipeline should not crash. Output should exist but only have 1 row
        out_file = os.path.join(OUTPUT_DIR, "V002.parquet")
        assert os.path.exists(out_file)
        
        table = pq.read_table(out_file)
        assert table.num_rows == 1
        assert table.column("frame_id")[0].as_py() == "V002_00000_010"
        
    finally:
        shutil.rmtree(err_meta_dir)
