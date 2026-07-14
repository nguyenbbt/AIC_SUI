import pytest
import os
import shutil
from pathlib import Path
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from feature_extraction.visual_embedding.pipeline import run_pipeline
from feature_extraction.visual_embedding.metadata_reader import read_metadata

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')
METADATA_DIR = os.path.join(FIXTURE_DIR, 'metadata')
KEYFRAME_DIR = os.path.join(FIXTURE_DIR, 'keyframes')
OUTPUT_DIR = os.path.join(FIXTURE_DIR, 'output')

@pytest.fixture(autouse=True)
def setup_teardown():
    # Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    yield
    # Teardown
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

def test_resume_skip_existing():
    # Run first time
    run_pipeline(
        metadata_dir=METADATA_DIR,
        keyframe_dir=KEYFRAME_DIR,
        output_dir=OUTPUT_DIR,
        model_id="hf-hub:timm/PE-Core-bigG-14-448",
        device="cpu",
        precision="fp32",
        batch_size=1,
        num_workers=0,
        force=False
    )
    
    # Check that output exists
    assert os.path.exists(os.path.join(OUTPUT_DIR, "V001.parquet"))
    
    # Run second time without force
    # We should mock process_video_batch or just observe the logs, but it's simpler to just ensure it doesn't fail.
    # To truly verify, we could check the modification time of the file.
    mtime1 = os.path.getmtime(os.path.join(OUTPUT_DIR, "V001.parquet"))
    
    run_pipeline(
        metadata_dir=METADATA_DIR,
        keyframe_dir=KEYFRAME_DIR,
        output_dir=OUTPUT_DIR,
        model_id="hf-hub:timm/PE-Core-bigG-14-448",
        device="cpu",
        precision="fp32",
        batch_size=1,
        num_workers=0,
        force=False
    )
    
    mtime2 = os.path.getmtime(os.path.join(OUTPUT_DIR, "V001.parquet"))
    assert mtime1 == mtime2 # File was not modified because it was skipped
    
    # Run third time with force
    run_pipeline(
        metadata_dir=METADATA_DIR,
        keyframe_dir=KEYFRAME_DIR,
        output_dir=OUTPUT_DIR,
        model_id="hf-hub:timm/PE-Core-bigG-14-448",
        device="cpu",
        precision="fp32",
        batch_size=1,
        num_workers=0,
        force=True
    )
    
    mtime3 = os.path.getmtime(os.path.join(OUTPUT_DIR, "V001.parquet"))
    assert mtime3 > mtime2 # File was modified
