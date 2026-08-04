from ocr_module.pipeline import OCRPipeline
import logging
import pytest
from unittest.mock import patch, MagicMock

def test_pipeline_missing_image(tmp_path, caplog):
    # Setup
    k_dir = tmp_path / "keyframes"
    m_dir = tmp_path / "metadata"
    o_dir = tmp_path / "ocr"
    
    k_dir.mkdir()
    m_dir.mkdir()
    o_dir.mkdir()
    
    video_id = "V_ERR"
    (k_dir / video_id).mkdir()
    
    import json
    with open(m_dir / f"{video_id}.json", "w") as f:
        json.dump({"frames": [{"frame_id": "V_ERR_001"}]}, f)
        
    # We will patch the detector and recognizer to not actually initialize models
    with patch('ocr_module.pipeline.TextDetector'), patch('ocr_module.pipeline.TextRecognizer'):
        pipeline = OCRPipeline(use_gpu=False)
        
        with caplog.at_level(logging.ERROR):
            with pytest.raises(FileNotFoundError, match="Image not found"):
                pipeline.process_video(video_id, k_dir, m_dir, o_dir)
            
        assert "Image not found" in caplog.text
        assert not (o_dir / f"{video_id}.json").exists()


def test_pipeline_does_not_write_partial_output_for_unreadable_image(
    tmp_path,
):
    k_dir = tmp_path / "keyframes"
    m_dir = tmp_path / "metadata"
    o_dir = tmp_path / "ocr"
    video_id = "V_ERR"
    video_dir = k_dir / video_id
    video_dir.mkdir(parents=True)
    m_dir.mkdir()
    o_dir.mkdir()
    (video_dir / "bad.webp").write_bytes(b"not an image")

    import json
    with open(m_dir / f"{video_id}.json", "w") as f:
        json.dump(
            {
                "frames": [
                    {
                        "frame_id": "V_ERR_00000_000",
                        "file_path": "bad.webp",
                    }
                ]
            },
            f,
        )

    with patch('ocr_module.pipeline.TextDetector'), patch(
        'ocr_module.pipeline.TextRecognizer'
    ), patch('ocr_module.pipeline.cv2.imread', return_value=None):
        pipeline = OCRPipeline(use_gpu=False)

        with pytest.raises(RuntimeError, match="Failed to load image"):
            pipeline.process_video(video_id, k_dir, m_dir, o_dir)

    assert not (o_dir / f"{video_id}.json").exists()
