import pytest
import os
import tempfile
import json
from data_pipeline.shot_keyframe.pipeline import VideoProcessor

def test_pipeline_e2e(mock_video_path):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock TransNetPredictor to return deterministic shots (since we don't want to run the real heavy model in unit test by default)
        # Actually, if we have transnetv2-pytorch installed, it will download weights and run it. 
        # But let's mock it to make the test fast and offline-friendly.
        
        processor = VideoProcessor(output_dir=temp_dir, device="cpu")
        
        # Override transnet with a mock
        class MockTransNet:
            def predict_shots(self, video_path, threshold):
                return [(0, 29), (30, 59), (60, 89)]
        
        processor.transnet = MockTransNet()
        
        # Run pipeline
        success = processor.process_video(mock_video_path)
        assert success == True
        
        video_id = "test_video"
        
        # Check metadata
        meta_path = os.path.join(temp_dir, "metadata", f"{video_id}.json")
        assert os.path.exists(meta_path)
        
        with open(meta_path, 'r') as f:
            data = json.load(f)
            
        assert data["video_id"] == video_id
        assert data["contract_version"] == "self-indexed-v2"
        assert data["source_video_rel_path"] == "videos/test_video.mp4"
        assert data["frame_count"] == 90
        assert data["width"] == 320
        assert data["height"] == 240
        assert data["num_shots"] == 3
        assert len(data["shots"]) == 3
        
        # Check if 3 * 3 = 9 images are generated
        for shot in data["shots"]:
            assert len(shot["keyframes"]) == 3
            for kf in shot["keyframes"]:
                assert kf["source_frame_idx"] == kf["frame_index"]
                assert kf["image_rel_path"] == kf["file_path"]
                assert kf["position_code"] in (15, 50, 85)
                abs_path = os.path.join(temp_dir, kf["file_path"])
                assert os.path.exists(abs_path)

def test_pipeline_resume(mock_video_path):
    with tempfile.TemporaryDirectory() as temp_dir:
        processor = VideoProcessor(output_dir=temp_dir, device="cpu")
        class MockTransNet:
            def predict_shots(self, video_path, threshold):
                return [(0, 89)] # single shot
        processor.transnet = MockTransNet()
        
        # Run first time
        success1 = processor.process_video(mock_video_path)
        assert success1 == True
        
        # Check metadata exists
        meta_path = os.path.join(temp_dir, "metadata", "test_video.json")
        assert os.path.exists(meta_path)
        
        # Run second time
        # We can check if TransNet is called. If not, resume works.
        class ErrorTransNet:
            def predict_shots(self, video_path, threshold):
                raise Exception("Should not be called during resume!")
        processor.transnet = ErrorTransNet()
        
        success2 = processor.process_video(mock_video_path)
        assert success2 == True

        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        missing_keyframe = os.path.join(
            temp_dir,
            metadata["shots"][0]["keyframes"][0]["file_path"],
        )
        os.remove(missing_keyframe)

        class CountingTransNet:
            called = False

            def predict_shots(self, video_path, threshold):
                self.called = True
                return [(0, 89)]

        counting_transnet = CountingTransNet()
        processor.transnet = counting_transnet

        success3 = processor.process_video(mock_video_path)
        assert success3 == True
        assert counting_transnet.called
        assert os.path.exists(missing_keyframe)
