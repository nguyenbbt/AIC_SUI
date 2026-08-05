import pytest

from data_pipeline.shot_keyframe.transnet_wrapper import TransNetPredictor


def _predictor_with_model(model):
    predictor = TransNetPredictor.__new__(TransNetPredictor)
    predictor.model = model
    return predictor


def test_predict_shots_passes_configured_threshold():
    class ThresholdModel:
        received_threshold = None

        def detect_scenes(self, video_path, threshold=0.5):
            self.received_threshold = threshold
            return [(0, 9)]

    model = ThresholdModel()
    predictor = _predictor_with_model(model)

    assert predictor.predict_shots("video.mp4", threshold=0.73) == [(0, 9)]
    assert model.received_threshold == 0.73


def test_predict_shots_falls_back_for_legacy_detect_scenes_signature():
    class LegacyModel:
        def detect_scenes(self, video_path):
            return [(0, 9)]

    predictor = _predictor_with_model(LegacyModel())

    assert predictor.predict_shots("video.mp4", threshold=0.73) == [(0, 9)]


def test_predict_shots_does_not_mask_internal_type_error():
    class BrokenModel:
        def detect_scenes(self, video_path, threshold):
            raise TypeError("internal model failure")

    predictor = _predictor_with_model(BrokenModel())

    with pytest.raises(TypeError, match="internal model failure"):
        predictor.predict_shots("video.mp4", threshold=0.73)
