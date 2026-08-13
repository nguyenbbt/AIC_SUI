from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from online.ports.records import FrameMetadata, ObjectLabelStat, VideoMetadata
from retrieval_api.search_engine import create_app
from retrieval_api.ui_resources import DatasetUIResources


class DatasetReader:
    def __init__(self) -> None:
        self.frames = {
            "V001_00000_001": FrameMetadata(frame_id="V001_00000_001", video_id="V001", shot_id=0, source_frame_idx=30, timestamp_sec=1, image_rel_path="keyframes/V001/001.jpg"),
            "V001_00000_002": FrameMetadata(frame_id="V001_00000_002", video_id="V001", shot_id=0, source_frame_idx=60, timestamp_sec=2, image_rel_path="keyframes/V001/002.jpg"),
        }

    def get_frames_by_ids(self, ids):
        return {value: self.frames[value] for value in ids if value in self.frames}

    def get_ordered_frames_by_video(self, video_id):
        return tuple(self.frames.values()) if video_id == "V001" else ()

    def get_videos_by_ids(self, ids):
        video = VideoMetadata(video_id="V001", source_video_rel_path="videos/V001.mp4", fps=30, duration_sec=2, frame_count=60, width=2, height=2)
        return {value: video for value in ids if value == "V001"}

    def list_object_labels(self):
        return (ObjectLabelStat(label="person", detection_count=9),)


def test_catalog_neighbors_and_media_are_dataset_backed(tmp_path: Path) -> None:
    (tmp_path / "keyframes/V001").mkdir(parents=True)
    (tmp_path / "videos").mkdir()
    (tmp_path / "keyframes/V001/001.jpg").write_bytes(b"jpeg-one")
    (tmp_path / "keyframes/V001/002.jpg").write_bytes(b"jpeg-two")
    (tmp_path / "videos/V001.mp4").write_bytes(b"0123456789")
    reader = DatasetReader()
    resources = DatasetUIResources(
        data_root=tmp_path,
        metadata_reader=reader,
        object_catalog=reader,
        identity_provider=lambda: ("dataset-a", "sha256:" + "a" * 64),
    )
    client = TestClient(create_app(ui_resources=resources))

    catalog = client.get("/catalog/object-labels").json()
    assert catalog["source"] == "sqlite"
    assert catalog["labels"] == [{"label": "person", "detection_count": 9}]
    assert client.get("/media/keyframes/V001_00000_001").content == b"jpeg-one"
    neighbors = client.get("/media/keyframes/V001_00000_001/neighbors?before=0&after=1").json()
    assert [item["source_frame_idx"] for item in neighbors["frames"]] == [30, 60]
    ranged = client.get("/media/videos/V001", headers={"Range": "bytes=2-5"})
    assert ranged.status_code == 206
    assert ranged.content == b"2345"


def test_unconfigured_ui_resources_fail_closed() -> None:
    response = TestClient(create_app()).get("/catalog/object-labels")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "RESOURCE_UNAVAILABLE"
