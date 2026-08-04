import json

from data_pipeline.shot_keyframe.metadata_schema import (
    KeyframeMetadata,
    ShotMetadata,
    VideoMetadata,
)
from feature_extraction.ocr.src.ocr_module.metadata_reader import (
    get_keyframes_from_metadata,
)
from feature_extraction.text_embedding.src.text_embedding.data_readers import (
    parse_ocr_file,
)


def test_ocr_contract_uses_canonical_frame_id_across_m1_m4_m6(tmp_path):
    metadata = VideoMetadata(
        video_id="V001",
        source_path="videos/V001.mp4",
        fps=30.0,
        duration_sec=1.0,
        num_shots=1,
        shots=[
            ShotMetadata(
                shot_id=0,
                start_frame=0,
                end_frame=30,
                start_time_sec=0.0,
                end_time_sec=1.0,
                keyframes=[
                    KeyframeMetadata(
                        position=0.15,
                        frame_index=4,
                        time_sec=0.133,
                        file_path="keyframes/V001/shot_00000_pos_015.webp",
                    )
                ],
            )
        ],
    )
    metadata_path = tmp_path / "V001.json"
    metadata_path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")

    frames = get_keyframes_from_metadata(metadata_path)

    assert frames == [
        {
            "frame_id": "V001_00000_015",
            "shot_id": 0,
            "position": 0.15,
            "file_path": "keyframes/V001/shot_00000_pos_015.webp",
        }
    ]

    ocr_dir = tmp_path / "ocr"
    ocr_dir.mkdir()
    ocr_path = ocr_dir / "V001.json"
    ocr_path.write_text(
        json.dumps(
            {
                "video_id": "V001",
                "frames": [
                    {
                        **frames[0],
                        "ocr_text_concat": "Biển báo giao thông",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    records = parse_ocr_file(ocr_path)
    assert records[0]["video_id"] == "V001"
    assert records[0]["frame_id"] == "V001_00000_015"
