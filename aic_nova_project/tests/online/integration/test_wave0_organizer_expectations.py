"""C-side Wave 0 consumer expectations for the A0 organizer-v1 contract."""

from __future__ import annotations

import unittest
from collections.abc import Mapping
from pathlib import PurePosixPath

from online.domain.candidates import FrameCandidate, FusedFrameCandidate
from online.domain.trake import TRAKEFrameMatch
from online.domain.vqa import ImageEvidence
from online.ports.records import FrameMetadata
from tests.online.fixtures.mode_fixtures import organizer_wave0_mode_expectations


_C_REQUIRED_IDENTITY_FIELDS = {
    FrameMetadata: {
        "frame_id",
        "video_id",
        "keyframe_no",
        "local_index",
        "timestamp_sec",
        "fps",
        "source_frame_idx",
        "image_rel_path",
    },
    FrameCandidate: {
        "frame_id",
        "video_id",
        "keyframe_no",
        "local_index",
        "timestamp_sec",
        "source_frame_idx",
    },
    FusedFrameCandidate: {
        "frame_id",
        "video_id",
        "keyframe_no",
        "local_index",
        "timestamp_sec",
        "source_frame_idx",
    },
    TRAKEFrameMatch: {
        "frame_id",
        "video_id",
        "keyframe_no",
        "local_index",
        "timestamp_sec",
        "source_frame_idx",
    },
    ImageEvidence: {
        "frame_id",
        "video_id",
        "keyframe_no",
        "timestamp_sec",
        "source_frame_idx",
    },
}


def _a0_contract_is_available() -> bool:
    return all(
        required.issubset(model.model_fields)
        and "shot_id" not in model.model_fields
        for model, required in _C_REQUIRED_IDENTITY_FIELDS.items()
    )


class CWave0OrganizerExpectationTests(unittest.TestCase):
    def test_cases_cover_duplicate_source_frame_variable_fps_and_safe_paths(self) -> None:
        fixture = organizer_wave0_mode_expectations()
        metadata = fixture.frame_metadata_payloads

        self.assertEqual({row["fps"] for row in metadata}, {30.0, 29.97})
        self.assertEqual(
            tuple(row["source_frame_idx"] for row in metadata[:2]),
            (0, 0),
        )
        self.assertNotEqual(metadata[0]["frame_id"], metadata[1]["frame_id"])
        for row in metadata:
            self.assertEqual(row["local_index"], row["keyframe_no"] - 1)
            self.assertEqual(
                row["frame_id"],
                f"{row['video_id']}_{row['keyframe_no']:03d}",
            )
            path = PurePosixPath(str(row["image_rel_path"]))
            self.assertFalse(path.is_absolute())
            self.assertNotIn("..", path.parts)

    def test_expected_asr_mapping_uses_timestamp_not_source_frame_or_fps(self) -> None:
        fixture = organizer_wave0_mode_expectations()
        interval = fixture.asr_interval_payload
        mapped = tuple(
            str(row["frame_id"])
            for row in fixture.frame_metadata_payloads
            if row["video_id"] == interval["video_id"]
            and interval["start_time_sec"] <= row["timestamp_sec"] <= interval["end_time_sec"]
        )
        self.assertEqual(mapped, fixture.expected_asr_mapped_frame_ids)
        self.assertEqual(mapped, ("L21_V001_001", "L21_V001_002"))

    def test_expected_kis_dedup_and_tie_break_are_competition_frame_based(self) -> None:
        fixture = organizer_wave0_mode_expectations()
        fused = fixture.expected_fused_payloads
        self.assertEqual(
            tuple(str(row["frame_id"]) for row in fused),
            (
                "L21_V001_002",
                "L21_V001_001",
                "L21_V002_001",
                "L21_V001_003",
            ),
        )

        kept: list[Mapping[str, object]] = []
        seen: set[tuple[object, object]] = set()
        for row in fused:
            key = (row["video_id"], row["source_frame_idx"])
            if key not in seen:
                seen.add(key)
                kept.append(row)

        self.assertEqual(
            tuple(str(row["frame_id"]) for row in kept),
            fixture.expected_kis_frame_ids_after_dedup,
        )
        self.assertEqual(
            tuple((str(row["video_id"]), int(row["source_frame_idx"])) for row in kept),
            fixture.expected_kis_competition_rows,
        )

    def test_mode_outputs_use_source_frame_idx_and_never_internal_frame_id(self) -> None:
        fixture = organizer_wave0_mode_expectations()
        kis_rows = fixture.expected_kis_competition_rows
        trake_video, trake_frames = fixture.expected_trake_competition_row
        vqa_video, vqa_frame, answer = fixture.expected_vqa_competition_row

        self.assertTrue(all(isinstance(frame_idx, int) for _, frame_idx in kis_rows))
        self.assertEqual((trake_video, trake_frames), ("L21_V001", (0, 30)))
        self.assertIsInstance(vqa_frame, int)
        self.assertEqual(vqa_video, "L21_V001")
        self.assertNotIn("L21_V001_001", answer)

    @unittest.skipUnless(
        _a0_contract_is_available(),
        "A0 organizer-v1 shared models are not merged on this branch",
    )
    def test_a0_shared_models_materialize_all_c_side_wave0_payloads(self) -> None:
        fixture = organizer_wave0_mode_expectations()

        metadata = tuple(
            FrameMetadata.model_validate(payload)
            for payload in fixture.frame_metadata_payloads
        )
        candidates = tuple(
            FrameCandidate.model_validate(payload)
            for payload in fixture.frame_candidate_payloads
        )
        fused = tuple(
            FusedFrameCandidate.model_validate(payload)
            for payload in fixture.expected_fused_payloads
        )
        match = TRAKEFrameMatch(
            event_id="event-1",
            frame_id=metadata[1].frame_id,
            video_id=metadata[1].video_id,
            keyframe_no=metadata[1].keyframe_no,
            local_index=metadata[1].local_index,
            timestamp_sec=metadata[1].timestamp_sec,
            source_frame_idx=metadata[1].source_frame_idx,
            similarity_score=0.92,
        )
        image = ImageEvidence(
            evidence_id=f"image:{metadata[1].frame_id}",
            video_id=metadata[1].video_id,
            frame_id=metadata[1].frame_id,
            keyframe_no=metadata[1].keyframe_no,
            timestamp_sec=metadata[1].timestamp_sec,
            source_frame_idx=metadata[1].source_frame_idx,
            image_reference=f"fixture://organizer/{metadata[1].frame_id}",
        )

        self.assertEqual(candidates[1].source_frame_idx, 0)
        self.assertEqual(fused[0].keyframe_no, 2)
        self.assertEqual(match.local_index, 1)
        self.assertEqual(image.source_frame_idx, 0)

if __name__ == "__main__":
    unittest.main()
