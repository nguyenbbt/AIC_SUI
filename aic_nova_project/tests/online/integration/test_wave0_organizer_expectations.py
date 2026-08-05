"""C-side consumer expectations for the self-indexed-v2 contract."""

from __future__ import annotations

import unittest
from collections.abc import Mapping
from pathlib import PurePosixPath

from online.domain.candidates import FrameCandidate, FusedFrameCandidate
from online.domain.identifiers import parse_canonical_frame_id
from online.domain.trake import TRAKEFrameMatch
from online.domain.vqa import ImageEvidence
from online.ports.records import FrameMetadata
from tests.online.fixtures.mode_fixtures import self_indexed_mode_expectations


class CSelfIndexedContractTests(unittest.TestCase):
    def test_cases_cover_duplicate_source_frame_and_safe_paths(self) -> None:
        metadata = self_indexed_mode_expectations().frame_metadata_payloads
        self.assertEqual(tuple(row["source_frame_idx"] for row in metadata[:2]), (0, 0))
        self.assertNotEqual(metadata[0]["frame_id"], metadata[1]["frame_id"])
        for row in metadata:
            identity = parse_canonical_frame_id(str(row["frame_id"]))
            self.assertEqual(identity.video_id, row["video_id"])
            self.assertEqual(identity.shot_id, row["shot_id"])
            path = PurePosixPath(str(row["image_rel_path"]))
            self.assertFalse(path.is_absolute())
            self.assertNotIn("..", path.parts)

    def test_asr_mapping_uses_timestamp(self) -> None:
        fixture = self_indexed_mode_expectations()
        interval = fixture.asr_interval_payload
        mapped = tuple(
            str(row["frame_id"])
            for row in fixture.frame_metadata_payloads
            if row["video_id"] == interval["video_id"]
            and interval["start_time_sec"] <= row["timestamp_sec"] <= interval["end_time_sec"]
        )
        self.assertEqual(mapped, fixture.expected_asr_mapped_frame_ids)

    def test_kis_dedup_uses_competition_frame_identity(self) -> None:
        fixture = self_indexed_mode_expectations()
        kept: list[Mapping[str, object]] = []
        seen: set[tuple[object, object]] = set()
        for row in fixture.expected_fused_payloads:
            key = (row["video_id"], row["source_frame_idx"])
            if key not in seen:
                seen.add(key)
                kept.append(row)
        self.assertEqual(tuple(str(row["frame_id"]) for row in kept), fixture.expected_kis_frame_ids_after_dedup)
        self.assertEqual(tuple((str(row["video_id"]), int(row["source_frame_idx"])) for row in kept), fixture.expected_kis_competition_rows)

    def test_shared_models_materialize_all_payloads(self) -> None:
        fixture = self_indexed_mode_expectations()
        metadata = tuple(FrameMetadata.model_validate(row) for row in fixture.frame_metadata_payloads)
        candidates = tuple(FrameCandidate.model_validate(row) for row in fixture.frame_candidate_payloads)
        fused = tuple(FusedFrameCandidate.model_validate(row) for row in fixture.expected_fused_payloads)
        match = TRAKEFrameMatch(
            event_id="event-1",
            frame_id=metadata[1].frame_id,
            video_id=metadata[1].video_id,
            shot_id=metadata[1].shot_id,
            local_index=1,
            timestamp_sec=metadata[1].timestamp_sec,
            source_frame_idx=metadata[1].source_frame_idx,
            image_rel_path=metadata[1].image_rel_path,
            similarity_score=0.92,
        )
        image = ImageEvidence(
            evidence_id=f"image:{metadata[1].frame_id}",
            video_id=metadata[1].video_id,
            frame_id=metadata[1].frame_id,
            shot_id=metadata[1].shot_id,
            timestamp_sec=metadata[1].timestamp_sec,
            source_frame_idx=metadata[1].source_frame_idx,
            image_reference=f"fixture://self-indexed/{metadata[1].frame_id}",
        )
        self.assertEqual(candidates[1].source_frame_idx, 0)
        self.assertEqual(fused[0].shot_id, 0)
        self.assertEqual(match.local_index, 1)
        self.assertEqual(image.source_frame_idx, 0)


if __name__ == "__main__":
    unittest.main()
