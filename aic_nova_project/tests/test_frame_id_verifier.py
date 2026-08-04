from verify_frame_id_consistency import (
    VerificationSnapshot,
    build_consistency_report,
)


def test_verifier_compares_the_same_cross_db_record():
    snapshot = VerificationSnapshot(
        visual_frame_ids={"V001_00000_015"},
        ocr_vector_frame_ids=set(),
        ocr_text_frame_ids=set(),
        metadata_frame_ids={"V001_00000_050"},
        object_frame_ids=set(),
        asr_vector_ids=set(),
        asr_text_ids=set(),
        summary_vector_ids=set(),
        summary_text_ids=set(),
    )

    report = build_consistency_report(snapshot)

    assert any("visual" in error and "metadata" in error for error in report)


def test_verifier_accepts_joinable_records_across_all_backends():
    frame_id = "V001_00000_015"
    snapshot = VerificationSnapshot(
        visual_frame_ids={frame_id},
        ocr_vector_frame_ids={frame_id},
        ocr_text_frame_ids={frame_id},
        metadata_frame_ids={frame_id},
        object_frame_ids={frame_id},
        asr_vector_ids={("V001", "0")},
        asr_text_ids={("V001", "0")},
        summary_vector_ids={"V001"},
        summary_text_ids={"V001"},
    )

    assert build_consistency_report(snapshot) == []
