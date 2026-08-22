import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts.btc_storage_manager import (
    audit_backup_online_artifacts,
    build_backup_online_layout,
    main,
)


VIDEO_IDS = ("L21_V001", "L21_V002")


def _write_artifact(path: Path, payload: str = "artifact") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _build_complete_backup(root: Path, videos_root: Path) -> None:
    for video_id in VIDEO_IDS:
        _write_artifact(
            root / "metadata" / f"{video_id}.json",
            json.dumps({"video_id": video_id}),
        )
        _write_artifact(root / "keyframes" / video_id / "frame.webp")
        _write_artifact(root / "embeddings" / "visual" / f"{video_id}.parquet")
        _write_artifact(
            root
            / "asr-final"
            / "transcripts"
            / "transcripts"
            / f"{video_id}_cleaned.json",
            json.dumps({"video_id": video_id, "intervals": []}),
        )
        _write_artifact(
            root
            / "asr-final"
            / "transcripts"
            / "transcripts"
            / f"{video_id}_raw.json",
            json.dumps({"video_id": video_id, "segments": []}),
        )
        _write_artifact(
            root
            / "asr-final"
            / "summaries"
            / "summaries"
            / f"{video_id}.json",
            json.dumps({"video_id": video_id, "summary": "Tóm tắt"}),
        )
        _write_artifact(
            root / "ocr" / f"{video_id}.json",
            json.dumps({"video_id": video_id, "frames": []}),
        )
        for embedding_type in ("text_asr", "text_ocr", "text_summary"):
            _write_artifact(
                root / "module6-local" / embedding_type / f"{video_id}.parquet"
            )
        _write_artifact(
            videos_root / "Videos_L21_a" / "video" / f"{video_id}.mp4"
        )


def _write_objects(root: Path) -> None:
    for video_id in VIDEO_IDS:
        _write_artifact(
            root / "object_detection" / f"{video_id}.json",
            json.dumps({"video_id": video_id, "frames": []}),
        )


def test_backup_layout_maps_existing_artifacts_without_copying(tmp_path):
    backup_root = tmp_path / "modal_backup_safe"
    videos_root = tmp_path / "raw_videos"

    layout = build_backup_online_layout(backup_root, videos_root)

    assert layout["processed/metadata"] == backup_root / "metadata"
    assert layout["processed/keyframes"] == backup_root / "keyframes"
    assert layout["processed/videos"] == videos_root
    assert layout["processed/embeddings/visual"] == (
        backup_root / "embeddings" / "visual"
    )
    assert layout["processed/embeddings/text_asr"] == (
        backup_root / "module6-local" / "text_asr"
    )
    assert layout["processed/transcripts"] == (
        backup_root / "asr-final" / "transcripts" / "transcripts"
    )
    assert layout["processed/summaries"] == (
        backup_root / "asr-final" / "summaries" / "summaries"
    )
    assert layout["processed/object_detection"] == (
        backup_root / "object_detection"
    )


def test_backup_audit_allows_preparation_before_objects_finish(tmp_path):
    backup_root = tmp_path / "modal_backup_safe"
    videos_root = tmp_path / "raw_videos"
    _build_complete_backup(backup_root, videos_root)

    audit = audit_backup_online_artifacts(
        backup_root,
        videos_root,
        expected_video_count=2,
        expected_keyframe_count=2,
        require_objects=False,
    )

    assert audit["artifact_counts"]["metadata"] == 2
    assert audit["artifact_counts"]["transcripts"] == 2
    assert audit["artifact_counts"]["keyframes"] == 2
    assert audit["artifact_counts"]["objects"] == 0
    assert audit["ready_for_indexing"] is False


def test_backup_audit_opens_indexing_gate_only_for_valid_complete_objects(tmp_path):
    backup_root = tmp_path / "modal_backup_safe"
    videos_root = tmp_path / "raw_videos"
    _build_complete_backup(backup_root, videos_root)
    _write_objects(backup_root)

    audit = audit_backup_online_artifacts(
        backup_root,
        videos_root,
        expected_video_count=2,
        expected_keyframe_count=2,
        require_objects=True,
    )

    assert audit["artifact_counts"]["objects"] == 2
    assert audit["ready_for_indexing"] is True
    assert audit["video_ids"] == list(VIDEO_IDS)


def test_backup_audit_rejects_cross_module_video_id_mismatch(tmp_path):
    backup_root = tmp_path / "modal_backup_safe"
    videos_root = tmp_path / "raw_videos"
    _build_complete_backup(backup_root, videos_root)
    (backup_root / "module6-local" / "text_ocr" / "L21_V002.parquet").unlink()

    with pytest.raises(ValueError, match="text_ocr.*missing.*L21_V002"):
        audit_backup_online_artifacts(
            backup_root,
            videos_root,
            expected_video_count=2,
            expected_keyframe_count=2,
            require_objects=False,
        )


def test_backup_audit_rejects_invalid_object_payload(tmp_path):
    backup_root = tmp_path / "modal_backup_safe"
    videos_root = tmp_path / "raw_videos"
    _build_complete_backup(backup_root, videos_root)
    _write_objects(backup_root)
    _write_artifact(
        backup_root / "object_detection" / "L21_V002.json",
        json.dumps({"video_id": "WRONG", "frames": []}),
    )

    with pytest.raises(ValueError, match="Object Detection video_id mismatch"):
        audit_backup_online_artifacts(
            backup_root,
            videos_root,
            expected_video_count=2,
            expected_keyframe_count=2,
            require_objects=True,
        )


def test_backup_audit_rejects_duplicate_raw_video_basename(tmp_path):
    backup_root = tmp_path / "modal_backup_safe"
    videos_root = tmp_path / "raw_videos"
    _build_complete_backup(backup_root, videos_root)
    _write_artifact(
        videos_root / "Videos_L21_b" / "video" / "L21_V001.mp4"
    )

    with pytest.raises(ValueError, match="Duplicate raw video_id L21_V001"):
        audit_backup_online_artifacts(
            backup_root,
            videos_root,
            expected_video_count=2,
            expected_keyframe_count=2,
            require_objects=False,
        )


def test_backup_audit_cli_writes_a_junction_plan_without_objects(
    tmp_path,
    monkeypatch,
):
    backup_root = tmp_path / "modal_backup_safe"
    videos_root = tmp_path / "raw_videos"
    output = tmp_path / "audit.json"
    _build_complete_backup(backup_root, videos_root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "btc_storage_manager",
            "audit-backup",
            "--backup-root",
            str(backup_root),
            "--videos-root",
            str(videos_root),
            "--expected-videos",
            "2",
            "--expected-keyframes",
            "2",
            "--output",
            str(output),
        ],
    )

    assert main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ready_for_indexing"] is False
    assert "processed/metadata" in payload["links"]
    assert "processed/object_detection" not in payload["links"]


def test_dotenv_storage_update_preserves_credentials(tmp_path):
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "SECRET_TOKEN=keep-me\nAIC_LOCAL_DATA_ROOT=old-root\nOTHER=value\n",
        encoding="utf-8",
    )
    storage_script = Path(__file__).resolve().parents[1] / "scripts" / "storage_paths.ps1"
    command = (
        f". '{storage_script}'; "
        f"Set-AicDotEnvValue -Path '{env_file}' "
        "-Name 'AIC_LOCAL_DATA_ROOT' -Value 'E:\\DATA AIC\\modal_backup_safe'"
    )

    result = subprocess.run(
        [powershell, "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    content = env_file.read_text(encoding="utf-8")
    assert "SECRET_TOKEN=keep-me" in content
    assert "OTHER=value" in content
    assert content.count("AIC_LOCAL_DATA_ROOT=") == 1
    assert "AIC_LOCAL_DATA_ROOT=E:\\DATA AIC\\modal_backup_safe" in content


def test_prepare_script_uses_junctions_without_destructive_fallbacks():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "prepare_backup_online_root.ps1"
    ).read_text(encoding="utf-8")

    assert '"audit-backup"' in script
    assert 'New-Item -ItemType Junction' in script
    assert "Set-AicDotEnvValue" in script
    assert "RequireObjects" in script
    assert "processed/object_detection" in script
    assert "Remove-Item" not in script
    assert "Copy-Item" not in script
