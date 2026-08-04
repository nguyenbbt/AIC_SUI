from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_module1_dockerfile_uses_project_root_build_context():
    dockerfile_path = (
        PROJECT_ROOT / "data_pipeline" / "shot_keyframe" / "Dockerfile"
    )
    requirements_path = dockerfile_path.with_name("requirements.txt")
    dockerfile = dockerfile_path.read_text(encoding="utf-8")

    assert requirements_path.is_file()
    assert (
        "COPY data_pipeline/shot_keyframe/requirements.txt "
        "/tmp/requirements.txt"
    ) in dockerfile
    assert (
        "COPY data_pipeline/shot_keyframe "
        "/app/data_pipeline/shot_keyframe"
    ) in dockerfile
    assert "COPY weights/" not in dockerfile


def test_module2_entrypoint_runs_cli_inside_its_package():
    dockerfile = (
        PROJECT_ROOT
        / "feature_extraction"
        / "visual_embedding"
        / "Dockerfile"
    ).read_text(encoding="utf-8")

    assert (
        "COPY feature_extraction/visual_embedding "
        "/app/feature_extraction/visual_embedding"
    ) in dockerfile
    assert (
        'ENTRYPOINT ["python", "-m", '
        '"feature_extraction.visual_embedding.cli"]'
    ) in dockerfile


def test_module3_entrypoint_runs_cli_inside_its_package():
    dockerfile = (
        PROJECT_ROOT
        / "feature_extraction"
        / "asr_transcript"
        / "Dockerfile"
    ).read_text(encoding="utf-8")

    assert (
        "COPY feature_extraction/asr_transcript "
        "/app/feature_extraction/asr_transcript"
    ) in dockerfile
    assert (
        'ENTRYPOINT ["python", "-m", '
        '"feature_extraction.asr_transcript.cli"]'
    ) in dockerfile


def test_module4_dockerfile_uses_project_root_build_context():
    dockerfile = (
        PROJECT_ROOT / "feature_extraction" / "ocr" / "Dockerfile"
    ).read_text(encoding="utf-8")

    assert (
        "COPY feature_extraction/ocr/requirements.txt "
        "/tmp/requirements.txt"
    ) in dockerfile
    assert "COPY feature_extraction/ocr/src /app/src" in dockerfile
    assert 'ENTRYPOINT ["python", "-m", "ocr_module.cli"]' in dockerfile
