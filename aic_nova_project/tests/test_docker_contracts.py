from pathlib import Path

import yaml


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


def test_text_model_download_is_pinned_to_the_online_revision():
    dockerfile = (
        PROJECT_ROOT / "feature_extraction" / "text_embedding" / "Dockerfile"
    ).read_text(encoding="utf-8")

    assert "dangvantuan/vietnamese-embedding" in dockerfile
    assert (
        "--model-revision "
        "4ab46e46ba5902328ba0742e489e75f787932f2b"
    ) in dockerfile


def test_indexing_image_contains_the_full_contract_verifier():
    dockerfile = (PROJECT_ROOT / "indexing" / "Dockerfile").read_text(
        encoding="utf-8"
    )

    assert "COPY verify_frame_id_consistency.py /app/" in dockerfile


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


def test_compose_uses_configurable_host_bind_mounts():
    compose = yaml.safe_load(
        (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    )

    expected_targets = {
        "etcd": "/etcd",
        "minio": "/minio_data",
        "milvus-standalone": "/var/lib/milvus",
        "elasticsearch": "/usr/share/elasticsearch/data",
        "indexing": "/workspace/data",
    }
    for service_name, target in expected_targets.items():
        mounts = compose["services"][service_name]["volumes"]
        mount = next(item for item in mounts if item["target"] == target)
        assert mount["type"] == "bind"
        assert "AIC_LOCAL_DATA_ROOT" in mount["source"]

    assert "volumes" not in compose


def test_rollback_compose_maps_exact_legacy_external_volumes():
    rollback = yaml.safe_load(
        (PROJECT_ROOT / "docker-compose.rollback.yml").read_text(
            encoding="utf-8"
        )
    )

    expected = {
        "legacy_etcd_data": "aic_nova_project_etcd_data",
        "legacy_minio_data": "aic_nova_project_minio_data",
        "legacy_milvus_data": "aic_nova_project_milvus_data",
        "legacy_es_data": "aic_nova_project_es_data",
    }
    assert {
        name: config["name"] for name, config in rollback["volumes"].items()
    } == expected
    assert all(config["external"] is True for config in rollback["volumes"].values())


def test_green_compose_is_fully_isolated_from_the_blue_runtime():
    compose = yaml.safe_load(
        (PROJECT_ROOT / "docker-compose.green.yml").read_text(
            encoding="utf-8"
        )
    )

    services = compose["services"]
    assert set(services) == {
        "etcd",
        "minio",
        "milvus-standalone",
        "elasticsearch",
        "indexing",
    }
    blue_container_names = {
        "aic_nova_etcd",
        "aic_nova_minio",
        "aic_nova_milvus",
        "aic_nova_elasticsearch",
        "aic_nova_indexing",
    }
    green_container_names = {
        service["container_name"] for service in services.values()
    }
    assert green_container_names.isdisjoint(blue_container_names)
    assert all("_green_" in name for name in green_container_names)

    rendered = (PROJECT_ROOT / "docker-compose.green.yml").read_text(
        encoding="utf-8"
    )
    assert "AIC_GREEN_DATA_ROOT:?" in rendered
    assert "AIC_GREEN_BACKUP_ROOT:?" in rendered
    assert "${AIC_LOCAL_DATA_ROOT" not in rendered
    for blue_port in ('"9001:9001"', '"19530:19530"', '"9091:9091"', '"9200:9200"'):
        assert blue_port not in rendered

    indexing_mounts = services["indexing"]["volumes"]
    source_mounts = {
        mount["target"]: mount for mount in indexing_mounts
    }
    assert source_mounts["/workspace/data"]["type"] == "bind"
    for target in (
        "/workspace/data/processed/metadata",
        "/workspace/data/processed/keyframes",
        "/workspace/data/processed/embeddings/visual",
        "/workspace/data/processed/object_detection",
    ):
        assert source_mounts[target]["read_only"] is True

    for service_name in services:
        assert "mem_limit" in services[service_name]
        assert "cpus" in services[service_name]
