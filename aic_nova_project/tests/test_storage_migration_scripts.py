from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_legacy_rollback_smoke_preserves_named_volumes():
    script = (PROJECT_ROOT / "scripts" / "test_legacy_rollback.ps1").read_text(
        encoding="utf-8"
    )

    assert "docker-compose.rollback.yml" in script
    assert '"compose", "down", "--remove-orphans"' in script
    assert '"compose", "down", "-v"' not in script
    assert "online.validate_contract" in script
    assert "--fail-on-partial" in script
    for volume in (
        "aic_nova_project_etcd_data",
        "aic_nova_project_minio_data",
        "aic_nova_project_milvus_data",
        "aic_nova_project_es_data",
    ):
        assert volume in script


def test_candidate_promotion_revalidates_and_rolls_back_on_failure():
    script = (PROJECT_ROOT / "scripts" / "promote_btc_candidate.ps1").read_text(
        encoding="utf-8"
    )

    assert '"scripts.btc_storage_manager", "promote"' in script
    assert '"scripts.btc_storage_manager", "rollback"' in script
    assert "docker-compose.rollback.yml" in script
    assert "online.validate_contract" in script
    assert '"compose", "down", "--remove-orphans"' in script
    assert '"compose", "down", "-v"' not in script
    assert script.index('"scripts.btc_storage_manager", "promote"') < script.index(
        "online.validate_contract"
    )


def test_cleanup_is_explicit_and_never_uses_compose_down_v():
    script = (PROJECT_ROOT / "scripts" / "cleanup_legacy_test_data.ps1").read_text(
        encoding="utf-8"
    )

    assert "ConfirmDestructiveCleanup" in script
    assert "ExpectedTestVideoIds" in script
    assert "AcceptSliceReplacement" in script
    assert '"volume", "rm", $volume' in script
    assert '"compose", "down", "-v"' not in script
    assert "docker image prune" not in script
    assert "docker builder prune" not in script
    for volume in (
        "aic_nova_project_etcd_data",
        "aic_nova_project_minio_data",
        "aic_nova_project_milvus_data",
        "aic_nova_project_es_data",
    ):
        assert volume in script
