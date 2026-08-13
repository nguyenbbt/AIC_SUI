from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "online_modal_encoders.py"


def test_online_modal_encoder_is_private_cost_bounded_and_model_pinned() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert 'modal.App("aic-nova-online-encoders")' in source
    assert 'gpu="L4"' in source
    assert "max_containers=1" in source
    assert "scaledown_window=300" in source
    assert "@modal.web_endpoint" not in source
    assert 'OPENCLIP_MODEL_ID = "ViT-B-32::openai"' in source
    assert 'VIETNAMESE_MODEL_NAME = "dangvantuan/vietnamese-embedding"' in source
    assert (
        'VIETNAMESE_MODEL_REVISION = "4ab46e46ba5902328ba0742e489e75f787932f2b"'
        in source
    )
    assert 'MODAL_ENCODER_SCHEMA_VERSION = "aic-online-encoder-v1"' in source
    assert "create_if_missing=True" in source


def test_online_modal_encoder_bounds_untrusted_batch_input() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "MAX_BATCH_SIZE = 64" in source
    assert "MAX_TEXT_LENGTH = 4096" in source
    assert "normalize_embeddings=True" in source


def test_online_modal_sdk_is_pinned_to_the_verified_version() -> None:
    requirements = (
        PROJECT_ROOT / "online" / "requirements-modal.txt"
    ).read_text(encoding="utf-8").splitlines()

    assert requirements == ["modal==1.5.2"]
