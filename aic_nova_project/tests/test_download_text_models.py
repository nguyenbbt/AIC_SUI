from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "download_text_models.py"


def _load_module():
    spec = spec_from_file_location("download_text_models", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_download_model_uses_locked_snapshot_without_loading_weights(tmp_path):
    module = _load_module()

    with patch.object(module, "snapshot_download") as download:
        module.download_model(
            "example/model",
            str(tmp_path),
            "locked-revision",
        )

    download.assert_called_once_with(
        repo_id="example/model",
        revision="locked-revision",
        cache_dir=str(tmp_path),
    )
