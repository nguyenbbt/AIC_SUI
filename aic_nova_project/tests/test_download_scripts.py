from types import SimpleNamespace

import pytest

from scripts import download_object_detectors
from scripts import download_transnet_weights


def test_object_downloader_returns_failure_if_any_required_model_fails(
    monkeypatch,
):
    monkeypatch.setattr(
        download_object_detectors,
        "download_yolo_world",
        lambda: None,
    )

    def fail_codetr():
        raise RuntimeError("download failed")

    monkeypatch.setattr(
        download_object_detectors,
        "download_codetr",
        fail_codetr,
    )

    assert download_object_detectors.main() == 1


def test_transnet_download_rejects_missing_package_weights(
    tmp_path,
    monkeypatch,
):
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    monkeypatch.setitem(
        __import__("sys").modules,
        "transnetv2_pytorch",
        SimpleNamespace(__file__=str(package_dir / "__init__.py")),
    )

    with pytest.raises(FileNotFoundError):
        download_transnet_weights.download_weights(
            tmp_path / "weights" / "transnetv2-pytorch-weights.pth"
        )


def test_transnet_downloader_returns_failure_when_extraction_fails(
    monkeypatch,
):
    def fail_download():
        raise RuntimeError("missing weights")

    monkeypatch.setattr(
        download_transnet_weights,
        "download_weights",
        fail_download,
    )

    assert download_transnet_weights.main() == 1
