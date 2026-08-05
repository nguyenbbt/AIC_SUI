import pytest

from data_pipeline.shot_keyframe.cli import resolve_worker_count


@pytest.mark.parametrize(
    ("device", "cuda_available", "expected"),
    [
        ("cuda", False, 1),
        (None, True, 1),
        ("cpu", True, 4),
        (None, False, 4),
    ],
)
def test_resolve_worker_count_prevents_cuda_model_replication(
    device,
    cuda_available,
    expected,
):
    assert resolve_worker_count(4, device, cuda_available) == expected


def test_resolve_worker_count_rejects_non_positive_values():
    with pytest.raises(ValueError, match="workers"):
        resolve_worker_count(0, "cpu", False)
