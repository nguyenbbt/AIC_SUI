from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_indexing_client_stays_compatible_with_elasticsearch_8():
    requirements = (
        PROJECT_ROOT / "indexing" / "requirements.txt"
    ).read_text(encoding="utf-8")

    assert "elasticsearch>=8.0.0,<9.0.0" in requirements
