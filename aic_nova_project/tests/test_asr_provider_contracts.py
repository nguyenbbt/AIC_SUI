from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_asr_provider_dependencies_and_environment_template_are_complete():
    requirements = (
        PROJECT_ROOT
        / "feature_extraction"
        / "asr_transcript"
        / "requirements.txt"
    ).read_text(encoding="utf-8")
    environment_template = (PROJECT_ROOT / ".env.example").read_text(
        encoding="utf-8"
    )

    requirement_names = {
        line.split("=", 1)[0].split(">", 1)[0].strip().lower()
        for line in requirements.splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert "openai" in requirement_names
    assert "accelerate" in requirement_names

    for variable_name in (
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "BASE_URL",
        "API_VERSION",
    ):
        assert f"{variable_name}=" in environment_template
