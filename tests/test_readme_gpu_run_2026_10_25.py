"""README states the current blob's hosted GPU run (2026-10-25 Colab T4 run of `b41a7f56`)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_does_not_deny_the_recorded_gpu_run():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "has not had a fresh GPU execution" not in text
    assert "`b41a7f56` ran its default path in one pass on a fresh Colab Tesla T4 on 2026-10-25" in text
