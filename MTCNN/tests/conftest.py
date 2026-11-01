"""Make repository-root imports work when running MTCNN/tests directly."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: MTCNN inference with bundled pretrained checkpoints and sample image",
    )
