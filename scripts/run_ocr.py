"""Command-line entry point for selective PDF OCR."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from digital_product_rag.ingestion.ocr import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(repository_root=REPOSITORY_ROOT))
