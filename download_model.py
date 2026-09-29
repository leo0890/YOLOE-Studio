"""
Скрипт для скачивания весов YOLOE-26s-seg и text-encoder MobileCLIP2.
...
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("❌ huggingface_hub не установлен.")
    print("   Установи:  pip install huggingface_hub")
    sys.exit(1)


PROJECT_ROOT = Path(__file__).parent.resolve()


def download_file(repo_id: str, filename: str, target_name: str) -> Path:
    ...


def main() -> None:
    ...


if __name__ == "__main__":
    main()