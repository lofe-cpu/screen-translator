import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ocr import RapidBackend

for lang in ("auto", "ru"):
    RapidBackend(lang, gpu=False).warmup()
    print("модель готова:", lang)
