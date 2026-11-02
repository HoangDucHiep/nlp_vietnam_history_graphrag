#!/usr/bin/env python
"""In ra **văn bản thô** của một verdict LLM judge đã cache.

`_probe_judge_row.py` chỉ in payload đã parse.  Nếu `score` bị gán sai (ví dụ
do trường JSON bị đảo thứ tự), ta chỉ thấy nó ở dạng thô.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

NEEDLE = "Đáp án đúng hoàn toàn so với đáp án tham chiếu và gold facts"

for path in glob.glob(str(REPO_ROOT / "artifacts/llm_cache/**/*.json"), recursive=True):
    try:
        data = json.loads(Path(path).read_text())
    except (ValueError, OSError):
        continue
    text = data.get("text", "") if isinstance(data, dict) else ""
    if NEEDLE in text:
        print("FILE:", path.replace(str(REPO_ROOT), "."))
        print("RAW TEXT >>>")
        print(text)
        print("<<< END")
        break
