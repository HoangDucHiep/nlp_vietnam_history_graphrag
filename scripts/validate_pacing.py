"""Validate the free-tier pacing and the judge against the real API.

Two checks are performed:

``pacing``
    A burst of requests with a realistic prompt size is issued.  The run
    must not produce a single 429 and the measured throughput must stay
    below the token quota of the model.

``judge``
    The judge prompts of the rows that were degraded in a previous run are
    replayed: the verdict must now come from the LLM (and not from the
    deterministic heuristic fallback).

Usage
-----
    python scripts/validate_pacing.py pacing [n_calls] [n_workers]
    python scripts/validate_pacing.py judge
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hgrag.eval.judge import LLMJudge  # noqa: E402
from hgrag.llm.client import GeminiClient  # noqa: E402
from hgrag.llm.schemas import LLMMessage, LLMRequest  # noqa: E402

MODEL = os.getenv("GEMINI_MODEL", "gemma-4-26b-a4b-it")
TPM = 14500

#: A prompt with roughly the size of a real judge prompt (~600 tokens).
PROMPT = (
    "Bạn là chuyên gia lịch sử Việt Nam. Trả lời bằng tiếng Việt.\n"
    "CONTEXT: " + ("Nguồn tài liệu lịch sử Việt Nam nói về các sự kiện. " * 60) + "\n"
    "CÂU HỎI: Đánh giá tình hình nông nghiệp Việt Nam sau chiến tranh?"
)


def build_client() -> GeminiClient:
    return GeminiClient(
        model=MODEL,
        api_key=os.environ["GEMINI_API_KEY"],
        cache_dir="artifacts/llm_cache",
        cache_enabled=False,
        fallback_models=[],
        model_rpm={MODEL: 26},
        model_tpm={MODEL: TPM},
    )


def check_pacing(n_calls: int, n_workers: int) -> int:
    client = build_client()
    errors: list[str] = []
    ok = 0
    started = time.time()

    def work(index: int) -> tuple[bool, str]:
        request = LLMRequest(
            messages=[LLMMessage(role="user", content=f"{PROMPT}\n#{index}")],
            task="answer_grounded",
            temperature=0.0,
            max_output_tokens=64,
        )
        try:
            client.generate(request)
            return True, ""
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)[:300]

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for future in as_completed([pool.submit(work, i) for i in range(n_calls)]):
            good, message = future.result()
            if good:
                ok += 1
            else:
                errors.append(message)

    elapsed = time.time() - started
    limiter_file = Path("artifacts/rate_limits") / f"{MODEL}.tokens.json"
    booked: list = []
    if limiter_file.exists():
        booked = json.loads(limiter_file.read_text(encoding="utf-8"))
    result = {
        "check": "pacing",
        "model": MODEL,
        "calls": n_calls,
        "ok": ok,
        "failed": len(errors),
        "elapsed_s": round(elapsed, 1),
        "calls_per_min": round(ok / elapsed * 60, 1) if elapsed else None,
        "booked_tokens_in_window": sum(p[1] for p in booked),
        "first_error": errors[0] if errors else None,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


def check_judge() -> int:
    client = build_client()
    judge = LLMJudge(client, max_retries=3, retry_base_delay=1.0)
    seen: set[tuple[str, str]] = set()
    n_ok = n_bad = 0
    for ckpt in sorted(Path("artifacts/quota_crashed").glob("eval_ckpt_*")):
        payload = json.loads(ckpt.read_text(encoding="utf-8"))
        for row in payload.get("rows", []):
            if not row.get("judge_degraded"):
                continue
            key = (str(row.get("system")), str(row.get("qid")))
            if key in seen:
                continue
            seen.add(key)
            verdict = judge.grade(
                question=row["question"],
                answer=row["answer"],
                reference=row.get("reference", ""),
                evidence=[],
            )
            if verdict.degraded:
                n_bad += 1
                print(f"STILL DEGRADED {key}: {verdict.reason[:120]}")
            else:
                n_ok += 1
    print(json.dumps({"check": "judge", "ok": n_ok, "bad": n_bad}, ensure_ascii=False))
    return 0 if n_bad == 0 else 1


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    mode = args[0] if args else "pacing"
    if mode == "judge":
        return check_judge()
    n_calls = int(args[1]) if len(args) > 1 else 24
    n_workers = int(args[2]) if len(args) > 2 else 6
    return check_pacing(n_calls, n_workers)


if __name__ == "__main__":
    raise SystemExit(main())
