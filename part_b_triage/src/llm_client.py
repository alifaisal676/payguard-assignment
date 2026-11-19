"""Provider-agnostic LLM client with validation retries, backoff and a safe fallback.

Providers (chosen by LLM_PROVIDER in .env): groq | gemini | ollama. All three expose an
OpenAI-compatible endpoint, so one code path serves them all.

Two separate retry layers:
1. Transport retries (tenacity): rate limits (429), timeouts, connection errors, 5xx.
   Exponential backoff with jitter, honouring the server's `retry-after` header.
   A *daily* quota error is not retried; it raises DailyLimitReached so the batch can stop
   cleanly and resume later.
2. Content retries (max MAX_ATTEMPTS): if the reply isn't valid JSON or fails the Pydantic
   schema, the validation error is sent back to the model and it gets another try.
After the last failure the client returns schema.fallback(): "Other / unclear", requires_human=True.
It never crashes the batch and never guesses.

Every API call is logged (tokens, latency, attempt, error) to data/llm_calls.jsonl.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import openai
from dotenv import load_dotenv
from pydantic import ValidationError
from tenacity import retry, retry_if_exception, stop_after_attempt

from part_b_triage.src.prompts import FIX_TEMPLATE, PROMPT_VERSION, build_messages
from part_b_triage.src.schema import Triage, fallback

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
CALL_LOG = ROOT / "part_b_triage" / "data" / "llm_calls.jsonl"

PROVIDERS = {
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY",
             "model_env": "GROQ_MODEL", "default_model": "openai/gpt-oss-120b"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "key_env": "GEMINI_API_KEY",
               "model_env": "GEMINI_MODEL", "default_model": "gemini-2.0-flash"},
    "ollama": {"base_url_env": "OLLAMA_BASE_URL", "base_url": "http://localhost:11434/v1", "key_env": None,
               "model_env": "OLLAMA_MODEL", "default_model": "llama3.1:8b"},
}
MAX_ATTEMPTS = 3          # content attempts (first try + 2 repairs)
MAX_TRANSPORT_TRIES = 8   # per API call
MAX_OUTPUT_TOKENS = 1024  # reasoning models spend part of this on hidden reasoning


class DailyLimitReached(RuntimeError):
    """Provider's per-day quota is exhausted: stop the batch, resume tomorrow."""


@dataclass
class TriageRecord:
    complaint_id: str
    provider: str
    model: str
    prompt_version: str
    triage: dict
    attempts: int = 0
    schema_failures: int = 0          # invalid replies that needed a repair attempt
    fallback: bool = False
    error: str | None = None
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw_replies: list = field(default_factory=list)


def _is_daily_limit(e: Exception) -> bool:
    msg = str(e).lower()
    return isinstance(e, openai.RateLimitError) and ("per day" in msg or "tpd" in msg or "rpd" in msg)


def _transient(e: BaseException) -> bool:
    if isinstance(e, Exception) and _is_daily_limit(e):
        return False
    return isinstance(e, (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError,
                          openai.InternalServerError))


def _wait(retry_state) -> float:
    """Honour retry-after if the server sent one, else exponential backoff with jitter."""
    e = retry_state.outcome.exception()
    headers = getattr(getattr(e, "response", None), "headers", None) or {}
    try:
        return float(headers.get("retry-after")) + random.uniform(0, 1)
    except (TypeError, ValueError):
        return min(60.0, 2 ** retry_state.attempt_number) + random.uniform(0, 1)


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None, reasoning_effort: str | None = "low"):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "groq")).lower()
        cfg = PROVIDERS[self.provider]
        self.model = model or os.getenv(cfg["model_env"]) or cfg["default_model"]
        base_url = os.getenv(cfg.get("base_url_env", ""), cfg["base_url"]) if cfg.get("base_url_env") else cfg["base_url"]
        api_key = os.getenv(cfg["key_env"]) if cfg["key_env"] else "ollama"
        if not api_key:
            raise RuntimeError(f"{cfg['key_env']} is not set (see .env.example)")
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=60, max_retries=0)
        # gpt-oss models are reasoning models; "low" keeps latency and token use down.
        self.extra = {"reasoning_effort": reasoning_effort} if reasoning_effort and "gpt-oss" in self.model else {}

    @retry(retry=retry_if_exception(_transient), wait=_wait, stop=stop_after_attempt(MAX_TRANSPORT_TRIES), reraise=True)
    def _call(self, messages):
        try:
            return self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0, max_completion_tokens=MAX_OUTPUT_TOKENS,
                response_format={"type": "json_object"}, extra_body=self.extra or None)
        except openai.RateLimitError as e:
            if _is_daily_limit(e):
                raise DailyLimitReached(str(e)[:300]) from e
            raise

    def _log(self, **row):
        CALL_LOG.parent.mkdir(parents=True, exist_ok=True)
        with CALL_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "provider": self.provider,
                                "model": self.model, **row}) + "\n")

    def triage(self, complaint_id: str, text: str) -> TriageRecord:
        rec = TriageRecord(complaint_id=complaint_id, provider=self.provider, model=self.model,
                           prompt_version=PROMPT_VERSION, triage={})
        messages = build_messages(text)
        last_error = None
        t_start = time.perf_counter()
        for attempt in range(1, MAX_ATTEMPTS + 1):
            rec.attempts = attempt
            t0 = time.perf_counter()
            try:
                resp = self._call(messages)
            except DailyLimitReached:
                raise
            except Exception as e:                       # transport failed even after retries
                last_error = f"api error: {e.__class__.__name__}: {str(e)[:200]}"
                self._log(complaint_id=complaint_id, attempt=attempt, latency_s=round(time.perf_counter() - t0, 3), error=last_error)
                break
            usage = resp.usage
            rec.prompt_tokens += usage.prompt_tokens if usage else 0
            rec.completion_tokens += usage.completion_tokens if usage else 0
            content = resp.choices[0].message.content or ""
            rec.raw_replies.append(content[:2000])
            try:
                parsed = Triage.model_validate_json(content)
                self._log(complaint_id=complaint_id, attempt=attempt, latency_s=round(time.perf_counter() - t0, 3),
                          prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens, error=None)
                rec.triage = parsed.model_dump()
                rec.latency_s = round(time.perf_counter() - t_start, 3)
                return rec
            except (ValidationError, ValueError) as e:
                rec.schema_failures += 1
                last_error = f"invalid output: {str(e)[:300]}"
                self._log(complaint_id=complaint_id, attempt=attempt, latency_s=round(time.perf_counter() - t0, 3),
                          prompt_tokens=usage.prompt_tokens if usage else 0,
                          completion_tokens=usage.completion_tokens if usage else 0, error=last_error)
                # Send the model its own answer and the exact validation error, then ask again.
                messages = messages + [{"role": "assistant", "content": content[:2000]},
                                       {"role": "user", "content": FIX_TEMPLATE.format(error=str(e)[:500])}]
        rec.fallback, rec.error = True, last_error
        rec.triage = fallback(last_error or "unknown").model_dump()
        rec.latency_s = round(time.perf_counter() - t_start, 3)
        return rec


def record_to_dict(rec: TriageRecord) -> dict:
    return asdict(rec)


if __name__ == "__main__":
    c = LLMClient()
    r = c.triage("demo", "Someone used my debit card to withdraw $900 at an ATM in another state. "
                         "The bank says I have to wait 10 days and I can't pay rent.")
    print(json.dumps(record_to_dict(r), indent=1)[:1500])
