from __future__ import annotations

import hashlib
import json
import os
import re
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

from google import genai
from google.genai import types

VAULT_VARIABLE = "GOOGLE_API_KEY_VAULT"
IGNORED_KEY_VARIABLES = ("GOOGLE_API_KEY", "GEMINI_API_KEY")
DEFAULT_STATE_FILE = Path(__file__).resolve().parents[1] / "runs" / ".key_state.json"
QUOTA_TIMEZONE = ZoneInfo("America/Los_Angeles")
RATE_LIMITED = 429
DAILY_QUOTA_MARKER = "PerDay"
FINGERPRINT_CHARS = 12
KEY_SEPARATORS = r"[\s,\[\]\"']+"


class DailyQuotaUsedUp(RuntimeError):
    pass


class AllKeysUsedUp(DailyQuotaUsedUp):
    pass


class NoApiKey(RuntimeError):
    pass


NO_KEY_MESSAGE = ("No Gemini API key. Put yours in .env as GOOGLE_API_KEY_VAULT='[AQ.your-key]' "
                  "(see .env.example). Get one free at https://aistudio.google.com/apikey")


def parse_vault(text: str | None) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        candidates = parsed if isinstance(parsed, list) else [parsed]
        candidates = [str(k) for k in candidates]
    except json.JSONDecodeError:
        candidates = re.split(KEY_SEPARATORS, text)
    return _unique(k.strip() for k in candidates)


def keys_from_env(env=None) -> list[str]:
    env = os.environ if env is None else env
    for name in IGNORED_KEY_VARIABLES:
        if env.get(name, "").strip():
            warnings.warn(f"{name} is ignored. Put your key in {VAULT_VARIABLE} instead, "
                          f"e.g. {VAULT_VARIABLE}='[AQ.your-key]'", stacklevel=2)
    return parse_vault(env.get(VAULT_VARIABLE))


def fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:FINGERPRINT_CHARS]


def next_quota_reset(now: datetime) -> datetime:
    local_now = now.astimezone(QUOTA_TIMEZONE)
    next_local_midnight = datetime.combine(local_now.date() + timedelta(days=1), datetime.min.time(),
                                           tzinfo=QUOTA_TIMEZONE)
    return next_local_midnight.astimezone(timezone.utc)


def is_daily_quota_error(exc: Exception) -> bool:
    return getattr(exc, "code", None) == RATE_LIMITED and DAILY_QUOTA_MARKER in str(exc)


class KeyPool:
    def __init__(self, keys: list[str], state_file: str | Path = DEFAULT_STATE_FILE,
                 clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        self.keys = list(keys)
        self.state_file = Path(state_file)
        self.clock = clock

    def __len__(self) -> int:
        return len(self.keys)

    def _load(self) -> dict[str, str]:
        try:
            return json.loads(self.state_file.read_text()).get("used_up_until", {})
        except (OSError, ValueError):
            return {}

    def _save(self, used_up_until: dict[str, str]) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps({"used_up_until": used_up_until}, indent=2))

    def _used_up(self) -> set[str]:
        now = self.clock()
        return {fp for fp, until in self._load().items() if datetime.fromisoformat(until) > now}

    def current(self) -> tuple[int, str] | None:
        used_up = self._used_up()
        for slot, key in enumerate(self.keys, 1):
            if fingerprint(key) not in used_up:
                return slot, key
        return None

    def mark_used_up(self, key: str) -> None:
        now = self.clock()
        state = {fp: until for fp, until in self._load().items() if datetime.fromisoformat(until) > now}
        state[fingerprint(key)] = next_quota_reset(now).isoformat()
        self._save(state)

    def describe(self, slot: int) -> str:
        return f"key slot {slot} of {len(self.keys)}"


class VaultClient:
    uses_real_api = True

    def __init__(self, pool: KeyPool, on_switch: Callable[[str], None] | None = None):
        if not len(pool):
            raise ValueError("The key pool is empty")
        self.pool = pool
        self.on_switch = on_switch or (lambda message: None)
        self._clients: dict[str, object] = {}
        self.models = _VaultModels(self)
        self.slot_in_use: int | None = None

    def _client(self, key: str):
        if key not in self._clients:
            self._clients[key] = genai.Client(api_key=key,
                                              http_options=types.HttpOptions(api_version="v1beta"))
        return self._clients[key]

    def _next(self) -> tuple[int, str]:
        found = self.pool.current()
        if found is None:
            raise AllKeysUsedUp(f"All {len(self.pool)} API keys have used up today's ER 2 quota "
                                "(it resets at midnight Pacific time).")
        return found

    def _used_up(self, slot: int, key: str) -> None:
        self.pool.mark_used_up(key)
        following = self.pool.current()
        message = f"{self.pool.describe(slot)} has used up today's quota"
        message += f"; switching to {self.pool.describe(following[0])}." if following else "."
        self.on_switch(message)


class _VaultModels:
    def __init__(self, vault: VaultClient):
        self.vault = vault

    def generate_content(self, **kwargs):
        while True:
            slot, key = self.vault._next()
            self.vault.slot_in_use = slot
            try:
                return self.vault._client(key).models.generate_content(**kwargs)
            except Exception as exc:
                if not is_daily_quota_error(exc):
                    raise
                self.vault._used_up(slot, key)

    def generate_content_stream(self, **kwargs):
        while True:
            slot, key = self.vault._next()
            self.vault.slot_in_use = slot
            try:
                stream = iter(self.vault._client(key).models.generate_content_stream(**kwargs))
                first_chunk = next(stream)
            except StopIteration:
                return iter(())
            except Exception as exc:
                if not is_daily_quota_error(exc):
                    raise
                self.vault._used_up(slot, key)
                continue
            return _prepend(first_chunk, stream)


def gemini_client() -> VaultClient:
    keys = keys_from_env()
    if not keys:
        raise NoApiKey(NO_KEY_MESSAGE)
    return VaultClient(KeyPool(keys))


def _prepend(first, rest):
    yield first
    yield from rest


def _unique(items) -> list[str]:
    seen, out = set(), []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out
