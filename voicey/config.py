import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv


@dataclass(frozen=True)
class Profile:
    id: str
    label: str
    kind: str
    backend: str
    model: str
    base_url: str = ""
    key_env: str = ""

    def ready(self) -> bool:
        if self.backend == "local":
            return Path(self.model).is_file()
        return bool(os.environ.get(self.key_env))


@dataclass
class Settings:
    token: str = field(repr=False)
    allowed_ids: frozenset[int] = frozenset()
    public: bool = False
    data_dir: Path = Path("data")
    profiles: dict[str, Profile] = field(default_factory=dict)
    whisper_bin: str = "/opt/runtimes/whisper/bin/whisper-cli"
    llama_bin: str = "/opt/runtimes/llama/llama-server"
    max_seconds: int = 600
    max_bytes: int = 19_000_000
    timeout: int = 900
    retention_days: int = 7
    max_pending: int = 12

    def available(self, kind: str) -> list[Profile]:
        return [p for p in self.profiles.values() if p.kind == kind and p.ready()]


def load_settings() -> Settings:
    load_dotenv(Path(".env"))
    token = os.getenv("BOT_TOKEN") or os.getenv("API_TOKEN", "")
    if not token:
        raise ValueError("Set BOT_TOKEN (legacy API_TOKEN also supported)")
    ids = frozenset(int(x.strip()) for x in os.getenv("ALLOWED_USER_IDS", "").split(",") if x.strip())
    public = os.getenv("PUBLIC_BOT", "false").lower() == "true"
    if not ids and not public:
        raise ValueError("Set ALLOWED_USER_IDS or explicitly PUBLIC_BOT=true")
    raw = json.loads(Path(os.getenv("PROFILES_FILE", "config/profiles.json")).read_text())
    profiles = {}
    for item in raw:
        p = Profile(**item)
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,24}", p.id) or p.id in profiles:
            raise ValueError("Invalid or duplicate profile ID")
        if p.kind not in {"stt", "llm"} or p.backend not in {"local", "api"}:
            raise ValueError("Unknown profile kind/backend")
        if p.backend == "api":
            url = urlsplit(p.base_url)
            if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
                raise ValueError("API base_url must be HTTPS without credentials/query")
            if not p.key_env or not p.model:
                raise ValueError("API profiles require model and key_env")
        profiles[p.id] = p
    return Settings(
        token=token,
        allowed_ids=ids,
        public=public,
        profiles=profiles,
        data_dir=Path(os.getenv("DATA_DIR", "data")),
        whisper_bin=os.getenv("WHISPER_BIN", "/opt/runtimes/whisper/bin/whisper-cli"),
        llama_bin=os.getenv("LLAMA_BIN", "/opt/runtimes/llama/llama-server"),
        max_seconds=int(os.getenv("MAX_AUDIO_SECONDS", "600")),
        timeout=int(os.getenv("JOB_TIMEOUT_SECONDS", "900")),
        retention_days=int(os.getenv("RETENTION_DAYS", "7")),
    )
