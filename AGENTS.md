# Voicey agent guide

Read README.md and docs/INSTALL.md before changing installation behavior.
See docs/ARCHITECTURE.md and docs/OPERATIONS.md for design and rollback.
Never print `.env`, credentials, audio, transcripts or raw upstream errors.
Never commit data/, models/, runtimes/ or `.env`.

- Python 3.12+, supported target Linux x86_64 CPU; local checks Python 3.13.
- `uv sync --frozen`; `uv run ruff check .`; `uv run python -m pytest -q`; `actionlint`.
- Tests need no Telegram/API keys or models. Runtime versions/checksums are pinned.
- No PyTorch, Redis, Postgres, resident LLM daemon, webhook or public port needed.
- One worker owns ALL inference. Keep child cleanup in `finally`, including cancellation.
- Use subprocess argv, never shell interpolation for user/model inputs.
- Operator profiles reference environment keys. Users select profiles; no URLs/keys in chat.
- Vocabulary is a hint, scoped per user. Never train on or export private audio/history.
- Preserve SQLite data. Back up online with SQLite backup(), not by copying only a WAL-mode DB.
- main.py is the entry point; the historical MVP is preserved in Git history.
- Resolve the target, existing poller and data paths before deployment. One poller per token.
- Host swap changes are operator decisions, separate from starting the application.

## Access and this VPS

- Production: SSH alias `keykomi`, project `/opt/voicey`, Compose service `bot`.
  Read personal infrastructure instructions before SSH; preserve the other services.
- User management: [docs/ACCESS.md](docs/ACCESS.md). Ask the person to open this
  bot privately and send `/start` or `/id`. The bot reports the sender's numeric
  **user ID**, including when access is denied. Do not confuse it with a bot/chat
  ID, username or an ID from a different Telegram account.
- Add the confirmed ID to `ALLOWED_USER_IDS` in `/opt/voicey/.env`, retaining existing
  IDs; keep `PUBLIC_BOT=false`, mode 600. Recreate only `bot` after draining jobs;
  `docker compose restart` alone does not reload changed env-file values.
- Never put actual user IDs, token values or the production `.env` in public docs.
- Supported STT: voice/video_note/audio/video and media documents; JSON documents
  still import vocabulary. Keep original-message replies/status/buttons for all.
- ffmpeg is required for local AND API STT. Only extracted WAV audio goes to API.
  Retain size/duration checks, safe errors and temporary-file cleanup on failure.
