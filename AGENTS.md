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
