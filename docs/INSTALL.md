# Install Voicey — guide for a person or AI agent

Copy/fork this repository into another project if desired; preserve LICENSE.
Read target host instructions. Resolve the target and existing services; preserve
unrelated changes. Never request tokens in chat or print request URLs containing them.

1. Check Git state, `.env` key names, existing pollers, RAM/disk/CPU, Docker and
   architecture. Verify bot identity with getMe without logging the token. Inspect
   getWebhookInfo; do not delete a live webhook automatically. One poller per token.
2. If `.env` is absent, copy `.env.example`; otherwise preserve it. Supply BOT_TOKEN
   (legacy API_TOKEN accepted) and numeric ALLOWED_USER_IDS, comma-separated.
   PUBLIC_BOT=true is an explicit public opt-in. Protect `.env` with mode 600.
3. Set profiles as below. Defaults are first available profiles; per-user selections
   persist in SQLite. Unavailable selections use the first configured profile of
   that kind. `/settings` shows the current choice.
4. Run README checks. For local models run `python3 scripts/download_models.py`;
   downloads are commit-pinned and SHA256-verified, existing matching files reused.
5. `sudo python3 scripts/prepare_data.py`. Models must be readable by UID
   10001. For rootless Docker resolve UID mapping; do not use world-writable data.
6. On Linux x86_64, `docker compose build`, then `docker compose up -d`. Inspect
   config without dumping expanded secrets. Adds one container, no published port.
7. Check logs, `/start`, `/settings`, one voice and each text button. Confirm full
   text/reply target, denied users, child PIDs gone and memory released after jobs.
8. Record deployed revision/image, checksums, measured memory/latency and rollback.
   Never mark untested API providers or local model paths as verified.

## Local or API

Default config/profiles.json uses Whisper Turbo Q5_0 and Bonsai 2 2B PQ2_0, mounted
at /models. The Docker runtime is Linux x86_64. For native Mac/ARM, supply native
whisper-cli, compatible PrismML llama-server, ffmpeg and model paths. Set
WHISPER_BIN/LLAMA_BIN; run `uv sync --frozen --no-dev`, `uv run python main.py`.
Do not try the included x86_64 Bonsai binary as an ARM executable.

For API, copy config/profiles.api.example.json to config/profiles.local.json,
replace placeholder hosts/model IDs and set PROFILES_FILE=config/profiles.local.json
in `.env`. Each profile needs a unique ID, kind (stt/llm), backend (api), base_url,
model and key_env. Add the named credential to `.env`, restart the container.
Remove local profiles for API-only operation; no model downloads then needed.
Native Python API-only operation does not need runtime binaries or ffmpeg.

OpenRouter base_url: https://openrouter.ai/api/v1; key_env: OPENROUTER_API_KEY;
model: exact provider/model ID from your account. STT uses a separate multipart
`{base_url}/audio/transcriptions` endpoint with file/model/response_format and
optional prompt/language. Check provider compatibility. LLM uses messages/model/
temperature/max_tokens at `{base_url}/chat/completions`. API compatibility is not
universal; test the exact provider before relying on it.

No paid call happens on startup. Users choose configured profiles in `/settings`.
Keys belong to the instance operator and stay on the server. Telegram users cannot
upload keys or set endpoints. For a friend's instance provide this repository/guide,
a separate bot token, fresh data directory and their credentials. Never share your
`.env` or SQLite file. UI explains that API sends audio/text to the chosen provider.

## Small VPS

Two requests/day suit cold-loading models. One vCPU can take minutes per voice;
keep concurrency one. For a 2 GB host, 2 GiB disk swap and optional zswap at 10% RAM
are a starting reserve after checking disk/kernel support. Compression uses CPU,
not GPU. Do not automatically combine zram/zswap. Host changes are separate from
app install; preserve existing SSH/VPN and service volumes.

For Whisper CPU tuning, set WHISPER_NATIVE=ON before building on the actual target
host. This enables the target CPU's SIMD features. The resulting image is tied to
that CPU feature set: rebuild it on another host or use the default OFF for a more
portable x86_64 build. Benchmark both if latency matters; do not promise a speedup.
