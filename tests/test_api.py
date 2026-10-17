import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import aiohttp
import pytest
from aiohttp import web

from voicey.bot import App
from voicey.config import Profile, Settings
from voicey.db import Store
from voicey.inference import InferenceError, chat, transcribe


@asynccontextmanager
async def server(handler, path):
    app = web.Application()
    app.router.add_post(path, handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


async def test_chat_protocol_and_truncation(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-only")

    async def handle(request):
        assert request.headers["Authorization"] == "Bearer test-only"
        data = await request.json()
        assert data["messages"][0]["role"] == "system"
        assert "исходный текст" in data["messages"][1]["content"]
        assert data["model"] == "provider/model"
        return web.json_response({"choices": [{"message": {"content": "готово"}, "finish_reason": "length"}]})

    p = Profile("api", "API", "llm", "api", "provider/model", key_env="TEST_API_KEY")
    async with server(handle, "/chat/completions") as url, aiohttp.ClientSession() as http:
        result = await chat(http, url + "/chat/completions", p, "исходный текст", "summary")
    assert result.startswith("готово") and "ограничен" in result


async def test_stt_multipart_protocol(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-only")
    audio = tmp_path / "voice.ogg"
    audio.write_bytes(b"fake-audio")

    async def handle(request):
        reader = await request.multipart()
        fields = {}
        while part := await reader.next():
            fields[part.name] = bytes(await part.read())
        assert fields["file"] == b"fake-audio"
        assert fields["model"] == b"whisper"
        assert fields["language"] == b"ru"
        assert fields["prompt"] == b"GitHub"
        return web.json_response({"text": "текст GitHub"})

    async with server(handle, "/audio/transcriptions") as url:
        p = Profile("api", "API", "stt", "api", "whisper", url, "TEST_API_KEY")
        assert await transcribe(Settings(token="hidden"), p, audio, "GitHub", "ru") == "текст GitHub"


async def test_api_error_does_not_echo_upstream_body(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-only")

    async def handle(request):
        return web.Response(status=401, text="secret upstream detail")

    p = Profile("api", "API", "llm", "api", "model", key_env="TEST_API_KEY")
    async with server(handle, "/chat") as url, aiohttp.ClientSession() as http:
        with pytest.raises(InferenceError) as err:
            await chat(http, url + "/chat", p, "text", "summary")
    assert "secret" not in str(err.value) and "401" in str(err.value)


async def test_worker_serializes_stt_and_text_jobs(tmp_path):
    store = Store(tmp_path / "state.sqlite3")
    store.enqueue(1, 1, 1, "stt", {})
    store.enqueue(1, 1, 2, "summary", {})
    bot = AsyncMock()
    app = App(Settings(token="hidden", data_dir=tmp_path), store, bot)
    active, peak, seen = 0, 0, []

    async def execute(job, progress=None):
        nonlocal active, peak
        active += 1
        peak = max(active, peak)
        seen.append(job["kind"])
        await asyncio.sleep(0.01)
        active -= 1
        return "result"

    app.execute = execute
    task = asyncio.create_task(app.worker())
    try:
        async with asyncio.timeout(2):
            while store.db.execute("SELECT count(*) FROM jobs WHERE status='done'").fetchone()[0] < 2:
                await asyncio.sleep(0.01)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        store.close()
    assert seen == ["stt", "summary"] and peak == 1 and active == 0
