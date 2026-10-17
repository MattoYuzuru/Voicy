import asyncio
import contextlib
import json
import shutil
import subprocess
import wave
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from aiogram.methods import SendMessage
from aiogram.types import Chat, Message, Update

from voicey.bot import App, build_dispatcher
from voicey.config import Profile, Settings
from voicey.db import Store
from voicey.inference import InferenceError, prepare_audio, transcribe


@pytest.mark.parametrize("kind,extra", [
    ("voice", {"duration": 2}),
    ("video_note", {"duration": 2, "length": 240}),
    ("audio", {"duration": 2}),
    ("video", {"duration": 2, "width": 240, "height": 240}),
    ("document", {"file_name": "../../recording.MP3"}),
    ("document", {"mime_type": "audio/flac"}),
    ("document", {"mime_type": "video/mp4"}),
])
async def test_media_dispatcher_queue_download_reply_cleanup(tmp_path, monkeypatch, kind, extra):
    bot = Bot("12345:test_only")
    calls = []

    async def api(_bot, method, **kwargs):
        calls.append(method)
        return Message(message_id=99, date=0, chat=Chat(id=1, type="private")).as_(bot)

    bot.session = AsyncMock(side_effect=api)
    bot.get_file = AsyncMock(return_value=type("File", (), {"file_path": "media", "file_size": 20})())

    async def download(path, destination):
        destination.write_bytes(b"downloaded media")

    bot.download_file = AsyncMock(side_effect=download)
    model = tmp_path / "model"
    model.touch()
    profile = Profile("local", "Local", "stt", "local", str(model))
    store = Store(tmp_path / "state.sqlite3")
    app = App(Settings(token="unused", allowed_ids=frozenset({1}), data_dir=tmp_path,
                       profiles={"local": profile}), store, bot)
    seen = []

    async def inference(s, p, path, hint, language, progress):
        assert path.name == "input.bin" and path.read_bytes() == b"downloaded media"
        seen.append(path)
        progress("transcribe", 50)
        return "Recognized speech"

    monkeypatch.setattr("voicey.bot.transcribe", inference)
    msg = {"message_id": 42, "date": 0, "chat": {"id": 1, "type": "private"},
           "from": {"id": 1, "is_bot": False, "first_name": "Test"},
           kind: {"file_id": "media-id", "file_unique_id": "unique", **extra}}
    await build_dispatcher(app).feed_update(bot, Update.model_validate({"update_id": 1, "message": msg}))
    job = store.db.execute("SELECT * FROM jobs").fetchone()
    assert job["message_id"] == 42 and json.loads(job["payload"])["file_id"] == "media-id"
    assert isinstance(calls[-1], SendMessage) and calls[-1].reply_parameters.message_id == 42
    worker = asyncio.create_task(app.worker())
    try:
        async with asyncio.timeout(2):
            while store.job(job["id"], 1)["status"] != "done":
                await asyncio.sleep(0.01)
        assert calls[-1].text == "Recognized speech" and calls[-1].message_id == 99
        assert calls[-1].reply_markup.inline_keyboard[0][0].text == "Кратко"
        assert not seen[0].exists() and not list((tmp_path / "jobs").iterdir())
    finally:
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker
        store.close()


async def test_json_document_still_imports_vocabulary(tmp_path):
    store = Store(tmp_path / "state.sqlite3")
    app = App(Settings(token="unused"), store, AsyncMock())
    msg = AsyncMock()
    msg.document.file_name, msg.document.mime_type = "terms.json", "application/json"
    msg.document.file_size, msg.from_user.id = 20, 1
    import io

    app.bot.download_file.return_value = io.BytesIO(b'[{"canonical":"GitHub"}]')
    await app.document(msg)
    assert store.terms(1)[0]["canonical"] == "GitHub"
    assert store.claim() is None
    store.close()


@pytest.mark.parametrize("extra", [{"duration": 601}, {"file_size": 19_000_001}])
async def test_media_limits_before_queue(tmp_path, extra):
    store = Store(tmp_path / "state.sqlite3")
    app = App(Settings(token="unused"), store, AsyncMock())
    msg = Message.model_validate({"message_id": 1, "date": 0, "chat": {"id": 1, "type": "private"},
        "video_note": {"file_id": "id", "file_unique_id": "u", "length": 240, "duration": 2, **extra}})
    msg = msg.as_(Bot("12345:test_only"))
    msg.bot.session = AsyncMock()
    await app.media(msg)
    assert store.claim() is None
    app.bot.get_file.assert_not_called()
    store.close()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg integration requires installed runtime")
async def test_real_ffmpeg_extracts_audio_and_rejects_bad_media(tmp_path):
    src = tmp_path / "source.wav"
    with wave.open(str(src), "wb") as f:
        f.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        f.writeframes(b"\x00\x00" * 32000)
    video = tmp_path / "video.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=size=32x32:rate=1",
                    "-i", str(src), "-shortest", "-c:v", "mpeg4", "-c:a", "aac", str(video)], check=True)
    s = Settings(token="unused", max_seconds=5)
    wav = await prepare_audio(s, video)
    with wave.open(str(wav)) as f:
        assert f.getnchannels() == 1 and f.getframerate() == 16000
        assert 1.9 < f.getnframes() / f.getframerate() < 2.2
    with pytest.raises(InferenceError, match="длиннее"):
        await prepare_audio(Settings(token="unused", max_seconds=1), video)
    silent = tmp_path / "silent.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-an", "-c:v", "copy", str(silent)], check=True)
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"this is not audio")
    for path in [silent, bad]:
        with pytest.raises(InferenceError, match="извлечь звук"):
            await prepare_audio(s, path)


async def test_api_receives_extracted_audio_instead_of_video(tmp_path, monkeypatch):
    video, wav = tmp_path / "input.mp4", tmp_path / "prepared.wav"
    video.write_bytes(b"private video frames")
    wav.write_bytes(b"audio only")
    prepare = AsyncMock(return_value=wav)
    monkeypatch.setattr("voicey.inference.prepare_audio", prepare)
    from test_api import server

    async def handler(request):
        from aiohttp import web

        part = await (await request.multipart()).next()
        assert part.filename == "audio.wav" and part.headers["Content-Type"] == "audio/wav"
        assert bytes(await part.read()) == b"audio only"
        return web.json_response({"text": "recognized"})

    monkeypatch.setenv("TEST_KEY", "test-only")
    async with server(handler, "/audio/transcriptions") as url:
        p = Profile("api", "API", "stt", "api", "test", url, "TEST_KEY")
        assert await transcribe(Settings(token="unused"), p, video, "", "ru") == "recognized"
    prepare.assert_awaited_once()
