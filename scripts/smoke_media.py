"""Synthetic media through real ffmpeg/Whisper and Dispatcher; no Telegram token.

Run inside the CPU image with models mounted, e.g.:
PYTHONPATH=/app python /checks/smoke_media.py --audio /fixtures/speech.ogg
The fake Telegram transport copies the supplied synthetic fixture on download.
Only validation metadata is printed, never the recognized text.
"""

import argparse
import asyncio
import contextlib
import json
import resource
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock

from aiogram import Bot
from aiogram.methods import EditMessageText, SendMessage
from aiogram.types import Chat, File, Message, Update

import voicey.bot as bot_module
from voicey.bot import App, build_dispatcher
from voicey.config import Profile, Settings
from voicey.db import Store
from voicey.inference import command, transcribe


async def check(root, kind, fixture):
    start = time.monotonic()
    bot = Bot("12345:synthetic_test_only")
    calls, progress = [], []

    async def api(_bot, method, **kwargs):
        calls.append(method)
        return Message(message_id=99, date=0, chat=Chat(id=1, type="private")).as_(bot)

    bot.session = AsyncMock(side_effect=api)
    bot.get_file = AsyncMock(return_value=File(file_id="fixture", file_unique_id="fixture",
                                             file_path="fixture", file_size=fixture.stat().st_size))

    async def download(path, destination):
        destination.write_bytes(fixture.read_bytes())

    bot.download_file = AsyncMock(side_effect=download)

    async def recognize(s, p, audio, hint, language, callback):
        def track(phase, percent=None):
            progress.append((phase, percent))
            callback(phase, percent)

        return await transcribe(s, p, audio, hint, language, track)

    bot_module.transcribe = recognize
    data = root / kind
    data.mkdir()
    store = Store(data / "test.sqlite3")
    store.set(1, "language", "ru")
    profile = Profile("whisper", "Whisper", "stt", "local", "/models/ggml-large-v3-turbo-q5_0.bin")
    app = App(Settings(token="unused", allowed_ids=frozenset({1}), data_dir=data,
                       profiles={profile.id: profile}, timeout=1800), store, bot)
    media = {"file_id": "fixture", "file_unique_id": "fixture", "duration": 11,
             "file_size": fixture.stat().st_size}
    if kind == "video_note":
        media["length"] = 240
    message = {"message_id": 42, "date": 0, "chat": {"id": 1, "type": "private"},
               "from": {"id": 1, "first_name": "Synthetic", "is_bot": False}, kind: media}
    await build_dispatcher(app).feed_update(bot, Update.model_validate({"update_id": 1, "message": message}))
    job = store.db.execute("SELECT * FROM jobs").fetchone()
    assert job and isinstance(calls[-1], SendMessage) and calls[-1].reply_parameters.message_id == 42
    worker = asyncio.create_task(app.worker())
    try:
        async with asyncio.timeout(1900):
            while store.job(job["id"], 1)["status"] not in {"done", "error"}:
                await asyncio.sleep(0.1)
        result = store.job(job["id"], 1)
        assert result["status"] == "done" and len(result["result"]) > 10, "inference failed"
        final = next(m for m in reversed(calls) if isinstance(m, EditMessageText))
        assert final.text == result["result"] and final.message_id == 99 and final.reply_markup
        assert final.entities[0].type == "blockquote"
        assert not list((data / "jobs").iterdir()), "temporary media remains"
        assert any(percent and percent > 0 for phase, percent in progress if phase == "transcribe")
        print(json.dumps({"media": kind, "seconds": round(time.monotonic() - start, 2),
                          "result_chars": len(result["result"]), "final_replaced_status": True,
                          "numeric_progress": True, "temporary_files": 0,
                          "child_maxrss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}), flush=True)
    finally:
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker
        store.close()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True, help="Synthetic speech only")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mp4, mp3 = root / "round.mp4", root / "audio.mp3"
        await command("ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "color=s=240x240:r=1",
                      "-i", str(args.audio), "-shortest", "-c:v", "libx264", "-threads", "1",
                      "-pix_fmt", "yuv420p", "-c:a", "aac", str(mp4), timeout=60)
        await command("ffmpeg", "-nostdin", "-v", "error", "-i", str(args.audio), "-vn",
                      "-c:a", "libmp3lame", "-b:a", "64k", str(mp3), timeout=60)
        await check(root, "video_note", mp4)
        await check(root, "audio", mp3)


if __name__ == "__main__":
    asyncio.run(main())
