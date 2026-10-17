import asyncio
import contextlib
import json
import sys
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import AnswerCallbackQuery, EditMessageText, SendMessage
from aiogram.types import Chat, Message, Update

from voicey.bot import App, build_dispatcher, send_text
from voicey.config import Profile, Settings
from voicey.db import Store
from voicey.inference import command
from voicey.status import JobStatus


async def test_native_progress_parses_only_numeric_callback_lines(caplog):
    code = "import sys; sys.stderr.write('private decoded text\\nwhisper_print_progress_callback: progress =  33%\\nwhisper_print_progress_callback: progress = 100%\\n')"
    seen = []
    await command(sys.executable, "-c", code, on_progress=seen.append)
    assert seen == [33, 100]
    assert "private decoded text" not in caplog.text


async def test_progress_reader_timeout_also_reaps_child(tmp_path):
    marker = tmp_path / "pid"
    code = f"import os,time; open({str(marker)!r},'w').write(str(os.getpid())); time.sleep(60)"
    with pytest.raises(TimeoutError):
        await command(sys.executable, "-c", code, timeout=0.5, on_progress=lambda v: None)
    import os

    with pytest.raises(ProcessLookupError):
        os.kill(int(marker.read_text()), 0)


async def test_status_persists_reply_id_and_stops_before_100(tmp_path):
    store = Store(tmp_path / "state.sqlite3")
    jid = store.enqueue(1, 10, 42, "stt", {})
    bot = AsyncMock()
    bot.send_message.return_value = Message(message_id=99, date=0, chat=Chat(id=10, type="private"))
    status = JobStatus(bot, store, store.job(jid, 1))
    status.update("queued")
    await status.refresh()
    assert json.loads(store.job(jid, 1)["payload"])["status_message_id"] == 99
    assert bot.send_message.call_args.kwargs["reply_parameters"].message_id == 42
    status.update("transcribe", 100)
    await status.refresh()
    assert "99%" in bot.edit_message_text.call_args.args[0]
    assert bot.edit_message_text.call_args.kwargs["message_id"] == 99
    status.start()
    await status.stop()
    assert status.task.done()
    store.close()


async def test_final_text_replaces_status_and_deleted_status_falls_back():
    bot = AsyncMock()
    await send_text(bot, 10, 42, "done", replace_message_id=99)
    bot.send_message.assert_not_called()
    assert bot.edit_message_text.call_args.kwargs["message_id"] == 99
    assert bot.edit_message_text.call_args.kwargs["entities"][0].type == "blockquote"
    bot.edit_message_text.side_effect = TelegramBadRequest(
        method=EditMessageText(text="x", chat_id=10, message_id=99), message="message to edit not found"
    )
    await send_text(bot, 10, 42, "done", replace_message_id=99)
    assert bot.send_message.call_args.kwargs["reply_parameters"].message_id == 42


async def test_dispatcher_start_voice_and_button_roundtrip(tmp_path):
    bot = Bot("12345:test_only")
    sent, counter = [], 900

    async def api(_bot, method, **kwargs):
        nonlocal counter
        sent.append(method)
        if isinstance(method, AnswerCallbackQuery):
            return True
        if isinstance(method, SendMessage):
            counter += 1
        return Message(message_id=counter, date=0, chat=Chat(id=1, type="private")).as_(bot)

    bot.session = AsyncMock(side_effect=api)
    model = tmp_path / "model"
    model.touch()
    stt = Profile("whisper", "Whisper", "stt", "local", str(model))
    llm = Profile("bonsai", "Bonsai", "llm", "local", str(model))
    store = Store(tmp_path / "state.sqlite3")
    app = App(
        Settings(token="unused", allowed_ids=frozenset({1}), profiles={"whisper": stt, "bonsai": llm}), store, bot
    )
    dp = build_dispatcher(app)

    def message(uid, text=None, voice=None):
        return {
            "message_id": 42,
            "date": 0,
            "chat": {"id": uid, "type": "private"},
            "from": {"id": uid, "first_name": "Test", "is_bot": False},
            "text": text,
            "voice": voice,
        }

    await dp.feed_update(bot, Update.model_validate({"update_id": 1, "message": message(2, "/start")}))
    assert "Telegram ID: 2" in sent[-1].text
    await dp.feed_update(bot, Update.model_validate({"update_id": 2, "message": message(1, "/start")}))
    assert "Пришли голосовое" in sent[-1].text
    await dp.feed_update(
        bot,
        Update.model_validate(
            {"update_id": 3, "message": message(1, voice={"file_id": "test", "file_unique_id": "u", "duration": 94})}
        ),
    )
    row = store.db.execute("SELECT * FROM jobs").fetchone()
    source_id = row["id"]
    status_id = json.loads(row["payload"])["status_message_id"]
    assert "очереди" in sent[-1].text and sent[-1].reply_parameters.message_id == 42

    async def execute(job, progress):
        progress("transcribe" if job["kind"] == "stt" else "edit", 33 if job["kind"] == "stt" else None)
        await asyncio.sleep(0.01)
        return "full transcript" if job["kind"] == "stt" else "summary result"

    app.execute = execute
    worker = asyncio.create_task(app.worker())
    try:
        async with asyncio.timeout(2):
            while store.job(source_id, 1)["status"] != "done":
                await asyncio.sleep(0.01)
        edits = [x for x in sent if isinstance(x, EditMessageText)]
        assert edits[-1].text == "full transcript" and edits[-1].message_id == status_id
        assert edits[-1].reply_markup.inline_keyboard[0][0].callback_data == f"do:{source_id}:summary"
        callback = {
            "id": "test-callback",
            "chat_instance": "test",
            "from": {"id": 1, "first_name": "Test", "is_bot": False},
            "message": message(1, "full transcript"),
            "data": f"do:{source_id}:summary",
        }
        await dp.feed_update(bot, Update.model_validate({"update_id": 4, "callback_query": callback}))
        async with asyncio.timeout(2):
            while store.db.execute("SELECT count(*) FROM jobs WHERE status='done'").fetchone()[0] < 2:
                await asyncio.sleep(0.01)
        assert any(isinstance(x, AnswerCallbackQuery) for x in sent)
        assert any(isinstance(x, EditMessageText) and x.text == "summary result" for x in sent)
    finally:
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker
        store.close()


async def test_status_network_failure_does_not_fail_job(tmp_path):
    store = Store(tmp_path / "state.sqlite3")
    jid = store.enqueue(1, 10, 42, "stt", {})
    bot = AsyncMock()
    bot.send_message.side_effect = TimeoutError("temporary")
    status = JobStatus(bot, store, store.job(jid, 1))
    await status.refresh()
    assert store.job(jid, 1)["status"] == "queued"
    store.close()


async def test_long_result_edits_first_chunk_and_keeps_reply_target():
    from voicey.bot import actions

    bot = AsyncMock()
    await send_text(bot, 10, 42, "я" * 4000, actions(1), replace_message_id=99)
    assert len(bot.edit_message_text.call_args.args[0]) == 3500
    assert bot.edit_message_text.call_args.kwargs["reply_markup"] is None
    assert len(bot.send_message.call_args.args[1]) == 500
    assert bot.send_message.call_args.kwargs["reply_parameters"].message_id == 42
    assert bot.send_message.call_args.kwargs["reply_markup"].inline_keyboard
