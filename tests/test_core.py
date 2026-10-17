import json
import os
import sys
from unittest.mock import AsyncMock

import pytest
from aiogram.types import Chat, Message, User, Voice

from voicey.bot import Access, App, send_text
from voicey.config import Profile, Settings, load_settings
from voicey.db import Store
from voicey.formatting import chunks, utf16_len
from voicey.inference import InferenceError, child_env, command, process_text


@pytest.fixture
def store(tmp_path):
    db = Store(tmp_path / "bot.sqlite3")
    yield db
    db.close()


def test_queue_survives_restart_and_deduplicates(tmp_path):
    path = tmp_path / "state.sqlite3"
    db = Store(path)
    jid = db.enqueue(1, 2, 3, "stt", {"file_id": "audio"})
    assert db.enqueue(1, 2, 3, "stt", {}) is None
    assert db.claim()["id"] == jid
    assert db.claim() is None
    db.close()
    db = Store(path)
    assert db.claim()["id"] == jid
    db.finish(jid, "result")
    assert db.claim() is None
    assert db.job(jid, 1)["result"] == "result"
    assert db.job(jid, 999) is None
    db.close()


def test_queue_limit_and_error_retry(store):
    jid = store.enqueue(1, 2, 3, "stt", {}, limit=1)
    with pytest.raises(ValueError):
        store.enqueue(1, 2, 4, "stt", {}, limit=1)
    store.finish(jid, None, "error")
    assert store.enqueue(1, 2, 3, "stt", {"retry": True}) == jid
    assert json.loads(store.claim()["payload"])["retry"]


def test_dictionary_isolation_atomic_import_and_short_hint(store):
    store.add_terms(1, [{"canonical": "GitHub", "aliases": ["гитхаб"]}])
    assert store.terms(2) == []
    assert store.terms(1)[0]["aliases"] == "гитхаб"
    with pytest.raises(ValueError):
        store.add_terms(1, [{"canonical": "valid"}, {"canonical": ""}])
    assert len(store.terms(1)) == 1
    store.add_terms(1, [{"canonical": "term" + str(i)} for i in range(40)])
    assert len(store.hint(1)) <= 400
    assert "term39" in store.hint(1)
    store.set(1, "prompt", "my task")
    assert store.get(2, "prompt") == ""


def test_retention_preserves_vocabulary_and_pending(store):
    jid = store.enqueue(1, 2, 3, "stt", {})
    store.finish(jid, "secret transcript")
    store.db.execute("UPDATE jobs SET updated=0")
    pending = store.enqueue(1, 2, 4, "stt", {})
    store.add_terms(1, [{"canonical": "Docker"}])
    store.cleanup(7)
    assert store.job(jid, 1) is None
    assert store.job(pending, 1)
    assert store.terms(1)


@pytest.mark.parametrize("text", ["<> & quotes", "😀" * 5000, "Русский\nEnglish " * 1000])
def test_telegram_chunks_preserve_unicode(text):
    parts = chunks(text)
    assert "".join(parts) == text
    assert all(utf16_len(part) <= 3500 for part in parts)


async def test_reply_and_blockquote_entities():
    bot = AsyncMock()
    await send_text(bot, 10, 42, "<hello> 😀")
    kwargs = bot.send_message.call_args.kwargs
    assert kwargs["reply_parameters"].message_id == 42
    assert kwargs["entities"][0].type == "blockquote"
    assert kwargs["entities"][0].length == utf16_len("<hello> 😀")


async def test_access_rejects_unknown_and_groups():
    middleware = Access(Settings(token="hidden", allowed_ids=frozenset({1})))
    handler = AsyncMock()

    def msg(uid, kind):
        return Message(
            message_id=1, date=0, chat=Chat(id=10, type=kind), from_user=User(id=uid, is_bot=False, first_name="User")
        )

    await middleware(handler, msg(2, "private"), {})
    await middleware(handler, msg(1, "group"), {})
    handler.assert_not_called()
    await middleware(handler, msg(1, "private"), {})
    handler.assert_awaited_once()


async def test_callback_cannot_read_another_users_transcript(store):
    jid = store.enqueue(1, 2, 3, "stt", {})
    store.finish(jid, "private")
    app = App(Settings(token="hidden"), store, AsyncMock())
    cb = AsyncMock()
    cb.data, cb.from_user.id = f"do:{jid}:summary", 999
    await app.callback(cb)
    assert store.claim() is None
    assert "недоступна" in cb.answer.call_args.args[0]


async def test_voice_limits_before_download(store):
    app = App(Settings(token="hidden", max_seconds=10), store, AsyncMock())
    msg = AsyncMock()
    msg.voice = Voice(file_id="id", file_unique_id="u", duration=11)
    await app.voice(msg)
    app.bot.get_file.assert_not_called()
    assert store.claim() is None


def test_children_do_not_inherit_secrets(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "hidden")
    monkeypatch.setenv("API_TOKEN", "hidden")
    monkeypatch.setenv("OPENROUTER_API_KEY", "hidden")
    assert not any(k in child_env() for k in ["BOT_TOKEN", "API_TOKEN", "OPENROUTER_API_KEY"])


async def test_subprocess_timeout_reaps_child(tmp_path):
    marker = tmp_path / "pid"
    code = f"import os,time; open({str(marker)!r},'w').write(str(os.getpid())); time.sleep(60)"
    with pytest.raises(TimeoutError):
        await command(sys.executable, "-c", code, timeout=0.5)
    pid = int(marker.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


async def test_local_long_text_rejected_before_model_start():
    s = Settings(token="hidden")
    p = Profile("local", "local", "llm", "local", "/nonexistent")
    with pytest.raises(InferenceError, match="слишком длинный"):
        await process_text(s, p, "я" * 4000, "summary")


def test_config_legacy_token_and_api_profiles(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BOT_TOKEN", raising=False)
    monkeypatch.setenv("API_TOKEN", "legacy")
    monkeypatch.setenv("ALLOWED_USER_IDS", "1,2")
    profile = {
        "id": "api",
        "label": "API",
        "kind": "llm",
        "backend": "api",
        "model": "model",
        "base_url": "https://example.com/v1",
        "key_env": "TEST_API_KEY",
    }
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps([profile]))
    monkeypatch.setenv("PROFILES_FILE", str(path))
    s = load_settings()
    assert s.token == "legacy" and s.allowed_ids == frozenset({1, 2})
    assert "legacy" not in repr(s)
    assert s.available("llm") == []
    monkeypatch.setenv("TEST_API_KEY", "hidden")
    assert s.available("llm")[0].id == "api"
    profile["base_url"] = "https://user:secret@example.com"
    path.write_text(json.dumps([profile]))
    with pytest.raises(ValueError):
        load_settings()


async def test_delivery_retries_without_running_model(monkeypatch):
    from aiogram.exceptions import TelegramNetworkError
    from aiogram.methods import SendMessage

    bot = AsyncMock()
    bot.send_message.side_effect = [
        TelegramNetworkError(method=SendMessage(chat_id=1, text="x"), message="temporary"),
        None,
    ]
    monkeypatch.setattr("voicey.bot.asyncio.sleep", AsyncMock())
    await send_text(bot, 1, 42, "saved result")
    assert bot.send_message.await_count == 2


async def test_last_transcript_is_user_scoped_and_replies_to_voice(store):
    jid = store.enqueue(1, 10, 42, "stt", {})
    store.finish(jid, "saved transcript")
    assert store.last_transcript(2) is None
    bot = AsyncMock()
    app = App(Settings(token="hidden"), store, bot)
    message = AsyncMock()
    message.from_user.id, message.text = 1, "/last"
    await app.commands(message)
    assert bot.send_message.call_args.args == (10, "saved transcript")
    assert bot.send_message.call_args.kwargs["reply_parameters"].message_id == 42


async def test_repeat_text_button_returns_cache_without_inference(store, tmp_path):
    import hashlib

    from voicey.inference import TASKS

    model = tmp_path / "model.gguf"
    model.touch()
    p = Profile("bonsai", "Bonsai", "llm", "local", str(model))
    app = App(Settings(token="hidden", profiles={p.id: p}), store, AsyncMock())
    source = store.enqueue(1, 10, 42, "stt", {})
    store.finish(source, "original")
    fingerprint = hashlib.sha256((p.id + TASKS["summary"]).encode()).hexdigest()[:12]
    jid = store.enqueue(1, 10, 42, f"summary:{fingerprint}", {})
    store.finish(jid, "cached summary")
    cb = AsyncMock()
    cb.data, cb.from_user.id = f"do:{source}:summary", 1
    await app.callback(cb)
    assert store.claim() is None
    assert app.bot.send_message.call_args.args == (10, "cached summary")


def test_unavailable_local_profile_never_silently_switches_to_api(store, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "test-only")
    local = Profile("local", "Local", "stt", "local", "/missing/model.bin")
    api = Profile("api", "API", "stt", "api", "model", "https://example.com", "TEST_KEY")
    app = App(Settings(token="hidden", profiles={local.id: local, api.id: api}), store, AsyncMock())
    assert app.selected(1, "stt") is None
    store.set(1, "stt", "local")
    assert app.selected(1, "stt") is None
    store.set(1, "stt", "api")
    assert app.selected(1, "stt") == api
