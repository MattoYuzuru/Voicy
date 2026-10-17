import asyncio
import contextlib
import fcntl
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

from aiogram import BaseMiddleware, Bot, Dispatcher, F
from aiogram.exceptions import TelegramNetworkError, TelegramRetryAfter, TelegramServerError
from aiogram.filters import Command
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    MessageEntity,
    ReplyParameters,
)

from voicey.config import Settings, load_settings
from voicey.db import Store
from voicey.formatting import chunks, utf16_len
from voicey.inference import TASKS, InferenceError, process_text, transcribe

log = logging.getLogger("voicey")
HELP = (
    "Пришли голосовое — отвечу полной расшифровкой. Под ней: краткое содержание, "
    "чистый текст, структура и свой промпт. Обработка текста меняет формулировки; оригинал остаётся выше.\n\n"
    "/settings — модели и язык (RU/EN/авто)\n"
    "/last — повторить последнюю расшифровку\n"
    "/add GitHub | гитхаб — добавить термин\n"
    "/remove GitHub — удалить термин\n"
    "/dict — словарь; /export — JSON; /seed — стартовые IT-термины\n"
    "Для импорта пришли JSON-файл с массивом {canonical, aliases}.\n"
    "/prompt текст инструкции — промпт для кнопки «Свой промпт»\n"
    "Аудио удаляется после обработки; расшифровки для кнопок хранятся 7 дней. "
    "При выборе API аудио или текст отправляется этому провайдеру."
)


class Access(BaseMiddleware):
    def __init__(self, settings):
        self.s = settings

    async def __call__(self, handler, event, data):
        user = event.from_user
        message = event.message if isinstance(event, CallbackQuery) else event
        if not user or (not self.s.public and user.id not in self.s.allowed_ids):
            if isinstance(event, CallbackQuery):
                await event.answer("Нет доступа.")
            return
        if not message or message.chat.type != "private":
            return
        return await handler(event, data)


def actions(jid: int):
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=title, callback_data=f"do:{jid}:{kind}") for kind, title in row]
            for row in [
                [("summary", "Кратко"), ("clean", "Чистый текст")],
                [("bullets", "Структура"), ("custom", "Свой промпт")],
            ]
        ]
    )


async def send_text(bot: Bot, chat: int, reply: int, text: str, markup=None):
    parts = chunks(text)
    for i, part in enumerate(parts):
        for attempt in range(3):
            try:
                await bot.send_message(
                    chat,
                    part,
                    entities=[MessageEntity(type="blockquote", offset=0, length=utf16_len(part))],
                    reply_parameters=ReplyParameters(message_id=reply, allow_sending_without_reply=True),
                    reply_markup=markup if i == len(parts) - 1 else None,
                )
                break
            except TelegramRetryAfter as e:
                if attempt == 2 or e.retry_after > 60:
                    raise
                await asyncio.sleep(e.retry_after + 0.1)
            except (TelegramNetworkError, TelegramServerError):
                if attempt == 2:
                    raise
                await asyncio.sleep(attempt + 1)


class App:
    def __init__(self, s: Settings, store: Store, bot: Bot):
        self.s, self.store, self.bot = s, store, bot
        self.wake = asyncio.Event()

    def selected(self, uid: int, kind: str):
        pid = self.store.get(uid, kind)
        p = self.s.profiles.get(pid) if pid else next((p for p in self.s.profiles.values() if p.kind == kind), None)
        return p if p and p.kind == kind and p.ready() else None

    async def settings(self, message: Message):
        uid = message.from_user.id
        rows, labels = [], []
        for kind, label in [("stt", "Распознавание"), ("llm", "Обработка текста")]:
            selected = self.selected(uid, kind)
            labels.append(f"{label}: {selected.label if selected else 'не настроено'}")
            for p in self.s.available(kind):
                rows.append(
                    [
                        InlineKeyboardButton(
                            text=("✓ " if p == selected else "") + p.label, callback_data=f"set:{kind}:{p.id}"
                        )
                    ]
                )
        language = self.store.get(uid, "language", "auto")
        rows.append(
            [
                InlineKeyboardButton(text=("✓ " if k == language else "") + v, callback_data=f"lang:{k}")
                for k, v in [("auto", "Авто RU/EN"), ("ru", "Русский"), ("en", "English")]
            ]
        )
        await message.answer(
            "\n".join(labels) + "\nПоказаны настроенные на сервере варианты. "
            "API отправляет данные выбранному провайдеру.",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=rows),
        )

    async def voice(self, message: Message):
        v = message.voice
        if v.duration > self.s.max_seconds or (v.file_size or 0) > self.s.max_bytes:
            await message.reply(f"Лимит: {self.s.max_seconds // 60} минут и 19 МБ.")
            return
        uid = message.from_user.id
        p = self.selected(uid, "stt")
        if not p:
            await message.reply("Распознавание не настроено. Проверь /settings.")
            return
        try:
            jid = self.store.enqueue(
                uid,
                message.chat.id,
                message.message_id,
                "stt",
                {
                    "file_id": v.file_id,
                    "profile": p.id,
                    "language": self.store.get(uid, "language", "auto"),
                    "hint": self.store.hint(uid),
                },
                self.s.max_pending,
            )
        except ValueError as e:
            await message.reply(str(e))
            return
        self.wake.set()
        if jid:
            await message.reply("Принято, распознаю. На CPU это может занять несколько минут.")

    async def callback(self, cb: CallbackQuery):
        parts = (cb.data or "").split(":")
        uid = cb.from_user.id
        if parts[0] == "set" and len(parts) == 3:
            kind, pid = parts[1:]
            p = self.s.profiles.get(pid)
            if p and p.kind == kind and p.ready():
                self.store.set(uid, kind, pid)
                await cb.answer("Выбрано: " + p.label)
                await self.settings_for_callback(cb)
            else:
                await cb.answer("Этот вариант недоступен.")
            return
        if parts[0] == "lang" and len(parts) == 2 and parts[1] in {"ru", "en", "auto"}:
            self.store.set(uid, "language", parts[1])
            await cb.answer("Язык сохранён.")
            await self.settings_for_callback(cb)
            return
        if len(parts) != 3 or parts[0] != "do" or not parts[1].isdigit():
            await cb.answer("Неизвестная кнопка.")
            return
        source = self.store.job(int(parts[1]), uid)
        if not source or source["kind"] != "stt" or source["status"] != "done" or not source["result"]:
            await cb.answer("Расшифровка недоступна или срок хранения истёк.")
            return
        action = parts[2]
        task = self.store.get(uid, "prompt") if action == "custom" else TASKS.get(action)
        if not task:
            await cb.answer("Сначала задай /prompt инструкцию.")
            return
        p = self.selected(uid, "llm")
        if not p:
            await cb.answer("Обработка текста не настроена. См. /settings.")
            return
        import hashlib

        fingerprint = hashlib.sha256((p.id + task).encode()).hexdigest()[:12]
        kind = f"{action}:{fingerprint}"
        try:
            jid = self.store.enqueue(
                uid,
                source["chat_id"],
                source["message_id"],
                kind,
                {"source_id": source["id"], "profile": p.id, "task": task},
                self.s.max_pending,
            )
        except ValueError as e:
            await cb.answer(str(e))
            return
        self.wake.set()
        cached = self.store.cached_job(uid, source["chat_id"], source["message_id"], kind) if not jid else None
        if cached and cached["result"]:
            await cb.answer("Возвращаю готовый результат.")
            await send_text(self.bot, source["chat_id"], source["message_id"], cached["result"])
        else:
            await cb.answer("Добавлено в очередь." if jid else "Этот вариант уже в очереди.")

    async def settings_for_callback(self, cb):
        # Build using the caller's ID, not the bot sender ID on cb.message.
        message = cb.message.model_copy(update={"from_user": cb.from_user})
        await self.settings(message)

    async def commands(self, message: Message):
        uid = message.from_user.id
        cmd, _, arg = (message.text or "").partition(" ")
        cmd = cmd.split("@")[0]
        arg = arg.strip()
        try:
            if cmd in {"/start", "/help"}:
                await message.answer(HELP)
            elif cmd == "/settings":
                await self.settings(message)
            elif cmd == "/last":
                source = self.store.last_transcript(uid)
                if source:
                    await send_text(
                        self.bot, source["chat_id"], source["message_id"], source["result"], actions(source["id"])
                    )
                else:
                    await message.answer("Сохранённых расшифровок пока нет.")
            elif cmd == "/add":
                canonical, _, aliases = arg.partition("|")
                self.store.add_terms(uid, [{"canonical": canonical, "aliases": aliases.strip()}])
                await message.answer("Термин сохранён как подсказка. Автоматических замен нет.")
            elif cmd == "/remove":
                self.store.remove_term(uid, arg)
                await message.answer("Термин удалён.")
            elif cmd == "/dict":
                text = "\n".join(
                    t["canonical"] + (" | " + t["aliases"] if t["aliases"] else "") for t in self.store.terms(uid)
                )
                await send_text(self.bot, message.chat.id, message.message_id, text or "Словарь пуст. /add или /seed")
            elif cmd == "/export":
                data = json.dumps(self.store.terms(uid), ensure_ascii=False, indent=2).encode()
                await message.answer_document(BufferedInputFile(data, filename="vocabulary.json"))
            elif cmd == "/seed":
                self.store.add_terms(uid, json.loads(Path("config/vocabulary.json").read_text()))
                await message.answer("Добавлен стартовый словарь IT-терминов. /dict — посмотреть.")
            elif cmd == "/prompt":
                if arg:
                    if len(arg) > 1000:
                        raise ValueError("Промпт: до 1000 символов.")
                    self.store.set(uid, "prompt", arg)
                    await message.answer("Промпт сохранён для кнопки «Свой промпт».")
                else:
                    await message.answer(
                        self.store.get(uid, "prompt") or "Задай: /prompt инструкцию для редактирования"
                    )
            else:
                await message.answer("Пришли голосовое или /help.")
        except (ValueError, KeyError, TypeError):
            await message.answer(
                "Не удалось сохранить. /add термин | варианты; JSON: [{canonical, aliases}]. Лимит 200 терминов."
            )

    async def document(self, message: Message):
        d = message.document
        if not (d.file_name or "").lower().endswith(".json") or (d.file_size or 0) > 64000:
            await message.reply("Для словаря нужен JSON до 64 КБ.")
            return
        try:
            f = await self.bot.get_file(d.file_id)
            data = await self.bot.download_file(f.file_path)
            entries = json.loads(data.read(64001))
            if not isinstance(entries, list):
                raise ValueError("list required")
            self.store.add_terms(message.from_user.id, entries)
            await message.reply("Словарь импортирован. /dict")
        except (ValueError, KeyError, TypeError, AttributeError):
            await message.reply('Неверный JSON словаря. Формат: [{"canonical":"GitHub","aliases":"гитхаб"}]')

    async def worker(self):
        last_cleanup = 0
        while True:
            if time.monotonic() - last_cleanup > 3600:
                self.store.cleanup(self.s.retention_days)
                last_cleanup = time.monotonic()
            job = self.store.claim()
            if not job:
                self.wake.clear()
                try:
                    await asyncio.wait_for(self.wake.wait(), 5)
                except TimeoutError:
                    pass
                continue
            started = time.monotonic()
            try:
                async with asyncio.timeout(self.s.timeout + 180):
                    result = await self.execute(job)
                # Persist before Telegram send. Delivery errors do not rerun expensive inference.
                self.store.finish(job["id"], result)
                await send_text(
                    self.bot,
                    job["chat_id"],
                    job["message_id"],
                    result,
                    actions(job["id"]) if job["kind"] == "stt" and result else None,
                )
                log.info("job=%s completed seconds=%.1f", job["id"], time.monotonic() - started)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # Exception strings can contain request URLs/tokens or user content.
                log.warning("job=%s failed type=%s", job["id"], type(e).__name__)
                if self.store.job(job["id"], job["user_id"])["status"] != "done":
                    self.store.finish(job["id"], None, "error")
                safe = (
                    str(e)
                    if isinstance(e, InferenceError)
                    else "Ошибка обработки. Проверь настройки или попробуй позже."
                )
                with contextlib.suppress(Exception):
                    await self.bot.send_message(
                        job["chat_id"],
                        safe,
                        reply_parameters=ReplyParameters(
                            message_id=job["message_id"], allow_sending_without_reply=True
                        ),
                    )

    async def execute(self, job):
        payload = json.loads(job["payload"])
        p = self.s.profiles[payload["profile"]]
        if not p.ready():
            raise InferenceError("Выбранная модель сейчас недоступна.")
        if job["kind"] != "stt":
            source = self.store.job(payload["source_id"], job["user_id"])
            if not source or not source["result"]:
                raise InferenceError("Исходная расшифровка больше не хранится.")
            return await process_text(self.s, p, source["result"], payload["task"])
        work = self.s.data_dir / "jobs"
        work.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=work) as directory:
            audio = Path(directory) / "voice.ogg"
            f = await self.bot.get_file(payload["file_id"])
            if (f.file_size or 0) > self.s.max_bytes:
                raise InferenceError("Файл превышает лимит 19 МБ.")
            await self.bot.download_file(f.file_path, destination=audio)
            if audio.stat().st_size > self.s.max_bytes:
                raise InferenceError("Файл превышает лимит 19 МБ.")
            return await transcribe(self.s, p, audio, payload["hint"], payload["language"])


async def main():
    os.umask(0o077)
    s = load_settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    # Prevent two pollers/workers sharing a DB/model budget.
    with (s.data_dir / "instance.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        work = s.data_dir / "jobs"
        if work.exists():
            for d in work.iterdir():
                if d.is_dir() and d.name.startswith("tmp"):
                    shutil.rmtree(d)
        store = Store(s.data_dir / "voicey.sqlite3")
        async with Bot(s.token) as bot:
            me = await bot.get_me()
            log.info("starting @%s", me.username)
            if (await bot.get_webhook_info()).url:
                raise RuntimeError("Webhook configured: explicitly remove it before polling")
            app, dp = App(s, store, bot), Dispatcher()
            dp.message.outer_middleware(Access(s))
            dp.callback_query.outer_middleware(Access(s))
            dp.message.register(
                app.commands,
                Command("start", "help", "settings", "last", "add", "remove", "dict", "export", "seed", "prompt"),
            )
            dp.message.register(app.voice, F.voice)
            dp.message.register(app.document, F.document)
            dp.message.register(app.commands)
            dp.callback_query.register(app.callback)
            try:
                async with asyncio.TaskGroup() as tasks:
                    worker = tasks.create_task(app.worker())
                    try:
                        # Sequential update persistence; inference happens only in worker.
                        await dp.start_polling(
                            bot, handle_as_tasks=False, allowed_updates=["message", "callback_query"]
                        )
                    finally:
                        worker.cancel()
            finally:
                store.close()


def run():
    logging.basicConfig(level=logging.WARNING)
    log.setLevel(logging.INFO)
    asyncio.run(main())
