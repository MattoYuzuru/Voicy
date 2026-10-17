"""A reply that tracks a job and is replaced with its final text."""

import asyncio
import contextlib
import json
import logging
import time

from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.types import ReplyParameters

log = logging.getLogger("voicey")
STAGES = {
    "queued": "Принято. Файл в очереди на транскрибацию.",
    "queued_text": "Принято. Обработка текста в очереди.",
    "starting": "Начинаю обработку…",
    "download": "Скачиваю файл…",
    "prepare": "Извлекаю и подготавливаю звук…",
    "transcribe": "Транскрибация",
    "api_stt": "Распознаю через API…",
    "load_llm": "Загружаю модель для обработки текста…",
    "edit": "Обрабатываю текст…",
}


class JobStatus:
    def __init__(self, bot, store, job):
        self.bot, self.store, self.job = bot, store, job
        self.message_id = json.loads(job["payload"]).get("status_message_id")
        self.phase, self.percent = "starting", None
        self.started = time.monotonic()
        self.task = None
        self.retry_at = 0

    def update(self, phase: str, percent: int | None = None):
        if (phase, percent) != (self.phase, self.percent):
            log.info("job=%s progress stage=%s percent=%s", self.job["id"], phase, percent)
        self.phase = phase
        self.percent = min(99, max(0, percent)) if percent is not None else None

    def text(self):
        elapsed = int(time.monotonic() - self.started)
        stage = STAGES[self.phase]
        if self.percent is not None:
            stage += f": {self.percent}%"
        if self.phase.startswith("queued"):
            return stage + "\nНа CPU это может занять несколько минут."
        return stage + f"\nПрошло {elapsed // 60}:{elapsed % 60:02d}."

    async def refresh(self):
        if time.monotonic() < self.retry_at:
            return
        try:
            if self.message_id:
                await self.bot.edit_message_text(self.text(), chat_id=self.job["chat_id"], message_id=self.message_id)
            else:
                message = await self.bot.send_message(
                    self.job["chat_id"],
                    self.text(),
                    reply_parameters=ReplyParameters(
                        message_id=self.job["message_id"], allow_sending_without_reply=True
                    ),
                )
                if isinstance(message.message_id, int):
                    self.message_id = message.message_id
                    self.store.patch_payload(self.job["id"], status_message_id=self.message_id)
        except TelegramRetryAfter as e:
            self.retry_at = time.monotonic() + e.retry_after
        except TelegramBadRequest as e:
            if "message is not modified" not in str(e).lower():
                log.warning("job=%s status_failed type=%s", self.job["id"], type(e).__name__)
        except Exception as e:
            # Progress delivery must never interrupt inference.
            log.warning("job=%s status_failed type=%s", self.job["id"], type(e).__name__)

    async def run(self):
        while True:
            await self.refresh()
            await asyncio.sleep(5)

    def start(self):
        self.task = asyncio.create_task(self.run())

    async def stop(self):
        if self.task:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task
