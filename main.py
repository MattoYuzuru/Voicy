import asyncio
import logging
import os
import sys

from aiogram import Bot, Dispatcher
from aiogram.filters import CommandStart
from aiogram.types import Message

from config import TOKEN
from text_summary_ai import create_summary
from voice_to_text import convert_voice_to_text

dp = Dispatcher()
bot = Bot(TOKEN)


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    await message.answer(
        f"Привет, {message.from_user.full_name}! Отправь мне голосовое сообщение для обобщения.")


@dp.message(lambda message: message.voice is not None)
async def handle_voice_message(message: Message) -> None:
    file = await bot.get_file(message.voice.file_id)
    id = file.file_id
    await bot.download_file(file_path=file.file_path, destination=f"{id}.ogg")

    try:
        converted_text = convert_voice_to_text(f"{id}.ogg")
        print(converted_text)
        summary = create_summary(converted_text)
    except Exception as e:
        summary = f"Ошибочка вышла: {str(e)}"
    finally:
        os.remove(f"{id}.ogg")

    await message.answer(summary)


async def main() -> None:
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
