"""Model smoke test, no Telegram/API credentials; run inside the built CPU image."""

import argparse
import asyncio
import json
import resource
import shutil
import tempfile
import time
from pathlib import Path

from voicey.config import Profile, Settings
from voicey.inference import TASKS, process_text, transcribe


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["stt", "llm"])
    parser.add_argument("--audio", type=Path)
    args = parser.parse_args()
    s = Settings(token="unused")
    start = time.monotonic()
    if args.mode == "stt":
        p = Profile("whisper", "Whisper", "stt", "local", "/models/ggml-large-v3-turbo-q5_0.bin")
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "voice.ogg"
            shutil.copyfile(args.audio, target)
            text = await transcribe(s, p, target, "Telegram, VPS, Docker, SQLite, API, GitHub Actions", "ru")
    else:
        p = Profile("bonsai", "Bonsai", "llm", "local", "/models/Ternary-Bonsai-2-2B-PQ2_0.gguf")
        text = await process_text(
            s,
            p,
            "Ну, короче, нужно сделать Telegram-бота на VPS. Используем Docker и SQLite. "
            "Сначала проверим GitHub Actions. Потом перезапустим сервис. Базу данных не удалять.",
            TASKS["summary"],
        )
    print(
        json.dumps(
            {
                "mode": args.mode,
                "seconds": round(time.monotonic() - start, 2),
                "child_maxrss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                "result": text,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
