import os

import whisper
from pydub import AudioSegment


def convert_voice_to_text(voice_filename: str) -> str:
    rename_file = f"{voice_filename.split('.')[0]}.wav"

    audio = AudioSegment.from_ogg(voice_filename)
    audio.export(rename_file, format="wav")

    model = whisper.load_model("base")
    converted_text = model.transcribe(f"{rename_file}", language="ru")

    os.remove(rename_file)

    return converted_text['text']
