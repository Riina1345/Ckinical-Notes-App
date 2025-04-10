import openai
import os

def transcribe_audio(audio_path):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    with open(audio_path, "rb") as f:
        transcript = openai.Audio.transcribe("whisper-1", f)
    return transcript["text"]
