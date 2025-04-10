# live_transcriber.py

import openai
import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import tempfile
import asyncio
from pydub import AudioSegment

openai.api_key = os.getenv("OPENAI_API_KEY")

# Whisper transcribe helper
def transcribe_audio_bytes(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(audio_bytes)
        tmpfile.flush()
        with open(tmpfile.name, "rb") as f:
            transcript = openai.Audio.transcribe("whisper-1", f)
        return transcript["text"]
