import streamlit as st
import os
import re
import time
from openai import OpenAI
from filters import detect_non_clinical_terms
from transcription import transcribe_audio
from live_transcriber import transcribe_audio_bytes
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# --- App Config ---
st.set_page_config(page_title="Clinical Notes AI", layout="centered")
st.title("🧠 Clinical Notes AI")

# --- Load API Key ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OPENAI_API_KEY not found. Please set it in your environment.")
    st.stop()

client = OpenAI(api_key=api_key)

# --- CPT Codes ---
cpt_codes = [("90832", "Outpatient 15-30 minutes"), ("90834", "Outpatient"), ("90837", "OP Session 61-90 Minutes")]
# Truncated for brevity — use full list from earlier

# --- Input Method Selector ---
input_method = st.selectbox("Choose input method:", ["Type Notes", "Upload Audio", "Live Dictation"])

session_input = ""

# --- Typed Notes ---
if input_method == "Type Notes":
    typed_input = st.text_area("✍️ Type or paste your session notes", height=200)
    if typed_input:
        session_input = typed_input.strip()

# --- Uploaded Audio ---
if input_method == "Upload Audio":
    audio_file = st.file_uploader("🎙 Upload .mp3, .wav, or .m4a", type=["mp3", "wav", "m4a"])
    if audio_file:
        with open("temp_audio", "wb") as f:
            f.write(audio_file.read())
        session_input = transcribe_audio("temp_audio")
        st.text_area("📝 Transcribed text:", value=session_input, height=200)

# --- Continuous Dictation ---
if input_method == "Live Dictation":
    RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
   ctx = webrtc_streamer(
    key="live-dictation",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)


    if "live_text" not in st.session_state:
        st.session_state["live_text"] = ""

    if ctx.audio_receiver:
        st.warning("🎙 Listening... Keep speaking. Transcribing continuously...")
        audio_bytes = ctx.audio_receiver.get_frames(timeout=1)
        if audio_bytes:
            try:
                new_text = transcribe_audio_bytes(audio_bytes[0].to_ndarray().tobytes())
                st.session_state["live_text"] += " " + new_text
            except:
                pass

    session_input = st.session_state["live_text"]
    st.text_area("📝 Live transcript:", value=session_input, height=200)

# --- Format, Role, and Model ---
if session_input:
    st.markdown("### 📋 Summary Format")
    soap = st.checkbox("Generate SOAP", value=True)
    dap = st.checkbox("Generate DAP", value=False)
    role = st.selectbox("Clinician Role", ["Therapist", "Recovery Coach", "Social Worker", "SUD Counselor", "Other"])
    use_gpt4 = st.checkbox("Use GPT-4", value=False)

    if st.button("Generate Summary"):
        model = "gpt-4" if use_gpt4 else "gpt-3.5-turbo"
        formats = []
        if soap:
            formats.append("SOAP")
        if dap:
            formats.append("DAP")

        for fmt in formats:
            cpt_prompt = "\n".join([f"- {code}: {desc}" for code, desc in cpt_codes])
            prompt = f"""
            You are a {role} generating a {fmt} note for insurance documentation.

            Requirements:
            - Use clinical, professional language.
            - Include only medically recognized diagnoses (DSM-5 / ICD-10).
            - Avoid non-clinical or informal terms like 'inner child', 'chakra', etc.
            - Format as {fmt}.

            At the end of the note, suggest ONE CPT code from this list only:
            {cpt_prompt}

            Session Text:
            {session_input}
            """

            with st.spinner(f"Generating {fmt} note..."):
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a compliant clinical documentation assistant."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    summary = response.choices[0].message.content
                    st.subheader(f"📝 {fmt} Summary")
                    st.code(summary, language="markdown")

                    # Flagging
                    flags = detect_non_clinical_terms(summary)
                    if flags:
                        st.error("⚠️ Non-clinical terms detected:")
                        for f in flags:
                            st.write(f"- {f}")
                    else:
                        st.success("✅ No non-clinical terms detected.")

                    # Suggested CPT
                    suggested_cpt = next((f"{code} – {desc}" for code, desc in cpt_codes if re.search(rf"\\b{code}\\b", summary)), None)
                    if suggested_cpt:
                        st.markdown("### 💡 GPT-Suggested CPT Code")
                        st.code(suggested_cpt)
                    else:
                        st.warning("⚠️ No CPT code found in summary.")

                    # Manual Override
                    st.markdown("### 📝 Confirm or override CPT code")
                    options = [f"{code} – {desc}" for code, desc in cpt_codes]
                    index = options.index(suggested_cpt) if suggested_cpt in options else 0
                    selected_cpt = st.selectbox("CPT Code", options, index=index)

                    st.markdown("### ✅ Final CPT Code")
                    st.success(selected_cpt)

                except Exception as e:
                    st.error(f"Error: {str(e)}")
else:
    st.info("Please provide input using your selected method above.")
