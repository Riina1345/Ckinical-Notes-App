import streamlit as st
import os
from openai import OpenAI
from filters import detect_non_clinical_terms
from transcription import transcribe_audio

st.set_page_config(page_title="Clinical Notes AI", layout="centered")
st.title("🧠 Clinical Notes AI (Local MVP)")

# Load API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OPENAI_API_KEY not found. Please set it in your environment.")
    st.stop()

client = OpenAI(api_key=api_key)

# --- Manual Text Input ---
st.markdown("### ✍️ Option 1: Type or paste your session notes")
typed_input = st.text_area("Session transcript (manual entry)", height=200, key="typed_input")

# --- Audio Upload ---
st.markdown("### 🎙 Option 2: Upload an audio file for transcription")
audio_file = st.file_uploader("Upload .mp3, .wav, or .m4a", type=["mp3", "wav", "m4a"])

transcript = ""
if audio_file:
    with open("temp_audio", "wb") as f:
        f.write(audio_file.read())
    transcript = transcribe_audio("temp_audio")
    st.success("✅ Audio successfully transcribed.")

# --- Final input logic ---
session_input = typed_input if typed_input.strip() else transcript

# --- SOAP/DAP Format Options ---
st.markdown("### 📋 Select Summary Format(s)")
generate_soap = st.checkbox("Generate SOAP Note", value=True)
generate_dap = st.checkbox("Generate DAP Note", value=False)

# --- Clinician Role Selection ---
st.markdown("### 👤 Select Clinician Role")
role = st.selectbox("Clinician Role", [
    "Therapist", "Recovery Coach", "Social Worker", "SUD Counselor", "Other"
])

# --- GPT Toggle ---
use_gpt4 = st.checkbox("Use GPT-4 (slower, more expensive)", value=False)

# --- Generate Summary ---
if st.button("Generate Summary"):
    if not session_input:
        st.warning("Please provide session notes either by typing or uploading audio.")
    elif not generate_soap and not generate_dap:
        st.warning("Please select at least one summary format.")
    else:
        with st.spinner("Generating summary..."):
            model = "gpt-4" if use_gpt4 else "gpt-3.5-turbo"
            selected_formats = []
            if generate_soap:
                selected_formats.append("SOAP")
            if generate_dap:
                selected_formats.append("DAP")

            for fmt in selected_formats:
                prompt = f"""
                You are a {role} generating a {fmt} note for insurance documentation.

                Requirements:
                - Use clinical, professional language.
                - Include only medically recognized diagnoses (DSM-5 / ICD-10).
                - Avoid non-clinical or informal terms like 'inner child', 'chakra', etc.
                - Format as {fmt}.

                Session Text:
                {session_input}
                """

                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a compliant clinical documentation assistant."},
                            {"role": "user", "content": prompt}
                        ]
                    )

                    summary = response.choices[0].message.content
                    flags = detect_non_clinical_terms(summary)

                    st.subheader(f"📝 {fmt} Summary")
                    st.code(summary, language="markdown")

                    if flags:
                        st.error("⚠️ Non-clinical terms detected:")
                        for f in flags:
                            st.write(f"- {f}")
                    else:
                        st.success("✅ No non-clinical terms detected.")

                except Exception as e:
                    st.error(f"Something went wrong generating the {fmt} note: {str(e)}")
