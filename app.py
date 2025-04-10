import streamlit as st
import os
from openai import OpenAI
from filters import detect_non_clinical_terms
from transcription import transcribe_audio

st.set_page_config(page_title="Clinical Notes AI", layout="centered")
st.title("🧠 Clinical Notes AI (Local MVP)")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OPENAI_API_KEY not found. Please set it in your environment.")
    st.stop()

client = OpenAI(api_key=api_key)

# UI elements
st.markdown("### Step 1: Upload an audio file")
audio_file = st.file_uploader("Upload .mp3, .wav, or .m4a file", type=["mp3", "wav", "m4a"])

if audio_file:
    with open("temp_audio", "wb") as f:
        f.write(audio_file.read())

    transcript = transcribe_audio("temp_audio")
    st.markdown("### 📝 Transcription")
    st.text_area("Transcript", value=transcript, height=200, key="transcript")

    st.markdown("### Step 2: Select options and generate summary")
    role = st.selectbox("Clinician Role", ["Therapist", "Recovery Coach", "Social Worker", "SUD Counselor", "Other"])
    formats = []
    if st.checkbox("Generate SOAP"):
        formats.append("SOAP")
    if st.checkbox("Generate DAP"):
        formats.append("DAP")
    use_gpt4 = st.checkbox("Use GPT-4 (slower, more expensive)")

    if st.button("Generate Summary"):
        model = "gpt-4" if use_gpt4 else "gpt-3.5-turbo"
        for fmt in formats:
            prompt = f"""
            You are a {role} generating a {fmt} note for insurance documentation.

            Requirements:
            - Use clinical, professional language.
            - Include only medically recognized diagnoses (DSM-5 / ICD-10).
            - Avoid non-clinical or informal terms like 'inner child', 'chakra', etc.
            - Format as {fmt}.

            Session Text:
            {transcript}
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
                st.error(f"Something went wrong: {str(e)}")
