import streamlit as st
import os
import re
from openai import OpenAI
from filters import detect_non_clinical_terms
from transcription import transcribe_audio

st.set_page_config(page_title="Clinical Notes AI", layout="centered")
st.title("🧠 Clinical Notes AI (Local MVP)")

# Load API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OPENAI_API_KEY not found. Please set it in your environment.")
    st.stop()

client = OpenAI(api_key=api_key)

# --- CPT code list (code, description) ---
cpt_codes = [
    ("H0010", "Detox"), ("H0018", "Residential"), ("H0035", "PHP"), ("H0015", "IOP"),
    ("90792", "Psychiatric Diag. Eval. W. Med Services"), ("90791", "Assessment/Diag (BPS) w.out med services"),
    ("80305", "Utox"), ("87426", "COVID Antigen Testing w. BD Veritor"), ("87635", "COVID-19 Laboratory Testing"),
    ("90832", "Outpatient 15-30 minutes"), ("90834", "Outpatient"), ("90837", "OP Session 61-90 Minutes"),
    ("90839", "Crisis Psychotherapy 30-60min"), ("90840", "Crisis Psychotherapy additional 30 minutes"),
    ("90846", "Family Session w/out the Client"), ("90847", "Family Session with Client"),
    ("90853", "Outpatient Group"), ("93000", "EKG Performance"), ("96372", "MATs Admin/Injection"),
    ("97802", "Nutrition Initial Assessment"), ("97804", "Nutrition Group"), ("97810", "Acupuncture Add On"),
    ("97811", "Acupuncture – Add On Add'l 15 mins"), ("97026", "Acupuncture Add On Heat Lamp"),
    ("99211", "Medical Appointment (5-9 mins)"), ("99212", "Medical Appointment (10-14 mins)"),
    ("99213", "Acupuncture: New Client"), ("99214", "Psych Appointment (25-39 minutes)"),
    ("99215", "Psych Appointment (40+ minutes)"), ("99408", "SBI 15-30 MINS"), ("H0006", "Case Management"),
    ("99409", "SBI 30+ MINS"), ("98966", "Phone Assessment (5-10 mins)"), ("98967", "Phone Assessment (11-20 mins)"),
    ("98968", "Phone Assessment (21-30 mins)"), ("99366", "Team Conf. w/ Client or Family"),
    ("99367", "Team Conf. w/out Client – Physician"), ("99368", "Team Conf. w/out Client – Nonphysician"),
]

# --- UI: Input options ---
typed_input = st.text_area("✍️ Type or paste session notes here", height=200)

st.markdown("### 🎙 Or upload an audio file")
audio_file = st.file_uploader("Upload .mp3, .wav, or .m4a", type=["mp3", "wav", "m4a"])
transcript = ""

if audio_file:
    with open("temp_audio", "wb") as f:
        f.write(audio_file.read())
    transcript = transcribe_audio("temp_audio")
    st.success("✅ Audio transcribed successfully.")

session_input = typed_input if typed_input.strip() else transcript

# --- Format & role ---
st.markdown("### 📋 Select Summary Format(s)")
generate_soap = st.checkbox("Generate SOAP", value=True)
generate_dap = st.checkbox("Generate DAP", value=False)

st.markdown("### 🩺 Select Clinician Role")
role = st.selectbox("Role", ["Therapist", "Recovery Coach", "Social Worker", "SUD Counselor", "Other"])
use_gpt4 = st.checkbox("Use GPT-4", value=False)

# --- Generate ---
if st.button("Generate Summary"):
    if not session_input:
        st.warning("Please enter or upload session content.")
    elif not generate_soap and not generate_dap:
        st.warning("Please select at least one note format.")
    else:
        model = "gpt-4" if use_gpt4 else "gpt-3.5-turbo"
        formats = []
        if generate_soap:
            formats.append("SOAP")
        if generate_dap:
            formats.append("DAP")

        for fmt in formats:
            # Add CPT list to prompt
            cpt_prompt_list = "\\n".join([f"- {code}: {desc}" for code, desc in cpt_codes])

            prompt = f"""
            You are a {role} generating a {fmt} note for insurance documentation.

            Requirements:
            - Use clinical, professional language.
            - Include only medically recognized diagnoses (DSM-5 / ICD-10).
            - Avoid non-clinical or informal terms like 'inner child', 'chakra', etc.
            - Format as {fmt}.

            At the end of the note, suggest ONE CPT code from this list only:
            {cpt_prompt_list}

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
                    flags = detect_non_clinical_terms(summary)

                    # --- Output Summary ---
                    st.subheader(f"📝 {fmt} Summary")
                    st.code(summary, language="markdown")

                    # --- Flag non-clinical terms ---
                    if flags:
                        st.error("⚠️ Non-clinical terms detected:")
                        for f in flags:
                            st.write(f"- {f}")
                    else:
                        st.success("✅ No non-clinical terms detected.")

                    # --- Detect suggested CPT code from summary ---
                    suggested_code = None
                    for code, desc in cpt_codes:
                        if re.search(rf"\\b{code}\\b", summary):
                            suggested_code = f"{code} – {desc}"
                            break

                    if suggested_code:
                        st.markdown("### 💡 GPT-Suggested CPT Code")
                        st.code(suggested_code)
                    else:
                        st.warning("⚠️ No CPT code found in summary.")

                    # --- Manual override ---
                    st.markdown("### 📝 Confirm or override CPT code")
                    options = [f"{code} – {desc}" for code, desc in cpt_codes]
                    index = options.index(suggested_code) if suggested_code in options else 0
                    selected_cpt = st.selectbox("CPT Code", options, index=index)

                    st.markdown("### ✅ Final CPT Code Used")
                    st.success(selected_cpt)

                except Exception as e:
                    st.error(f"Something went wrong: {str(e)}")
