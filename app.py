import streamlit as st
from openai import OpenAI
import os
from filters import detect_non_clinical_terms

# 🔐 Load OpenAI API key from Streamlit Secrets
api_key = os.getenv("OPENAI_API_KEY")

# 🚨 Stop the app if key is missing
if not api_key:
    st.error("❌ OPENAI_API_KEY not found. Please set it in Streamlit Cloud > Settings > Secrets.")
    st.stop()

# ✅ Initialize OpenAI client
client = OpenAI(api_key=api_key)

# 🌐 Page settings
st.set_page_config(page_title="Clinical Notes AI", layout="centered")
st.title("🧠 Clinical Notes AI")

# 🧾 Format picker: SOAP, DAP, or both
note_formats = st.multiselect(
    "Select summary format(s) to generate:",
    options=["SOAP", "DAP"],
    default=["SOAP"]
)

# 📝 Session input
session_input = st.text_area("Enter session transcript or notes:", height=250)

# 🚀 Generate button
if st.button("Generate Summary"):
    if not session_input.strip():
        st.warning("Please enter some session text.")
    elif not note_formats:
        st.warning("Please select at least one format.")
    else:
        with st.spinner("Generating summary..."):
            for note_format in note_formats:
                format_instruction = (
                    "Format as SOAP (Subjective, Objective, Assessment, Plan)."
                    if note_format == "SOAP"
                    else "Format as DAP (Data, Assessment, Plan)."
                )

                prompt = f"""
                You are a licensed clinician generating a {note_format} note for insurance documentation.

                Requirements:
                - Use clinical, professional language.
                - Include only medically recognized diagnoses (DSM-5 / ICD-10).
                - Avoid non-clinical or informal terms like 'inner child', 'chakra', etc.
                - {format_instruction}

                Session Text:
                {session_input}
                """

                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a compliant clinical documentation assistant."},
                            {"role": "user", "content": prompt}
                        ]
                    )

                    summary = response.choices[0].message.content
                    flags = detect_non_clinical_terms(summary)

                    st.subheader(f"📝 {note_format} Summary")
                    st.code(summary, language="markdown")

                    if flags:
                        st.error("⚠️ Non-clinical terms detected:")
                        for f in flags:
                            st.write(f"- {f}")
                    else:
                        st.success("✅ No non-clinical terms detected.")

                except Exception as e:
                    st.error(f"Something went wrong generating the {note_format} note: {str(e)}")
