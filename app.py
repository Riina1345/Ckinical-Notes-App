import streamlit as st
from openai import OpenAI
import os
from filters import detect_non_clinical_terms

# 🔐 Load OpenAI API key from Streamlit Secrets
api_key = os.getenv("OPENAI_API_KEY")

# 🚨 Stop the app if the key is missing
if not api_key:
    st.error("❌ OPENAI_API_KEY not found. Please set it in Streamlit Cloud > Settings > Secrets.")
    st.stop()

# ✅ Initialize OpenAI client
client = OpenAI(api_key=api_key)

# 🌐 Page settings
st.set_page_config(page_title="Clinical Notes AI", layout="centered")
st.title("🧠 Clinical Notes AI")

# 🧾 Summary format selection
note_format = st.selectbox("Choose summary format:", ["SOAP", "DAP"])

# 📝 User input
session_input = st.text_area("Enter session transcript or notes:", height=250)

# 🚀 Generate button
if st.button("Generate Summary"):
    if not session_input.strip():
        st.warning("Please enter some session text.")
    else:
        with st.spinner("Generating summary..."):

            # 🧠 Prompt template based on format
            if note_format == "SOAP":
                format_instructions = (
                    "Format the response in SOAP format (Subjective, Objective, Assessment, Plan)."
                )
            else:  # DAP
                format_instructions = (
                    "Format the response in DAP format (Data, Assessment, Plan)."
                )

            prompt = f"""
            You are a licensed clinician generating a {note_format} note for insurance documentation.

            Requirements:
            - Use clinical, professional language.
            - Include only medically recognized diagnoses (DSM-5 / ICD-10).
            - Avoid non-clinical or informal terms like 'inner child', 'chakra', etc.
            - {format_instructions}

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
                st.error(f"Something went wrong: {str(e)}")
