import streamlit as st
from openai import OpenAI
import os
from filters import detect_non_clinical_terms

# 🔐 Read OpenAI API key from Streamlit Secrets
api_key = os.getenv("OPENAI_API_KEY")

# 🚨 Fail-safe: Stop the app if key is missing
if not api_key:
    st.error("❌ OPENAI_API_KEY not found. Please set it in Streamlit Cloud > Settings > Secrets.")
    st.stop()

# ✅ Initialize OpenAI client
client = OpenAI(api_key=api_key)

# 🌐 Page settings
st.set_page_config(page_title="Clinical Notes AI", layout="centered")
st.title("🧠 Clinical Notes AI (Demo)")

# 📝 User input
session_input = st.text_area("Enter session transcript or notes:", height=250)

# 🚀 Button: Trigger generation
if st.button("Generate SOAP Summary"):
    if not session_input.strip():
        st.warning("Please enter some session text.")
    else:
        with st.spinner("Generating summary..."):
            # 🧠 System prompt
            prompt = f"""
            You are a licensed clinician generating a SOAP note for insurance documentation.

            Requirements:
            - Use clinical, professional language.
            - Include medically recognized diagnoses (DSM-5 / ICD-10).
            - Avoid non-clinical terms like 'inner child', 'trauma bonding', etc.
            - Format as SOAP (Subjective, Objective, Assessment, Plan).

            Session Text:
            {session_input}
            """

            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",  # Change to gpt-4 if you have access
                    messages=[
                        {"role": "system", "content": "You are a compliant clinical documentation assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )

                summary = response.choices[0].message.content
                flags = detect_non_clinical_terms(summary)

                # 📄 Output
                st.subheader("📝 SOAP Summary")
                st.code(summary, language="markdown")

                if flags:
                    st.error("⚠️ Non-clinical terms detected:")
                    for f in flags:
                        st.write(f"- {f}")
                else:
                    st.success("✅ No non-clinical terms detected.")

            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")
