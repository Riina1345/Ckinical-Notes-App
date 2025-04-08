import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
from filters import detect_non_clinical_terms

# Load .env with OpenAI key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Clinical Notes AI", layout="centered")
st.title("🧠 Clinical Notes AI (Demo)")

session_input = st.text_area("Enter session transcript or notes:", height=250)

if st.button("Generate SOAP Summary"):
    if not session_input.strip():
        st.warning("Please enter some session text.")
    else:
        with st.spinner("Generating summary..."):
            prompt = f"""
            You are a licensed clinician generating a SOAP note for insurance purposes.

            Requirements:
            - Use clinical, professional language.
            - Include medically recognized diagnoses (DSM-5 / ICD-10).
            - Do NOT include non-clinical terms like 'inner child', 'trauma bonding', etc.
            - Format the response in SOAP (Subjective, Objective, Assessment, Plan) format.

            Session Text:
            {session_input}
            """

            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",  # Use "gpt-3.5-turbo" if needed
                    messages=[
                        {"role": "system", "content": "You are a compliant clinical documentation assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )

                summary = response.choices[0].message.content
                flags = detect_non_clinical_terms(summary)

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
