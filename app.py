import streamlit as st
from summarizer import hybrid_summary, extract_text_from_pdf
import nltk
nltk.data.path.append("/home/erp/side-projects/notes-summarizer/venv/nltk_data")
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# App UI
st.title("📝 Hybrid Notes Summarizer")
st.write("Paste text or upload a PDF to get a summary!")

# Input options
input_option = st.radio("Input type:", ["Text", "PDF"])
text = ""
if input_option == "Text":
    text = st.text_area("Paste your notes here:", height=200)
else:
    uploaded_file = st.file_uploader("Upload a PDF:", type=["pdf"])
    if uploaded_file:
        text = extract_text_from_pdf(uploaded_file)
        st.text_area("Extracted Text:", text, height=200)

# Summary settings
st.sidebar.header("Settings")
length = st.sidebar.slider("Summary length (words):", 50, 300, 150)
extractive_ratio = st.sidebar.slider("Extractive ratio (%):", 10, 90, 50)

# Generate summary
if st.button("Summarize") and text:
    with st.spinner("Generating summary..."):
        summary = hybrid_summary(
            text,
            extractive_ratio=extractive_ratio/100,
            max_length=length
        )
    st.subheader("Summary:")
    st.success(summary)
else:
    st.warning("Please enter text or upload a PDF!")