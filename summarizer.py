import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import nltk
nltk.data.path.append("/home/erp/side-projects/notes-summarizer/venv/nltk_data")
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Download NLTK data (run once)
nltk.download('punkt')
nltk.download('stopwords')

# Extractive Summary (picks important sentences)
def extractive_summary(text, num_sentences=3):
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text

    # Compute sentence importance
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
    tfidf_matrix = vectorizer.fit_transform(sentences)
    sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

    # Pick top sentences
    top_sentences = [sentences[i] for i in np.argsort(-sentence_scores)[:num_sentences]]
    return ' '.join(top_sentences)

# Abstractive Summary (rewrites in own words)
def abstractive_summary(text, max_length=150):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Hybrid = Extractive + Abstractive
def hybrid_summary(text, extractive_ratio=0.5, max_length=150):
    # Step 1: Extractive (reduce text size)
    num_sentences = max(1, int(extractive_ratio * len(sent_tokenize(text))))
    extracted_text = extractive_summary(text, num_sentences)

    # Step 2: Abstractive (rewrite shortened text)
    return abstractive_summary(extracted_text, max_length=max_length)

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    import pdfplumber
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text