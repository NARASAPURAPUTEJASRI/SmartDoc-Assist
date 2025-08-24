import cv2
import re
import numpy as np
import pytesseract
from PIL import Image
from transformers import pipeline
from googletrans import Translator
from gtts import gTTS
import streamlit as st
import os

# Tesseract path (Windows only, adjust if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load more robust QA model
qa = pipeline("question-answering", model="deepset/minilm-uncased-squad2")

st.title("📄 Smart Document Analyzer")

# Preprocessing function
def preprocess_image(uploaded_file):
    uploaded_file.seek(0)
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return ""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    gray = cv2.medianBlur(gray, 3)

    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
    )

    config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(thresh, lang="eng", config=config)
    return text

# Clean extracted text
def clean_text(text):
    text = re.sub(r"[^A-Za-z0-9\s:/\-.,]", " ", text)  # remove junk chars
    return re.sub(r"\s+", " ", text).strip()

# File uploader
uploaded_file = st.file_uploader("Upload a document image", type=["jpg", "jpeg", "png", "bmp", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    with st.expander("🖼 Uploaded Document (click to expand/close)"):
        st.image(image, caption="Uploaded Document", use_container_width=True)

    extracted_text = preprocess_image(uploaded_file)
    extracted_text = clean_text(extracted_text)

    if extracted_text:
        st.success("✅ Document scanned successfully!")

        with st.expander("📜 Extracted Text (click to expand/close)"):
            st.text(extracted_text)

        user_question = st.text_input("❓ Ask a question about the document:")

        if user_question:
            try:
                answer = qa(question=user_question, context=extracted_text)
                final_answer = answer["answer"].strip()

                if not final_answer or final_answer == "":
                    final_answer = "No answer found in document"
            except Exception:
                final_answer = "No answer found in document"

            st.subheader("📖 Answer")
            st.write(final_answer)

            translator = Translator()
            translated = translator.translate(final_answer, dest="en")
            st.subheader("🌍 Translated (English)")
            st.write(translated.text)

            tts = gTTS(text=final_answer, lang="en")
            audio_path = "answer.mp3"
            tts.save(audio_path)
            st.audio(audio_path, format="audio/mp3")

    else:
        st.error("⚠️ No text could be extracted from this image.")












#python -m streamlit run main_project.py

