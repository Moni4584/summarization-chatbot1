import streamlit as st
import speech_recognition as sr
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import tempfile
import nltk

# Download required NLTK data
nltk.download('punkt')

st.set_page_config(page_title="Text & Voice Summarization App", layout="centered")
st.title("📄 Text & Voice Summarization Chatbot")
st.write("Enter text or upload a WAV audio file to get a summary.")

# -------- Text Summarization Function --------
def summarize_text(text, sentence_count=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join([str(sentence) for sentence in summary])

# -------- Audio Summarization Function --------
def summarize_audio(audio_file):
    recognizer = sr.Recognizer()

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_file.read())
        temp_path = temp_audio.name

    with sr.AudioFile(temp_path) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)

    summary = summarize_text(text)
    return text, summary


tab1, tab2 = st.tabs(["📝 Text Summarization", "🎤 Voice Summarization"])

# -------- TEXT TAB --------
with tab1:
    st.subheader("Enter Text to Summarize")
    text_input = st.text_area("Paste your text here:")

    if st.button("Summarize Text"):
        if text_input.strip():
            with st.spinner("Summarizing..."):
                summary = summarize_text(text_input)
            st.success("✅ Summary Generated")
            st.write("**Summary:**")
            st.write(summary)
        else:
            st.warning("Please enter some text.")


# -------- AUDIO TAB --------
with tab2:
    st.subheader("Upload WAV Audio File")
    audio_file = st.file_uploader("Upload audio file (WAV only)", type=["wav"])

    if st.button("Summarize Audio"):
        if audio_file:
            with st.spinner("Transcribing and summarizing..."):
                try:
                    text, summary = summarize_audio(audio_file)
                    st.success("✅ Audio Processed")
                    st.write("**Transcription:**")
                    st.write(text)
                    st.write("**Summary:**")
                    st.write(summary)
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please upload a WAV audio file.")
