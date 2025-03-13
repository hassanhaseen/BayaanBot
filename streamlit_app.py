import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import os
from datetime import datetime
import json
import base64
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Custom module imports (from src/)
from src.utils import load_history, save_to_history

# Page config
st.set_page_config(
    page_title="BayaanBot - Roman Urdu Poetry Generator",
    page_icon="🖋️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS (same as before)
st.markdown("""
    <style>
    /* ... your dark + gold theme CSS here ... */
    </style>
""", unsafe_allow_html=True)

# Utility functions (this stays if not moved to utils.py)
def get_download_link(text, filename, link_text):
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:text/plain;base64,{b64}" download="{filename}" class="download-button">📥 {link_text}</a>'

@st.cache_resource
def load_model_and_encoder():
    try:
        model = tf.keras.models.load_model("models/poetry_gru_model.h5")  # <-- updated path
        with open("models/word_encoder.pkl", "rb") as f:                  # <-- updated path
            word_encoder = pickle.load(f)

        word_to_index = {word: i for i, word in enumerate(word_encoder.classes_)}
        index_to_word = {i: word for word, i in word_to_index.items()}

        return model, word_to_index, index_to_word
    except Exception as e:
        st.error(f"Error loading model or encoder: {str(e)}")
        return None, None, None

def generate_poetry(start_text, words_per_line, total_lines, model, word_to_index, index_to_word):
    try:
        generated_text = start_text.split()
        for _ in range(total_lines * words_per_line):
            encoded_input = [word_to_index.get(word, 0) for word in generated_text[-5:]]
            encoded_input = pad_sequences([encoded_input], maxlen=5, truncating="pre")

            predicted_probs = model.predict(encoded_input, verbose=0)
            predicted_index = np.argmax(predicted_probs, axis=-1)[0]
            next_word = index_to_word.get(predicted_index, "")

            if not next_word:
                continue

            generated_text.append(next_word)

            if len(generated_text) % words_per_line == 0:
                generated_text.append("\n")

        return " ".join(generated_text)
    except Exception as e:
        st.error(f"🚫 Error generating poetry: {str(e)}")
        return ""

# Load model and encoder
model, word_to_index, index_to_word = load_model_and_encoder()

if not all([model, word_to_index, index_to_word]):
    st.error("⚠️ Failed to load required components. Please check if all files exist.")
    st.stop()

# Header
st.title("🖋️ BayaanBot")
st.markdown("##### Express your thoughts in Roman Urdu Poetry powered by AI")

# Tabs
tab1, tab2, tab3 = st.tabs(["Generate Poetry", "History", "Analysis"])

with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🎯 Start Your Nazam")
        start_text = st.text_input(
            "Starting Words",
            value="dil ke armaan",
            help="Enter your opening words in Roman Urdu"
        )

    with col2:
        st.subheader("⚙️ Settings")
        words_per_line = st.slider("Words per Line", 3, 15, 5)
        total_lines = st.slider("Total Lines", 2, 10, 5)

    if st.button("✨ Generate", use_container_width=True):
        with st.spinner("Generating your Bayaan..."):
            poetry = generate_poetry(start_text, words_per_line, total_lines, model, word_to_index, index_to_word)

            if poetry:
                st.markdown("### 📝 Generated Poetry")
                st.markdown(f'<div class="poetry-output">{poetry.replace("\\n", "<br>")}</div>', unsafe_allow_html=True)

                st.markdown(get_download_link(
                    poetry,
                    f"Bayaan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    "📥 Download Poetry"
                ), unsafe_allow_html=True)

                save_to_history(poetry, start_text)

                if st.button("📋 Copy to Clipboard"):
                    st.write('<script>navigator.clipboard.writeText(`' + poetry + '`);</script>', unsafe_allow_html=True)
                    st.success("Copied to clipboard!")

with tab2:
    st.subheader("📚 Poetry History")
    history = load_history()

    if history:
        for idx, entry in enumerate(reversed(history)):
            with st.expander(f"🕒 {entry['date']} - Prompt: {entry['prompt'][:30]}..."):
                st.text_area("Poetry", entry['poetry'], height=150, key=f"history_{idx}")

                st.markdown(get_download_link(
                    entry['poetry'],
                    f"Bayaan_{entry['date'].replace(' ', '_')}.txt",
                    "📥 Download"
                ), unsafe_allow_html=True)
    else:
        st.info("No poetry history yet. Start generating your Bayaan!")

with tab3:
    st.subheader("📊 Poetry Stats")
    try:
        if 'poetry' in locals():
            stats1, stats2 = st.columns(2)

            with stats1:
                words = poetry.split()
                st.metric("Total Words", len(words))
                st.metric("Unique Words", len(set(words)))
                richness = (len(set(words)) / len(words) * 100) if words else 0
                st.metric("Vocabulary Richness", f"{richness:.1f}%")

            with stats2:
                lines = [line for line in poetry.split('\n') if line.strip()]
                st.metric("Total Lines", len(lines))
                avg_words = (len(words) / len(lines)) if lines else 0
                st.metric("Avg Words per Line", f"{avg_words:.1f}")
        else:
            st.info("Generate poetry first to view stats!")
    except Exception as e:
        st.error("Something went wrong with the analysis.")

# Footer
st.markdown("""
---
<p class="footer">
    Created with ❤️ by BayaanBot Team |
    <a href="https://github.com/shaiiikh/Nazam-Generator-using-GRU" target="_blank">Inspiration Repo</a>
</p>
""", unsafe_allow_html=True)
