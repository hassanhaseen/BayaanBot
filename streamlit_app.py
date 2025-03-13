import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from datetime import datetime
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.utils import load_history, save_to_history

# Page configuration
st.set_page_config(
    page_title="BayaanBot - Roman Urdu Poetry Generator",
    page_icon="🖋️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #1A1A1D, #0D0D0D);
        color: #F0EAD6;
        font-family: 'Georgia', serif;
    }

    h1 {
        color: #D4AF37 !important;
        font-size: 3.5rem !important;
    }

    h5 {
        color: #F0EAD6 !important;
        font-size: 1.3rem !important;
    }

    .subheader {
        color: #D4AF37 !important;
    }

    .stTextInput > div > div > input {
        background-color: #262730;
        color: #F0EAD6;
    }

    .stButton > button {
        background: linear-gradient(90deg, #D4AF37, #FFD700);
        color: #0D0D0D;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(212, 175, 55, 0.3);
    }

    .poetry-output {
        background: rgba(38, 39, 48, 0.8);
        color: #F0EAD6;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #D4AF3722;
        margin-top: 1.5rem;
        line-height: 2.2;
        font-size: 1.3rem;
        text-align: left;
        white-space: pre-wrap;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }

    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #808495;
        margin-top: 3rem;
        border-top: 1px solid #D4AF3722;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and encoder
@st.cache_resource
def load_model_and_encoder():
    try:
        model = tf.keras.models.load_model("models/poetry_gru_model.h5")
        with open("models/word_encoder.pkl", "rb") as f:
            word_encoder = pickle.load(f)

        word_to_index = {word: i for i, word in enumerate(word_encoder.classes_)}
        index_to_word = {i: word for word, i in word_to_index.items()}

        return model, word_to_index, index_to_word
    except Exception as e:
        st.error(f"Error loading model or encoder: {str(e)}")
        return None, None, None

# Generate poetry function
def generate_poetry(start_text, words_per_line, total_lines, model, word_to_index, index_to_word):
    try:
        generated_words = start_text.strip().split() if start_text else []

        while len([w for w in generated_words if w != '\n']) < (words_per_line * total_lines):
            encoded_input = [word_to_index.get(word, 0) for word in generated_words[-5:]]
            encoded_input = pad_sequences([encoded_input], maxlen=5, truncating="pre")

            predicted_probs = model.predict(encoded_input, verbose=0)
            predicted_index = np.argmax(predicted_probs, axis=-1)[0]
            next_word = index_to_word.get(predicted_index, "")

            if not next_word:
                continue

            generated_words.append(next_word)

            # Insert a line break after every N words
            if len([w for w in generated_words if w != '\n']) % words_per_line == 0:
                generated_words.append('\n')

        formatted_poetry = ' '.join(generated_words).replace(' \n ', '\n').strip()
        return formatted_poetry

    except Exception as e:
        st.error(f"🚫 Error generating poetry: {str(e)}")
        return ""

# Load model and encoder
model, word_to_index, index_to_word = load_model_and_encoder()

if not all([model, word_to_index, index_to_word]):
    st.error("⚠️ Failed to load required components.")
    st.stop()

# Header
st.title("🖋️ BayaanBot")
st.markdown("##### Express your thoughts in Roman Urdu Poetry powered by AI")

# Tabs
tab1, tab2, tab3 = st.tabs(["Generate Poetry", "History", "Analysis"])

# Tab 1 - Generate Poetry
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🎯 Start Your Nazam")
        start_text = st.text_input(
            "Starting Words",
            value="",  # Empty by default
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

                poetry_lines = poetry.strip().split('\n')
                formatted_poetry = "<br>".join([f"{line.strip()}" for line in poetry_lines])

                st.markdown(f"""
                    <div class="poetry-output">
                        {formatted_poetry}
                    </div>
                """, unsafe_allow_html=True)

                save_to_history(poetry, start_text)

                # Copy to Clipboard button (custom styled)
                copy_code = f"""
                <button onclick="navigator.clipboard.writeText(`{poetry}`)" 
                    style="
                        display: block;
                        margin-top: 1rem;
                        padding: 0.75rem 1.5rem;
                        background: linear-gradient(90deg, #D4AF37, #FFD700);
                        color: #0D0D0D;
                        border: none;
                        border-radius: 8px;
                        font-size: 1rem;
                        font-weight: 600;
                        cursor: pointer;
                        transition: all 0.3s ease;
                    "
                    onmouseover="this.style.transform='translateY(-2px)';"
                    onmouseout="this.style.transform='translateY(0)';"
                >
                    📋 Copy to Clipboard
                </button>
                """

                st.markdown(copy_code, unsafe_allow_html=True)

# Tab 2 - History
with tab2:
    st.subheader("📚 Poetry History")
    history = load_history()

    if history:
        for idx, entry in enumerate(reversed(history)):
            with st.expander(f"🕒 {entry['date']} - Prompt: {entry['prompt'][:30]}..."):
                st.text_area("Poetry", entry['poetry'], height=150, key=f"history_{idx}")
    else:
        st.info("No poetry history yet. Start generating your Bayaan!")

# Tab 3 - Analysis
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

# Footer (no credits)
st.markdown("""
---
<p class="footer">
    Created with ❤️ by BayaanBot Team
</p>
""", unsafe_allow_html=True)
