import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from datetime import datetime
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.utils import load_history, save_to_history
import os

# Suppress TensorFlow warnings (CUDA, etc.)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Streamlit page config
st.set_page_config(
    page_title="BayaanBot - Roman Urdu Poetry Generator",
    page_icon="🖋️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling + tooltip fix
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #1A1A1D, #0D0D0D);
        color: #F0EAD6;
        font-family: 'Georgia', serif;
    }

    h1 {
        color: #D4AF37 !important;
    }

    .stTextInput > div > div > input {
        background-color: #262730;
        color: #F0EAD6;
    }

    .stButton > button {
        background: linear-gradient(90deg, #D4AF37, #FFD700);
        color: #0D0D0D;
        border: none;
        border-radius: 8px;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(212, 175, 55, 0.3);
    }

    .footer {
        position: relative;
        display: inline-block;
    }

    .footer span:hover::after {
        content: " Hassan Haseen & Sameen Muzaffar ";
        position: absolute;
        top: -30px;
        right: 0;  /* align it with the right edge */
        transform: translateX(0%);  /* no centering offset */
        background-color: #333;
        color: #fff;
        padding: 5px 10px;
        border-radius: 5px;
        white-space: nowrap;
        font-size: 0.8rem;
        opacity: 1;
        z-index: 10;
    }
    </style>
""", unsafe_allow_html=True)

# Cache the model and encoders
@st.cache_resource
def load_model_and_encoder():
    model = tf.keras.models.load_model("models/poetry_gru_model.h5", compile=False)
    with open("models/word_encoder.pkl", "rb") as f:
        word_encoder = pickle.load(f)

    word_to_index = {word: i for i, word in enumerate(word_encoder.classes_)}
    index_to_word = {i: word for word, i in word_to_index.items()}

    return model, word_to_index, index_to_word

# Generate poetry logic
def generate_poetry(start_text, words_per_line, total_lines, model, word_to_index, index_to_word):
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

        if len([w for w in generated_words if w != '\n']) % words_per_line == 0:
            generated_words.append('\n')

    poetry_lines = ' '.join(generated_words).strip().split('\n')
    cleaned_lines = [line.strip() for line in poetry_lines if line.strip()]
    formatted_poetry = '\n'.join(cleaned_lines)

    return formatted_poetry

# Load model and encoders
model, word_to_index, index_to_word = load_model_and_encoder()

# App Title & Subtitle
st.title("🖋️ BayaanBot")
st.markdown("##### Express your thoughts in Roman Urdu Poetry powered by AI")

# App Tabs
tab1, tab2, tab3 = st.tabs(["Generate Poetry", "History", "Analysis"])

# Tab 1: Generate Poetry
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("✍️ Compose Your Bayaan")
        start_text = st.text_input(
            "Starting Words",
            value="",
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

                # Display poetry with built-in copy button
                st.code(poetry, language=None)

                # Save generated poetry to history
                save_to_history(poetry, start_text)

# Tab 2: Poetry History
with tab2:
    st.subheader("📚 Poetry History")
    history = load_history()

    if history:
        for idx, entry in enumerate(reversed(history)):
            with st.expander(f"🕒 {entry['date']} - Prompt: {entry['prompt'][:30]}..."):
                st.text_area("Poetry", entry['poetry'], height=150, key=f"history_{idx}")
    else:
        st.info("No poetry history yet. Start generating your Bayaan!")

# Tab 3: Poetry Analysis
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

# Centered Footer with hover tooltip fixed
st.markdown("""
---
<p class="footer" style="text-align: center;">
    Created with ❤️ by <span>Team CodeRunners</span>
</p>
""", unsafe_allow_html=True)
