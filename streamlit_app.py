import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import os
from datetime import datetime
import json
import base64
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="Nazam Generator",
    page_icon="📜",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    /* Modern Dark Theme */
    .stApp {
        background: linear-gradient(to bottom right, #0E1117, #1A1C24);
    }
    
    /* Title and Headers */
    h1 {
        background: linear-gradient(90deg, #FF4B4B, #FF8F8F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
    }
    
    h5 {
        color: #B0B4BC !important;
        font-size: 1.2rem !important;
        font-weight: 400 !important;
        margin-bottom: 2rem !important;
    }
    
    .subheader {
        color: #FF4B4B !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        margin-bottom: 1rem !important;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input {
        background-color: #262730;
        color: white;
        border: 2px solid #FF4B4B22;
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #FF4B4B;
        box-shadow: 0 0 10px #FF4B4B44;
    }
    
    /* Sliders */
    .stSlider {
        padding: 1rem 0;
    }
    
    .stSlider > div > div > div > div {
        background-color: #FF4B4B;
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #FF4B4B, #FF8F8F);
        color: white;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(255, 75, 75, 0.3);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Poetry Output */
    .poetry-output {
        background: rgba(38, 39, 48, 0.8);
        color: #FFFFFF;
        padding: 2rem;
        border-radius: 12px;
        border: 2px solid #FF4B4B22;
        margin-top: 1.5rem;
        line-height: 1.8;
        font-size: 1.2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #262730;
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border: none;
        color: #B0B4BC;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #FF4B4B44, #FF8F8F44);
        color: white;
        border-radius: 8px;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #262730 !important;
        border-radius: 8px;
        border: 1px solid #FF4B4B22;
    }
    
    .streamlit-expanderContent {
        background-color: #1E1F24 !important;
        border-radius: 0 0 8px 8px;
        border: 1px solid #FF4B4B22;
        border-top: none;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        background: linear-gradient(90deg, #FF4B4B, #FF8F8F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    /* Download Link */
    a.download-button {
        display: inline-block;
        background: linear-gradient(90deg, #FF4B4B22, #FF8F8F22);
        color: #FF4B4B;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        text-decoration: none;
        font-weight: 600;
        margin-top: 1rem;
        border: 1px solid #FF4B4B44;
        transition: all 0.3s ease;
    }
    
    a.download-button:hover {
        background: linear-gradient(90deg, #FF4B4B, #FF8F8F);
        color: white;
        transform: translateY(-2px);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #FF4B4B !important;
    }
    
    /* Success Message */
    .success-message {
        background: linear-gradient(90deg, #00FF8822, #00FFB822);
        border: 1px solid #00FF88;
        color: #00FF88;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #808495;
        margin-top: 3rem;
        border-top: 1px solid #FF4B4B22;
    }
    
    .footer a {
        color: #FF4B4B;
        text-decoration: none;
        transition: color 0.3s ease;
    }
    
    .footer a:hover {
        color: #FF8F8F;
    }
    </style>
""", unsafe_allow_html=True)

# Define utility functions
def get_download_link(text, filename, link_text):
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:text/plain;base64,{b64}" download="{filename}" class="download-button">�� {link_text}</a>'

def load_history():
    try:
        with open('poetry_history.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        # Create file if it doesn't exist
        if not os.path.exists('poetry_history.json'):
            with open('poetry_history.json', 'w', encoding='utf-8') as f:
                json.dump([], f)
        return []

def save_to_history(poetry, prompt):
    history = load_history()
    history.append({
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'prompt': prompt,
        'poetry': poetry
    })
    with open('poetry_history.json', 'w', encoding='utf-8') as f:
        json.dump(history[-50:], f)  # Keep last 50 entries

# Load the model and encoder
@st.cache_resource
def load_model_and_encoder():
    try:
        model = tf.keras.models.load_model("poetry_gru_model.h5")
        
        with open("word_encoder.pkl", "rb") as f:
            word_encoder = pickle.load(f)
            
        word_to_index = {word: i for i, word in enumerate(word_encoder.classes_)}
        index_to_word = {i: word for word, i in word_to_index.items()}
        
        return model, word_to_index, index_to_word
    except Exception as e:
        st.error(f"Error loading model or encoder: {str(e)}")
        return None, None, None

def generate_nazam(start_text, words_per_line, total_lines, model, word_to_index, index_to_word):
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

# Load model and check
model, word_to_index, index_to_word = load_model_and_encoder()

if not all([model, word_to_index, index_to_word]):
    st.error("⚠️ Failed to load required components. Please check if all files exist.")
    st.stop()

# Header Section
st.title("✨ Nazam Generator")
st.markdown("##### Create beautiful Urdu poetry using AI")

# Add tabs for different sections
tab1, tab2, tab3 = st.tabs(["Generate", "History", "Analysis"])

with tab1:
    # Generate tab content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input Section
        st.subheader("🎯 Generate Your Poetry")
        start_text = st.text_input(
            "Starting Words",
            value="aek mein wafa nahi aur doosrey ko wafa ki ummeed hai",
            help="Enter some words in Roman Urdu to start your poetry"
        )

    with col2:
        # Parameters Section
        st.subheader("⚙️ Parameters")
        words_per_line = st.slider(
            "Words per Line",
            min_value=3,
            max_value=15,
            value=5,
            help="Number of words in each line of poetry"
        )
        total_lines = st.slider(
            "Total Lines",
            min_value=2,
            max_value=10,
            value=5,
            help="Number of lines in the generated poetry"
        )

    # Generate Button
    if st.button("🎨 Generate Poetry", use_container_width=True):
        with st.spinner("✍️ Creating your masterpiece..."):
            poetry = generate_nazam(start_text, words_per_line, total_lines, model, word_to_index, index_to_word)
            if poetry:
                st.markdown("### 📝 Generated Poetry")
                st.markdown('<div class="poetry-output">' + poetry.replace('\n', '<br>') + '</div>', 
                          unsafe_allow_html=True)
                
                # Add download button
                st.markdown(get_download_link(
                    poetry,
                    f"poetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    "📥 Download Poetry"
                ), unsafe_allow_html=True)
                
                # Save to history
                save_to_history(poetry, start_text)
                
                # Add copy button
                if st.button("📋 Copy to Clipboard"):
                    st.write('<script>navigator.clipboard.writeText(`' + poetry + '`);</script>', 
                            unsafe_allow_html=True)
                    st.success("Copied to clipboard!")

with tab2:
    # History tab content
    st.subheader("📚 Generation History")
    history = load_history()
    if history:
        for idx, entry in enumerate(reversed(history)):
            with st.expander(f"🕒 {entry['date']} - Prompt: {entry['prompt'][:30]}..."):
                st.text_area(
                    "Poetry", 
                    entry['poetry'], 
                    height=150,
                    key=f"history_poetry_{idx}"  # Add unique key for each text area
                )
                st.markdown(get_download_link(
                    entry['poetry'],
                    f"poetry_{entry['date'].replace(' ', '_')}.txt",
                    "📥 Download"
                ), unsafe_allow_html=True)
    else:
        st.info("No generation history yet. Create some poetry first!")

with tab3:
    # Analysis tab content
    st.subheader("📊 Poetry Analysis")
    try:
        if 'poetry' in locals():
            analysis_col1, analysis_col2 = st.columns(2)
            with analysis_col1:
                # Word count analysis
                words = poetry.split()
                st.metric("Total Words", len(words))
                unique_words = len(set(words))
                st.metric("Unique Words", unique_words)
                st.metric("Vocabulary Richness", f"{(unique_words/len(words)*100):.1f}%")
            
            with analysis_col2:
                # Line analysis
                lines = [line for line in poetry.split('\n') if line.strip()]
                st.metric("Total Lines", len(lines))
                avg_words_per_line = len(words)/len(lines) if lines else 0
                st.metric("Avg Words per Line", f"{avg_words_per_line:.1f}")
        else:
            st.info("Generate some poetry first to see the analysis!")
    except Exception as e:
        st.error("Please generate some poetry first to see the analysis.")

# About Section (at the bottom of all tabs)
with st.expander("ℹ️ About this Poetry Generator"):
    st.markdown("""
    ### How it Works
    This poetry generator uses a sophisticated Gated Recurrent Unit (GRU) neural network 
    trained on a vast collection of Roman Urdu poetry. The model learns patterns and 
    structures from existing poetry to generate new, unique verses.

    ### Tips for Best Results
    1. Start with meaningful Roman Urdu words
    2. Try different combinations of words per line
    3. Adjust total lines to get shorter or longer poems
    4. Experiment with different starting phrases

    ### Technical Details
    - Model: GRU (Gated Recurrent Unit)
    - Training Data: Roman Urdu Poetry Collection
    - Output: Generated poetry in Roman Urdu script
    """)

# Footer
st.markdown("""
---
<p style='text-align: center; color: #666;'>
    shaiiikh | 
    <a href="https://github.com/shaiiikh/Nazam-Generator-using-GRU" target="_blank">GitHub</a>
</p>
""", unsafe_allow_html=True)
