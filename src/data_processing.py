import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

DATA_PATH = "data/roman_urdu_poetry.csv"

def load_dataset():
    df = pd.read_csv(DATA_PATH)
    # Assuming the column name is "Poetry"
    texts = df['Poetry'].astype(str).tolist()
    return texts

def preprocess_texts(texts, num_words=5000):
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, padding='post')
    
    return tokenizer, padded_sequences
