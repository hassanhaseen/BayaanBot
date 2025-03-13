import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
import pickle

def build_model(vocab_size, embedding_dim=100, rnn_units=256):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        GRU(rnn_units, return_sequences=True),
        GRU(rnn_units),
        Dense(vocab_size, activation='softmax')
    ])
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(model, X, y, epochs=50, batch_size=64):
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    return history

def save_model_and_tokenizer(model, tokenizer, history, model_path, tokenizer_path, history_path):
    model.save(model_path)
    
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
