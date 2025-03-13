import os
import json
from datetime import datetime

HISTORY_PATH = "history/poetry_history.json"  # Update if you store history elsewhere

def load_history():
    try:
        with open(HISTORY_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        if not os.path.exists('history'):
            os.makedirs('history')
        with open(HISTORY_PATH, 'w', encoding='utf-8') as f:
            json.dump([], f)
        return []

def save_to_history(poetry, prompt):
    history = load_history()
    history.append({
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'prompt': prompt,
        'poetry': poetry
    })
    with open(HISTORY_PATH, 'w', encoding='utf-8') as f:
        json.dump(history[-50:], f)  # Keep only last 50 entries
