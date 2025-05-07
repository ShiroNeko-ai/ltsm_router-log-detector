from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

# Load model dan tokenizer
model = tf.keras.models.load_model('model_lstm.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

def extract_message(text):
    match = re.search(r": (.+)", text)
    return match.group(1).strip() if match else text

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_results = []
    if request.method == 'POST':
        logs = request.form['logs'].split('\n')
        cleaned_logs = [extract_message(log) for log in logs]
        seq = tokenizer.texts_to_sequences(cleaned_logs)
        padded = pad_sequences(seq, maxlen=30, padding='post')
        preds = model.predict(padded)

        for original, pred in zip(logs, preds):
            label = "Anomali" if pred[0] > 0.5 else "Normal"
            prediction_results.append((original, label, round(pred[0], 4)))

    return render_template('index.html', results=prediction_results)

if __name__ == '__main__':
    app.run(debug=True)
