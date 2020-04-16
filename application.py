import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, render_template, request, url_for
from wtforms import Form, TextAreaField, validators
import sys, os

sys.path.append(os.pardir)

import pickle

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

label_emoji_mapping = {0: '😂', 1: '❤', 2: '😍', 3: '😭', 4: '😊', 5: '🙄', 6: '😩', 7: '🔥', 8: '🤔', 9: '💕', 10: '💯', 11: '😘', 12: '💀', 13: '✨', 14: '🙃', 15: '👀', 16: '😒', 17: '☺', 18: '😢', 19: '😳'}

lstm_clf = load_model('LSTM_model2.h5')


app = Flask(__name__)
app.config['SECRET_KEY'] = 'MLFun'

class ReviewForm(Form):
    text = TextAreaField(' ', [validators.DataRequired(), validators.length(min=10)])

@app.route('/')
@app.route('/index')
def index():
    form = ReviewForm(request.form)
    return render_template('form.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['text']
        print(review)
        sequences = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(sequences, padding = 'post', maxlen = 26)
        labels = np.argsort(-lstm_clf.predict(padded))[:, :3].reshape(-1, 1)
        emojis = []

        for x in labels[:,0]:
            emojis.append(label_emoji_mapping[x])

        emojis = '\t\t'.join(emojis)

        return render_template('results.html',
                                content=review,
                                prediction=emojis)
    return render_template('form.html', form=form)

if __name__ == "__main__":
  app.run()

